# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import time
import copy
import importlib.util
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _alias_local_package(package_name: str, package_root: Path) -> None:
    if package_name in sys.modules:
        return
    init_file = package_root / "__init__.py"
    if not init_file.exists():
        return
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


_alias_local_package("gsplat_scene", REPO_ROOT / "libs" / "scene")
_alias_local_package("gsplat_stage", REPO_ROOT / "libs" / "stage")

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from gsplat.losses import (
    depth_l1_loss,
    l1_loss,
    opacity_reg_loss,
    scale_reg_loss,
    ssim_loss,
    total_variation_loss,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode
from gsplat_scene import GaussianScene
from gsplat_stage import Stage
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Resume training from a progressive checkpoint (e.g., a coarse-only .pt file).
    # Loads band splats and continues from the saved step; optimizer state is reset.
    resume_ckpt: Optional[str] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path: "interp", "ellipse", "spiral", or "raw" (use captured poses as-is)
    render_traj_path: str = "interp"

    # Dataset backend: "colmap" or "ncore"
    data_type: str = "colmap"
    # Path to the Mip-NeRF 360 dataset (colmap) or NCore v4 meta-JSON file (ncore)
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # --- NCore-specific options (only used when data_type="ncore") ---
    # Camera sensor IDs to load (auto-detected from sequence if empty)
    ncore_camera_ids: List[str] = field(default_factory=list)
    # Point cloud source IDs to load -- accepts lidar, radar, or native point cloud
    # source IDs (auto-detected from sequence if empty). Field name kept for backward compat.
    ncore_lidar_ids: List[str] = field(default_factory=list)
    # Temporal seek offset in seconds
    ncore_seek_offset_sec: Optional[float] = None
    # Clip duration in seconds (None = full sequence)
    ncore_duration_sec: Optional[float] = None
    # Maximum number of lidar init points
    ncore_max_lidar_points: int = 500_000
    # Generic-data key for lidar point RGB colors (fallback to gray if unavailable)
    ncore_lidar_color_generic_data_name: str = "rgb"
    # NCore component group names
    ncore_poses_component_group: str = "default"
    ncore_intrinsics_component_group: str = "default"
    ncore_masks_component_group: str = "default"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Progressive multi-scale training mode.
    progressive: bool = False
    stage_steps: Tuple[int, int, int] = (5000, 17000, 27000)
    coarse_init_scale_mult: float = 2.5
    mid_spawn_scale_mult: float = 0.6
    fine_spawn_scale_mult: float = 0.25
    coarse_ssim_lambda: float = 0.10
    mid_ssim_lambda: float = 0.20
    fine_ssim_lambda: float = 0.25
    coarse_res_scale: float = 0.67
    mid_res_scale: float = 0.75
    fine_res_scale: float = 1.0
    freeze_policy: str = "geometry_and_opacity"
    fine_absgrad: bool = True
    fine_grow_grad2d: float = 0.0008
    band_range_reg: float = 0.01
    overlap_reg: float = 0.0
    band_caps: Tuple[int, int, int] = (300000, 400000, 400000)
    spawn_topk: int = 50000
    child_init_opa: float = 0.05
    spawn_score_alpha: float = 1.0
    spawn_score_beta: float = 0.5
    spawn_score_gamma: float = 0.1
    # Weight for LoG-based high-frequency score in spawn selection.
    # Positive values bias spawning toward high-frequency image regions,
    # ensuring child Gaussians cover detail that the parent band cannot.
    spawn_score_delta: float = 0.3
    # Sigma values for the multi-scale Laplacian-of-Gaussian frequency map.
    log_sigma_scales: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    spawn_window: int = 3
    # Fraction of coarse splats to keep (by smallest scale) before spawning mid.
    # Large coarse splats are removed to avoid occluding finer-band splats.
    # Set to 1.0 to disable pruning.
    coarse_prune_keep_ratio: float = 0.6
    # Stop training after the coarse band stage (requires progressive=True).
    coarse_only: bool = False
    # Override training image directory (e.g., point to FFT low-freq output).
    # Camera poses and 3D points still come from the original data_dir.
    train_image_dir: Optional[str] = None
    # PLY file of a pre-downsampled point cloud to use for coarse band init.
    # When set (and progressive=True), the coarse band is initialized from
    # this PLY instead of the full COLMAP point cloud.
    coarse_init_ply: Optional[str] = None

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


PROGRESSIVE_BANDS = ("coarse", "mid", "fine")


def _make_log_kernel(sigma: float, size: int, device: torch.device) -> Tensor:
    """Laplacian-of-Gaussian kernel, shape [1, 1, size, size], zero-summed."""
    coords = torch.arange(size, device=device).float() - size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r2 = x**2 + y**2
    sigma2 = sigma**2
    kernel = (r2 - 2.0 * sigma2) / (sigma2**2) * torch.exp(-r2 / (2.0 * sigma2))
    kernel = kernel - kernel.mean()
    return kernel.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def compute_log_freq_map(
    image: Tensor,
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> Tensor:
    """
    Per-pixel high-frequency indicator via max LoG response across scales.
    image: [1, H, W, 3] or [H, W, 3], values in [0, 1].
    Returns [H, W] map in [0, 1], where 1 = high-frequency region.
    """
    img = image[0] if image.dim() == 4 else image          # [H, W, 3]
    gray = img.mean(dim=-1).unsqueeze(0).unsqueeze(0)      # [1, 1, H, W]
    max_resp = torch.zeros(gray.shape[2:], device=image.device)
    for sigma in sigmas:
        ksize = int(6 * sigma + 1) | 1
        kernel = _make_log_kernel(sigma, ksize, image.device)
        resp = F.conv2d(gray, kernel, padding=ksize // 2).abs()[0, 0]
        max_resp = torch.maximum(max_resp, resp)
    return max_resp / max_resp.max().clamp_min(1e-6)


def get_progressive_stage(step: int, cfg: Config) -> str:
    if step < cfg.stage_steps[0]:
        return "coarse"
    elif step < cfg.stage_steps[1]:
        return "mid"
    elif step < cfg.stage_steps[2]:
        return "fine"
    else:
        return "polish"


def create_optimizers_for_splats(
    splats: torch.nn.ParameterDict,
    lrs: Dict[str, float],
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    world_size: int = 1,
) -> Dict[str, torch.optim.Optimizer]:
    BS = batch_size * world_size
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    return {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, lr in lrs.items()
    }


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm" or init_type == "lidar":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or lidar")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = create_optimizers_for_splats(
        splats,
        {name: lr for name, _, lr in params},
        sparse_grad=sparse_grad,
        visible_adam=visible_adam,
        batch_size=batch_size,
        world_size=world_size,
    )
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.data_type == "ncore":
            from datasets.ncore import NCoreDataset, NCoreParser

            self.parser = NCoreParser(
                meta_json_path=cfg.data_dir,
                factor=1.0 / cfg.data_factor if cfg.data_factor > 1 else 1.0,
                test_every=cfg.test_every,
                camera_ids=cfg.ncore_camera_ids or None,
                lidar_ids=cfg.ncore_lidar_ids or None,
                seek_offset_sec=cfg.ncore_seek_offset_sec,
                duration_sec=cfg.ncore_duration_sec,
                max_lidar_points=cfg.ncore_max_lidar_points,
                lidar_color_generic_data_name=cfg.ncore_lidar_color_generic_data_name,
                poses_component_group=cfg.ncore_poses_component_group,
                intrinsics_component_group=cfg.ncore_intrinsics_component_group,
                masks_component_group=cfg.ncore_masks_component_group,
                normalize_world_space=cfg.normalize_world_space,
            )
            self.trainset = NCoreDataset(self.parser, split="train")
            self.valset = NCoreDataset(self.parser, split="val")
            self.ncore_camera_data = [
                self.parser.camera_render_data[cam_id]
                for cam_id in self.parser.camera_ids
            ]
            if (
                any(d.camera_model == "ftheta" for d in self.ncore_camera_data)
                and not cfg.with_eval3d
            ):
                print(
                    "[NCore] Warning: FTheta cameras detected; pass --with-eval3d True for correct results."
                )
        else:
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                load_exposure=cfg.load_exposure,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )
        if cfg.progressive and not isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError("Progressive training currently supports DefaultStrategy only.")
        if cfg.progressive and cfg.sparse_grad:
            raise ValueError("Progressive training does not support sparse_grad yet.")
        if cfg.progressive and cfg.visible_adam:
            raise ValueError("Progressive training does not support visible_adam yet.")
        if cfg.coarse_only and not cfg.progressive:
            raise ValueError("coarse_only=True requires progressive=True.")

        if cfg.train_image_dir is not None:
            self._override_train_image_paths(cfg.train_image_dir)

        if cfg.coarse_init_ply is not None:
            pts, rgb = self._load_points_from_ply(cfg.coarse_init_ply)
            print(
                f"[coarse_init_ply] Replacing parser points ({len(self.parser.points):,}) "
                f"with {len(pts):,} points from: {cfg.coarse_init_ply}"
            )
            self.parser.points = pts
            self.parser.points_rgb = rgb

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale
            * (cfg.coarse_init_scale_mult if cfg.progressive else 1.0),
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        self.progressive_active_band: Optional[str] = None
        self.progressive_stage: Optional[str] = None
        self.progressive_absgrad_enabled = False
        self.progressive_polish_warned = False
        self.progressive_spawn_scores: Dict[str, Tensor] = {}
        if cfg.progressive:
            self.band_splats: Dict[str, torch.nn.ParameterDict] = {
                "coarse": self.splats
            }
            self.band_optimizers: Dict[str, Dict[str, torch.optim.Optimizer]] = {
                "coarse": self.optimizers
            }
            self.band_strategies: Dict[str, DefaultStrategy] = {
                "coarse": self._make_band_strategy("coarse")
            }
            self.band_strategy_state: Dict[str, Dict[str, Any]] = {
                "coarse": self.band_strategies["coarse"].initialize_state(
                    scene_scale=self.scene_scale
                )
            }
            self.band_scale_stats: Dict[str, Tuple[float, float]] = {
                "coarse": self._band_scale_stats(self.band_splats["coarse"])
            }
            self.progressive_active_band = "coarse"
            self.progressive_stage = "coarse"
            self.splats = self.band_splats["coarse"]
            self.optimizers = self.band_optimizers["coarse"]
        self.scene = GaussianScene.from_splats(self.splats, id="scene")
        self.splats = self.scene.splats
        self.stage = Stage()
        self.stage.add_scene(self.scene, self.rasterize_splats)
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        if cfg.progressive:
            self.band_strategies["coarse"].check_sanity(
                self.band_splats["coarse"], self.band_optimizers["coarse"]
            )
        else:
            self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if cfg.progressive:
            self.strategy_state = None
        elif isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def _splat_lrs(self) -> Dict[str, float]:
        cfg = self.cfg
        lrs = {
            "means": cfg.means_lr * self.scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
        }
        if cfg.app_opt:
            lrs["features"] = cfg.sh0_lr
            lrs["colors"] = cfg.sh0_lr
        else:
            lrs["sh0"] = cfg.sh0_lr
            lrs["shN"] = cfg.shN_lr
        return lrs

    def _make_band_strategy(self, band: str) -> DefaultStrategy:
        assert isinstance(self.cfg.strategy, DefaultStrategy)
        strategy = copy.deepcopy(self.cfg.strategy)
        if band == "fine" and self.cfg.fine_absgrad:
            strategy.absgrad = True
            strategy.grow_grad2d = self.cfg.fine_grow_grad2d
        if band == "coarse":
            strategy.grow_scale3d *= self.cfg.coarse_init_scale_mult
            strategy.prune_scale3d *= self.cfg.coarse_init_scale_mult
        return strategy

    def _band_scale_stats(
        self, splats: Union[torch.nn.ParameterDict, Dict[str, Tensor]]
    ) -> Tuple[float, float]:
        scales = splats["scales"].detach()
        return scales.mean().item(), scales.std(unbiased=False).clamp_min(1e-6).item()

    def _override_train_image_paths(self, train_image_dir: str) -> None:
        """Replace parser image paths with images from train_image_dir.

        File names are matched by basename, so the override directory must
        contain images with the same filenames as the original dataset.
        """
        img_dir = Path(train_image_dir)
        if not img_dir.exists():
            raise ValueError(f"train_image_dir does not exist: {train_image_dir}")
        EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        name_to_path: Dict[str, str] = {}
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in EXTS:
                name_to_path[p.name] = str(p)
                name_to_path[p.stem] = str(p)  # stem fallback for extension mismatches
        new_paths = []
        for orig in self.parser.image_paths:
            orig_name = Path(orig).name
            orig_stem = Path(orig).stem
            if orig_name in name_to_path:
                new_paths.append(name_to_path[orig_name])
            elif orig_stem in name_to_path:
                new_paths.append(name_to_path[orig_stem])
            else:
                sample = sorted(name_to_path.keys())[:5]
                raise ValueError(
                    f"No matching image for '{orig_name}' in {train_image_dir}. "
                    f"Sample available: {sample}"
                )
        print(
            f"[train_image_dir] Overriding {len(new_paths)} image paths "
            f"with images from: {train_image_dir}"
        )
        self.parser.image_paths = new_paths

    def _load_points_from_ply(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read a point-cloud PLY (binary or ASCII) written by tools/downsample_pointcloud.py.
        Returns (N,3) float32 xyz and (N,3) uint8 rgb.
        """
        path = Path(ply_path)
        if not path.exists():
            raise ValueError(f"coarse_init_ply does not exist: {ply_path}")
        with open(path, "rb") as f:
            header: list[str] = []
            while True:
                line = f.readline().decode("ascii").strip()
                header.append(line)
                if line == "end_header":
                    break
            n = int(next(l.split()[-1] for l in header if l.startswith("element vertex")))
            is_binary = any("binary_little_endian" in l for l in header)
            if is_binary:
                dt = np.dtype([
                    ("x", np.float32), ("y", np.float32), ("z", np.float32),
                    ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
                ])
                data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            else:
                rows = [f.readline().decode("ascii").split() for _ in range(n)]
                arr = np.array(rows, dtype=np.float32)
                dt = None
                data = arr
        if is_binary:
            points = np.stack([data["x"], data["y"], data["z"]], axis=1).copy().astype(np.float32)
            colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).copy().astype(np.uint8)
        else:
            points = data[:, :3].astype(np.float32)
            colors = data[:, 3:6].astype(np.uint8)
        return points, colors

    def _band_counts(self) -> Dict[str, int]:
        if not self.cfg.progressive:
            return {"total": len(self.splats["means"])}
        counts = {
            band: len(self.band_splats[band]["means"])
            for band in PROGRESSIVE_BANDS
            if band in self.band_splats
        }
        counts["total"] = sum(counts.values())
        return counts

    def freeze_band(self, band: str, geometry_only: bool = False) -> None:
        if band not in self.band_splats:
            return
        geometry_keys = {"means", "scales", "quats"}
        if self.cfg.freeze_policy == "geometry_and_opacity":
            geometry_keys.add("opacities")
        for name, param in self.band_splats[band].items():
            if not geometry_only or name in geometry_keys:
                param.requires_grad = False
                param.grad = None

    def unfreeze_band(self, band: str, appearance_only: bool = False) -> None:
        if band not in self.band_splats:
            return
        appearance_keys = {"sh0", "shN", "colors", "features"}
        for name, param in self.band_splats[band].items():
            param.requires_grad = (name in appearance_keys) if appearance_only else True

    def build_merged_splats(
        self, active_band: Optional[str]
    ) -> Tuple[Dict[str, Tensor], Dict[str, slice]]:
        assert self.cfg.progressive
        existing_bands = [band for band in PROGRESSIVE_BANDS if band in self.band_splats]
        assert existing_bands, "Progressive rendering requires at least one band."
        keys = list(self.band_splats[existing_bands[0]].keys())
        merged: Dict[str, Tensor] = {}
        band_slices: Dict[str, slice] = {}
        offset = 0
        for band in existing_bands:
            splats = self.band_splats[band]
            assert set(splats.keys()) == set(keys), (
                f"Band {band} keys {set(splats.keys())} do not match {set(keys)}."
            )
            n = splats["means"].shape[0]
            for key in keys:
                assert splats[key].shape[0] == n, (
                    f"Band {band} key {key} has mismatched first dimension."
                )
            band_slices[band] = slice(offset, offset + n)
            offset += n

        for key in keys:
            tensors = []
            for band in existing_bands:
                value = self.band_splats[band][key]
                tensors.append(value if band == active_band else value.detach())
            merged[key] = torch.cat(tensors, dim=0)
        return merged, band_slices

    def refresh_progressive_scene(self, active_band: Optional[str]) -> None:
        if not self.cfg.progressive:
            return
        self.progressive_active_band = active_band
        self.splats, self.progressive_band_slices = self.build_merged_splats(
            active_band=active_band
        )
        self.scene.splats = self.splats

    @torch.no_grad()
    def _sample_residual_at_centers(
        self,
        residual_map: Tensor,
        centers: Tensor,
        valid: Tensor,
        width: int,
        height: int,
    ) -> Tensor:
        scores = torch.zeros(centers.shape[0], device=centers.device)
        if centers.numel() == 0:
            return scores
        xy = centers.detach()
        if xy.abs().max() <= 2.0:
            xs = ((xy[:, 0] + 1.0) * 0.5 * (width - 1)).round().long()
            ys = ((xy[:, 1] + 1.0) * 0.5 * (height - 1)).round().long()
        else:
            xs = xy[:, 0].round().long()
            ys = xy[:, 1].round().long()
        in_bounds = valid & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not in_bounds.any():
            return scores
        half_window = max(self.cfg.spawn_window // 2, 0)
        valid_ids = torch.where(in_bounds)[0]
        for idx in valid_ids.tolist():
            x = xs[idx].item()
            y = ys[idx].item()
            x0 = max(0, x - half_window)
            x1 = min(width, x + half_window + 1)
            y0 = max(0, y - half_window)
            y1 = min(height, y + half_window + 1)
            scores[idx] = residual_map[y0:y1, x0:x1].mean()
        return scores

    @torch.no_grad()
    def update_spawn_scores(
        self,
        band: str,
        colors: Tensor,
        pixels: Tensor,
        info: Dict[str, Any],
    ) -> None:
        if band not in self.band_splats:
            return
        band_slice = self.progressive_band_slices[band]
        start = 0 if band_slice.start is None else band_slice.start
        end = band_slice.stop
        assert end is not None
        active_n = end - start
        height, width = pixels.shape[1:3]
        residual_map = l1_loss(colors.detach(), pixels.detach()).mean(dim=-1)[0]
        residual_score = torch.zeros(active_n, device=colors.device)
        gradient_score = torch.zeros(active_n, device=colors.device)
        freq_score = torch.zeros(active_n, device=colors.device)

        # LoG frequency map: high values indicate fine-detail regions that
        # the current band (large Gaussians) cannot resolve well. Used to
        # bias spawning of the next-finer band toward those regions.
        freq_map: Optional[Tensor] = None
        if self.cfg.spawn_score_delta > 0.0:
            freq_map = compute_log_freq_map(
                pixels.detach(), self.cfg.log_sigma_scales
            )

        means2d = info.get("means2d")
        radii = info.get("radii")
        if isinstance(means2d, torch.Tensor) and isinstance(radii, torch.Tensor):
            grad_source = None
            if hasattr(means2d, "absgrad") and means2d.absgrad is not None:
                grad_source = means2d.absgrad
            elif means2d.grad is not None:
                grad_source = means2d.grad

            if self.cfg.packed:
                gids = info["gaussian_ids"]
                mask = (gids >= start) & (gids < end)
                if mask.any():
                    local_ids = gids[mask] - start
                    centers = means2d[mask]
                    visible = (radii[mask] > 0.0).all(dim=-1)
                    sampled = self._sample_residual_at_centers(
                        residual_map, centers, visible, width, height
                    )
                    counts = torch.zeros(active_n, device=colors.device)
                    residual_score.index_add_(0, local_ids, sampled)
                    counts.index_add_(0, local_ids, torch.ones_like(sampled))
                    residual_score = residual_score / counts.clamp_min(1.0)
                    if grad_source is not None:
                        grad_norm = grad_source[mask].norm(dim=-1)
                        grad_counts = torch.zeros(active_n, device=colors.device)
                        gradient_score.index_add_(0, local_ids, grad_norm)
                        grad_counts.index_add_(0, local_ids, torch.ones_like(grad_norm))
                        gradient_score = gradient_score / grad_counts.clamp_min(1.0)
                    if freq_map is not None:
                        freq_sampled = self._sample_residual_at_centers(
                            freq_map, centers, visible, width, height
                        )
                        freq_counts = torch.zeros(active_n, device=colors.device)
                        freq_score.index_add_(0, local_ids, freq_sampled)
                        freq_counts.index_add_(0, local_ids, torch.ones_like(freq_sampled))
                        freq_score = freq_score / freq_counts.clamp_min(1.0)
            else:
                centers = means2d[..., start:end, :].reshape(-1, active_n, 2)[0]
                visible = (radii[..., start:end, :] > 0.0).all(dim=-1).reshape(
                    -1, active_n
                )[0]
                residual_score = self._sample_residual_at_centers(
                    residual_map, centers, visible, width, height
                )
                if grad_source is not None:
                    gradient_score = grad_source[..., start:end, :].norm(dim=-1)
                    gradient_score = gradient_score.reshape(-1, active_n).mean(dim=0)
                if freq_map is not None:
                    freq_score = self._sample_residual_at_centers(
                        freq_map, centers, visible, width, height
                    )

        opacity_score = torch.sigmoid(self.band_splats[band]["opacities"].detach())
        if opacity_score.numel() != active_n:
            return
        score = (
            self.cfg.spawn_score_alpha * residual_score
            + self.cfg.spawn_score_beta * gradient_score
            + self.cfg.spawn_score_gamma * opacity_score.flatten()
            + self.cfg.spawn_score_delta * freq_score
        )
        self.progressive_spawn_scores[band] = torch.nan_to_num(score.detach(), nan=0.0)

    @torch.no_grad()
    def spawn_band_from_parent(
        self, new_band: str, parent_band: str, scale_mult: float
    ) -> None:
        if new_band in self.band_splats:
            return
        parent = self.band_splats[parent_band]
        parent_n = parent["means"].shape[0]
        cap = self.cfg.band_caps[PROGRESSIVE_BANDS.index(new_band)]
        child_n = min(parent_n, self.cfg.spawn_topk, cap)
        if child_n <= 0:
            raise ValueError(f"Cannot spawn {new_band}: parent {parent_band} is empty.")

        opacities = torch.sigmoid(parent["opacities"].detach()).flatten()
        spawn_scores = self.progressive_spawn_scores.get(parent_band)
        if spawn_scores is not None and spawn_scores.numel() == parent_n:
            scores = spawn_scores.to(opacities.device)
            score_source = "residual_grad_opacity"
        elif opacities.numel() == parent_n:
            scores = opacities
            score_source = "opacity"
        else:
            scores = torch.ones(parent_n, device=opacities.device)
            score_source = "uniform"
        scores = torch.nan_to_num(scores, nan=0.0)

        if child_n < parent_n:
            _, sel = torch.topk(scores, k=child_n, largest=True, sorted=False)
        else:
            sel = torch.arange(parent_n, device=opacities.device)

        child = torch.nn.ParameterDict()
        for key, value in parent.items():
            cloned = value.detach()[sel].clone()
            if key == "means":
                jitter = torch.randn_like(cloned) * torch.exp(parent["scales"].detach()[sel]) * 0.25
                cloned = cloned + jitter
            elif key == "scales":
                cloned = cloned + math.log(scale_mult)
            elif key == "opacities":
                child_logit = torch.logit(
                    torch.tensor(self.cfg.child_init_opa, device=cloned.device)
                ).item()
                cloned = torch.full_like(cloned, child_logit)
            child[key] = torch.nn.Parameter(cloned, requires_grad=True)

        self.band_splats[new_band] = child
        self.band_optimizers[new_band] = create_optimizers_for_splats(
            child,
            self._splat_lrs(),
            sparse_grad=False,
            visible_adam=self.cfg.visible_adam,
            batch_size=self.cfg.batch_size,
            world_size=self.world_size,
        )
        self.band_strategies[new_band] = self._make_band_strategy(new_band)
        self.band_strategies[new_band].check_sanity(
            self.band_splats[new_band], self.band_optimizers[new_band]
        )
        self.band_strategy_state[new_band] = self.band_strategies[
            new_band
        ].initialize_state(scene_scale=self.scene_scale)
        self.band_scale_stats[new_band] = self._band_scale_stats(child)
        selected_scores = scores[sel]
        print(
            f"[Progressive] Spawned {new_band} from {parent_band}: "
            f"{child_n} splats, scale_mult={scale_mult}, "
            f"score_source={score_source}, "
            f"score_mean={selected_scores.mean().item():.6f}, "
            f"score_max={selected_scores.max().item():.6f}"
        )

    @torch.no_grad()
    def prune_coarse_band(self) -> None:
        keep_ratio = self.cfg.coarse_prune_keep_ratio
        if keep_ratio >= 1.0:
            return
        splats = self.band_splats["coarse"]
        n = splats["means"].shape[0]
        # Sort by largest axis of each Gaussian's scale ellipsoid; keep smallest k.
        max_scale = splats["scales"].detach().exp().max(dim=-1).values
        k = max(1, int(n * keep_ratio))
        _, sel = torch.topk(max_scale, k=k, largest=False, sorted=False)
        sel, _ = sel.sort()

        new_splats = torch.nn.ParameterDict()
        for key, val in splats.items():
            new_splats[key] = torch.nn.Parameter(
                val.detach()[sel].clone(), requires_grad=True
            )
        self.band_splats["coarse"] = new_splats
        self.band_optimizers["coarse"] = create_optimizers_for_splats(
            new_splats,
            self._splat_lrs(),
            sparse_grad=False,
            visible_adam=self.cfg.visible_adam,
            batch_size=self.cfg.batch_size,
            world_size=self.world_size,
        )
        self.band_strategy_state["coarse"] = self.band_strategies[
            "coarse"
        ].initialize_state(scene_scale=self.scene_scale)
        self.band_scale_stats["coarse"] = self._band_scale_stats(new_splats)
        self.progressive_spawn_scores.pop("coarse", None)
        print(
            f"[Progressive] Pruned coarse band: {n} -> {sel.shape[0]} splats "
            f"(keep_ratio={keep_ratio:.2f}, removed {n - sel.shape[0]} large splats)"
        )

    def prepare_progressive_stage(self, step: int) -> Tuple[str, str]:
        stage = get_progressive_stage(step, self.cfg)
        active_band = "fine" if stage == "polish" else stage
        if stage == "mid" and "mid" not in self.band_splats:
            self.prune_coarse_band()
            self.freeze_band("coarse")
            self.spawn_band_from_parent("mid", "coarse", self.cfg.mid_spawn_scale_mult)
        elif stage == "fine" and "fine" not in self.band_splats:
            self.freeze_band("coarse")
            self.freeze_band("mid")
            parent = "mid" if "mid" in self.band_splats else "coarse"
            self.spawn_band_from_parent("fine", parent, self.cfg.fine_spawn_scale_mult)
        elif stage == "polish":
            for band in list(self.band_splats.keys()):
                self.freeze_band(band, geometry_only=True)
            if not self.progressive_polish_warned:
                print(
                    "[Progressive] Polish stage uses appearance-only gradients where "
                    "available; continuing with the fine band optimizer."
                )
                self.progressive_polish_warned = True
            if "fine" not in self.band_splats:
                parent = "mid" if "mid" in self.band_splats else "coarse"
                self.spawn_band_from_parent("fine", parent, self.cfg.fine_spawn_scale_mult)
            self.unfreeze_band("fine", appearance_only=True)

        if active_band == "mid":
            self.freeze_band("coarse")
            self.unfreeze_band("mid")
        elif active_band == "fine":
            self.freeze_band("coarse")
            self.freeze_band("mid")
            self.unfreeze_band("fine", appearance_only=(stage == "polish"))
        elif active_band == "coarse":
            self.unfreeze_band("coarse")

        self.progressive_stage = stage
        self.progressive_active_band = active_band
        self.optimizers = self.band_optimizers[active_band]
        self.progressive_absgrad_enabled = self.band_strategies[active_band].absgrad
        return stage, active_band

    def progressive_ssim_lambda(self, stage: str) -> float:
        if stage == "coarse":
            return self.cfg.coarse_ssim_lambda
        if stage == "mid":
            return self.cfg.mid_ssim_lambda
        return self.cfg.fine_ssim_lambda

    def progressive_res_scale(self, stage: str) -> float:
        if stage == "coarse":
            return self.cfg.coarse_res_scale
        if stage == "mid":
            return self.cfg.mid_res_scale
        return self.cfg.fine_res_scale

    def band_range_loss(self, band: str) -> Tensor:
        if self.cfg.band_range_reg <= 0.0:
            return self.band_splats[band]["scales"].sum() * 0.0
        factors = {
            "coarse": (1.8, 3.0),
            "mid": (0.6, 1.2),
            "fine": (0.15, 0.45),
        }
        mean_log_scale, _ = self.band_scale_stats[band]
        lo, hi = factors[band]
        min_log_scale = mean_log_scale + math.log(lo)
        max_log_scale = mean_log_scale + math.log(hi)
        log_scales = self.band_splats[band]["scales"]
        return (
            F.relu(min_log_scale - log_scales).mean()
            + F.relu(log_scales - max_log_scale).mean()
        )

    def retain_progressive_info_grad(self, info: Dict[str, Any]) -> None:
        key = self.band_strategies[
            self.progressive_active_band
        ].key_for_gradient  # type: ignore[index]
        assert key in info, f"{key} is required but missing."
        info[key].retain_grad()

    def filter_info_to_band(
        self, info: Dict[str, Any], band_slice: slice
    ) -> Dict[str, Any]:
        start = 0 if band_slice.start is None else band_slice.start
        end = band_slice.stop
        assert end is not None
        active_n = end - start
        total_n = self.splats["means"].shape[0]
        active_info = dict(info)
        if self.cfg.packed:
            gids = info["gaussian_ids"]
            mask = (gids >= start) & (gids < end)
            active_info["gaussian_ids"] = gids[mask] - start
            for key, value in info.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.shape[:1] == gids.shape[:1]
                    and key != "gaussian_ids"
                ):
                    active_info[key] = value[mask]
            gradient_key = self.band_strategies[
                self.progressive_active_band
            ].key_for_gradient  # type: ignore[index]
            if gradient_key in info and info[gradient_key].grad is not None:
                active_info[gradient_key].grad = info[gradient_key].grad[mask]
            if (
                gradient_key in info
                and hasattr(info[gradient_key], "absgrad")
                and info[gradient_key].absgrad is not None
            ):
                active_info[gradient_key].absgrad = info[gradient_key].absgrad[mask]
            assert active_info["gaussian_ids"].numel() <= gids.numel()
            if active_info["gaussian_ids"].numel() > 0:
                assert int(active_info["gaussian_ids"].min()) >= 0
                assert int(active_info["gaussian_ids"].max()) < active_n
            if "radii" in active_info:
                assert active_info["radii"].shape[0] == active_info["gaussian_ids"].shape[0]
            if gradient_key in active_info:
                assert active_info[gradient_key].shape[0] == active_info[
                    "gaussian_ids"
                ].shape[0]
        else:
            for key in [
                "means2d",
                "gradient_2dgs",
                "radii",
                "depths",
                "conics",
                "opacities",
            ]:
                value = info.get(key)
                if isinstance(value, torch.Tensor) and value.shape[-2] == total_n:
                    active_info[key] = value[..., start:end, :]
                    assert active_info[key].shape[-2] == active_n
            gradient_key = self.band_strategies[
                self.progressive_active_band
            ].key_for_gradient  # type: ignore[index]
            gradient_value = active_info.get(gradient_key)
            if (
                isinstance(gradient_value, torch.Tensor)
                and info[gradient_key].grad is not None
            ):
                gradient_value.grad = info[gradient_key].grad[..., start:end, :]
            if (
                isinstance(gradient_value, torch.Tensor)
                and hasattr(info[gradient_key], "absgrad")
                and info[gradient_key].absgrad is not None
            ):
                gradient_value.absgrad = info[gradient_key].absgrad[..., start:end, :]
            if "radii" in active_info:
                assert active_info["radii"].shape[-2] == active_n
            if isinstance(gradient_value, torch.Tensor):
                assert gradient_value.shape[-2] == active_n
        return active_info

    def save_progressive_checkpoint(self, step: int, stage: str) -> None:
        data = {
            "step": step,
            "progressive": True,
            "stage": stage,
            "bands": {
                band: self.band_splats[band].state_dict()
                for band in PROGRESSIVE_BANDS
                if band in self.band_splats
            },
            "cfg": {
                key: value
                if isinstance(value, (bool, int, float, str, type(None), list, tuple))
                else repr(value)
                for key, value in vars(self.cfg).items()
            },
        }
        if self.cfg.pose_opt:
            data["pose_adjust"] = (
                self.pose_adjust.module.state_dict()
                if self.world_size > 1
                else self.pose_adjust.state_dict()
            )
        if self.cfg.app_opt:
            data["app_module"] = (
                self.app_module.module.state_dict()
                if self.world_size > 1
                else self.app_module.state_dict()
            )
        if self.post_processing_module is not None:
            data["post_processing"] = self.post_processing_module.state_dict()
        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

    def load_progressive_checkpoint(self, ckpts: List[Dict[str, Any]]) -> None:
        ckpt = ckpts[0]
        if ckpt.get("progressive", False):
            self.band_splats = {}
            self.band_optimizers = {}
            self.band_strategies = {}
            self.band_strategy_state = {}
            self.band_scale_stats = {}
            for band, state in ckpt["bands"].items():
                splats = torch.nn.ParameterDict(
                    {
                        key: torch.nn.Parameter(value.to(self.device), requires_grad=True)
                        for key, value in state.items()
                    }
                )
                self.band_splats[band] = splats
                self.band_optimizers[band] = create_optimizers_for_splats(
                    splats,
                    self._splat_lrs(),
                    sparse_grad=False,
                    visible_adam=self.cfg.visible_adam,
                    batch_size=self.cfg.batch_size,
                    world_size=self.world_size,
                )
                self.band_strategies[band] = self._make_band_strategy(band)
                self.band_strategy_state[band] = self.band_strategies[
                    band
                ].initialize_state(scene_scale=self.scene_scale)
                self.band_scale_stats[band] = self._band_scale_stats(splats)
        else:
            self.band_splats = {"coarse": self.splats}
            for key in self.splats.keys():
                self.band_splats["coarse"][key].data = torch.cat(
                    [old_ckpt["splats"][key] for old_ckpt in ckpts]
                ).to(self.device)
            self.band_optimizers = {"coarse": self.optimizers}
            self.band_strategies = {"coarse": self._make_band_strategy("coarse")}
            self.band_strategy_state = {
                "coarse": self.band_strategies["coarse"].initialize_state(
                    scene_scale=self.scene_scale
                )
            }
            self.band_scale_stats = {
                "coarse": self._band_scale_stats(self.band_splats["coarse"])
            }
        self.refresh_progressive_scene(active_band=None)

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        splats: Optional[torch.nn.ParameterDict] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        splats = splats if splats is not None else self.splats
        means = splats["means"]  # [N, 3]
        # quats = F.normalize(splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        ftheta_coeffs = None
        radial_coeffs = None
        tangential_coeffs = None
        thin_prism_coeffs = None
        with_ut = self.cfg.with_ut

        if camera_idcs is not None and hasattr(self, "ncore_camera_data"):
            cam = self.ncore_camera_data[camera_idcs.item()]
            camera_model = cam.camera_model
            ftheta_coeffs = cam.ftheta_coeffs
            if cam.radial_coeffs is not None:
                radial_coeffs = (
                    torch.from_numpy(cam.radial_coeffs).to(means.device).unsqueeze(0)
                )
            if cam.tangential_coeffs is not None:
                tangential_coeffs = (
                    torch.from_numpy(cam.tangential_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )
            if cam.thin_prism_coeffs is not None:
                thin_prism_coeffs = (
                    torch.from_numpy(cam.thin_prism_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.progressive_absgrad_enabled
                if self.cfg.progressive
                else self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=with_ut,
            with_eval3d=self.cfg.with_eval3d,
            ftheta_coeffs=ftheta_coeffs,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        if cfg.coarse_only:
            max_steps = min(max_steps, cfg.stage_steps[0])
            print(f"[coarse_only] Training capped at {max_steps} steps (coarse stage only).")
        init_step = 0

        if cfg.resume_ckpt is not None:
            ckpt = torch.load(cfg.resume_ckpt, map_location=self.device, weights_only=True)
            if not ckpt.get("progressive", False):
                raise ValueError("resume_ckpt only supports progressive checkpoints.")
            self.load_progressive_checkpoint([ckpt])
            init_step = ckpt["step"] + 1
            print(
                f"[resume] Loaded checkpoint from {cfg.resume_ckpt} "
                f"(stage={ckpt.get('stage', '?')}, resuming at step {init_step})"
            )

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        if init_step > 0:
            # Fast-forward LR schedulers to match the resumed step.
            for scheduler in schedulers:
                for _ in range(init_step):
                    scheduler.step()

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            stage = None
            active_band = None
            if cfg.progressive:
                stage, active_band = self.prepare_progressive_stage(step)
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            if cfg.progressive:
                assert stage is not None
                res_scale = self.progressive_res_scale(stage)
                if res_scale != 1.0:
                    pixels = F.interpolate(
                        pixels.permute(0, 3, 1, 2),
                        scale_factor=res_scale,
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    Ks = Ks.clone()
                    Ks[:, :2, :] *= res_scale
                    if masks is not None:
                        masks = (
                            F.interpolate(
                                masks[:, None].float(),
                                size=pixels.shape[1:3],
                                mode="nearest",
                            )[:, 0]
                            > 0.5
                        )
                    if cfg.depth_loss:
                        points = points * res_scale

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            if cfg.progressive:
                assert active_band is not None
                self.refresh_progressive_scene(active_band=active_band)

            # forward
            renders, alphas, info = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if cfg.progressive and stage != "polish":
                assert active_band is not None
                self.retain_progressive_info_grad(info)
            elif not cfg.progressive:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            if masks is not None:
                # Exclude masked pixels (e.g. ego vehicle) from L1.
                # For SSIM (patch-based), zero out both sides at masked locations
                # so masked patches don't pull colors toward an arbitrary value.
                l1loss = l1_loss(colors[masks], pixels[masks]).mean()
                colors_ssim = colors * masks[..., None]
                pixels_ssim = pixels * masks[..., None]
            else:
                l1loss = l1_loss(colors, pixels).mean()
                colors_ssim = colors
                pixels_ssim = pixels
            ssimloss = ssim_loss(
                colors_ssim.permute(0, 3, 1, 2), pixels_ssim.permute(0, 3, 1, 2)
            )
            ssim_lambda = (
                self.progressive_ssim_lambda(stage)
                if cfg.progressive and stage is not None
                else cfg.ssim_lambda
            )
            loss = torch.lerp(l1loss, ssimloss, ssim_lambda)
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                depthloss = depth_l1_loss(
                    depths, depths_gt, scene_scale=self.scene_scale
                )
                loss += depthloss * cfg.depth_lambda
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                reg_splats = (
                    self.band_splats[active_band]
                    if cfg.progressive and active_band is not None
                    else self.splats
                )
                loss += cfg.opacity_reg * opacity_reg_loss(reg_splats["opacities"])
            if cfg.scale_reg > 0.0:
                reg_splats = (
                    self.band_splats[active_band]
                    if cfg.progressive and active_band is not None
                    else self.splats
                )
                loss += cfg.scale_reg * scale_reg_loss(reg_splats["scales"])
            band_range = None
            if cfg.progressive and active_band is not None:
                band_range = self.band_range_loss(active_band)
                loss += cfg.band_range_reg * band_range

            loss.backward()
            if cfg.progressive and active_band is not None and stage != "polish":
                self.update_spawn_scores(active_band, colors, pixels, info)

            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                counts = self._band_counts() if cfg.progressive else None
                self.writer.add_scalar(
                    "train/num_GS",
                    counts["total"] if counts is not None else len(self.splats["means"]),
                    step,
                )
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.progressive and counts is not None:
                    for band in PROGRESSIVE_BANDS:
                        self.writer.add_scalar(
                            f"train/num_GS_{band}", counts.get(band, 0), step
                        )
                    self.writer.add_scalar(
                        "train/band_range_loss",
                        band_range.item() if band_range is not None else 0.0,
                        step,
                    )
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                counts = self._band_counts()
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": counts["total"],
                }
                if cfg.progressive:
                    stats.update({f"num_GS_{k}": v for k, v in counts.items() if k != "total"})
                    stats["stage"] = stage
                    stats["active_band"] = active_band
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                if cfg.progressive:
                    assert stage is not None
                    self.save_progressive_checkpoint(step, stage)
                else:
                    data = {
                        "step": step,
                        "scene_id": self.scene.id,
                        "splats": self.splats.state_dict(),
                    }
                    if cfg.pose_opt:
                        if world_size > 1:
                            data["pose_adjust"] = self.pose_adjust.module.state_dict()
                        else:
                            data["pose_adjust"] = self.pose_adjust.state_dict()
                    if cfg.app_opt:
                        if world_size > 1:
                            data["app_module"] = self.app_module.module.state_dict()
                        else:
                            data["app_module"] = self.app_module.state_dict()
                    if self.post_processing_module is not None:
                        data["post_processing"] = self.post_processing_module.state_dict()
                    torch.save(
                        data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                    )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if cfg.progressive and stage != "polish":
                assert active_band is not None
                counts_before_strategy = self._band_counts()
                active_info = self.filter_info_to_band(
                    info, self.progressive_band_slices[active_band]
                )
                self.band_strategies[active_band].step_post_backward(
                    params=self.band_splats[active_band],
                    optimizers=self.band_optimizers[active_band],
                    state=self.band_strategy_state[active_band],
                    step=step,
                    info=active_info,
                    packed=cfg.packed,
                    scene=None,
                )
                counts_after_strategy = self._band_counts()
                for band in PROGRESSIVE_BANDS:
                    if band != active_band and band in self.band_splats:
                        assert counts_after_strategy[band] == counts_before_strategy[band], (
                            f"Frozen band {band} changed during {active_band} "
                            f"strategy update: {counts_before_strategy[band]} -> "
                            f"{counts_after_strategy[band]}"
                        )
                if world_rank == 0 and step % 100 == 0:
                    print(
                        "[Progressive] strategy counts "
                        f"before={counts_before_strategy} "
                        f"after={counts_after_strategy}"
                    )
            elif cfg.progressive:
                pass
            elif isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                    scene=self.scene,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                    scene=self.scene,
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

            if cfg.progressive and world_rank == 0 and step % 100 == 0:
                counts = self._band_counts()
                band_counts = ", ".join(
                    f"{band}={counts.get(band, 0)}" for band in PROGRESSIVE_BANDS
                )
                print(
                    f"[Progressive] step={step} stage={stage} active={active_band} "
                    f"{band_counts} total={counts['total']} loss={loss.item():.4f} "
                    f"band_range={band_range.item() if band_range is not None else 0.0:.6f} "
                    f"absgrad={self.progressive_absgrad_enabled}"
                )

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        if cfg.progressive:
            self.refresh_progressive_scene(active_band=None)

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": self._band_counts()["total"],
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        if cfg.progressive:
            self.refresh_progressive_scene(active_band=None)

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "raw":
            # Use captured poses as-is
            camtoworlds_all = camtoworlds_all[:, :3, :]  # [N, 3, 4]
        elif cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        cfg = self.cfg
        world_rank = self.world_rank
        if self.cfg.progressive:
            self.refresh_progressive_scene(active_band=None)

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if self.cfg.progressive:
            self.refresh_progressive_scene(active_band=None)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.stage.render(
            self.scene.id,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor(render_tab_state.backgrounds, device=self.device, dtype=torch.float32)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = self._band_counts()["total"]
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        if cfg.progressive:
            runner.load_progressive_checkpoint(ckpts)
        else:
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            runner.scene = GaussianScene.from_splats(runner.splats, id="scene")
            runner.splats = runner.scene.splats
            runner.stage = Stage()
            runner.stage.add_scene(runner.scene, runner.rasterize_splats)
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut and cfg.with_eval3d:
        print(
            "[Trainer] Note: with_ut=True + with_eval3d=True (full 3DGUT mode). "
            "DefaultStrategy is incompatible with eval3d; use MCMCStrategy (the `mcmc` subcommand)."
        )

    cli(main, cfg, verbose=True)
