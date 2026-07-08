"""
Depth-layered, coarse-then-fine 2D Gaussian image fitting.

Coarse splats are initialized from one of two color-block detectors and
pushed *back* in world-z (--coarse_z_offset); fine splats stay at z=0, i.e.
strictly in front, so alpha-compositing always resolves fine-over-coarse
where they overlap instead of an arbitrary same-depth tie -- fine splats are
free to be generated "on top of" the coarse layer rather than competing with
it for the same depth slot.

Stage 1 (coarse): detect flat color blocks and cover each with one big 2D
Gaussian, pushed back to z=coarse_z_offset. Two interchangeable detectors:
  - "meanshift": cv2.pyrMeanShiftFiltering + connected components.
  - "freq":      wavelet frequency-map low-freq threshold + connected
                 components (same detector as freq_image_fitting.py).
Trained with full-image MSE.

Stage 2 (fine): freeze the coarse splats, render them, compare against the
ground truth, and sample new z=0 splats with density proportional to the
residual error |coarse_render - gt| -- i.e. generate splats where the coarse
layer fits the image worst -- then train them (coarse frozen) to recover the
remaining detail.

A PNG render is exported after each stage, and the combined splats are
exported to a PLY.

Usage (from gsplat/ repo root):
    python examples/meanshift_image_fitting.py \
        --img_path   ../white_model.jpg \
        --output_dir ../results/meanshift_image_fitting/whiteModel

    python examples/meanshift_image_fitting.py \
        --coarse_method freq \
        --img_path   ../white_model.jpg \
        --freq_map_path ../frequency_maps/white_model_wavelet/white_model_wl_combined.png \
        --output_dir ../results/freq_image_fitting/whiteModel_layered
"""

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import Tensor, optim

# Make both the repo root (for `gsplat`) and this file's own directory (for
# sibling flat modules like `splat_ply_io`/`freq_image_fitting`) importable
# regardless of invocation style (`-m examples.x` vs `python examples/x.py`)
# or cwd -- same pattern simple_trainer.py uses.
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, EXAMPLES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gsplat import rasterization_2dgs  # noqa: E402
from splat_ply_io import save_2dgs_ply  # noqa: E402
from freq_image_fitting import (  # noqa: E402
    image_to_tensor, inv_sigmoid, make_params, FreqGuidedTrainer,
    detect_lowfreq_blobs, load_freq_map,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure  # noqa: E402


def meanshift_blobs(gt_image: np.ndarray, sp: float, sr: float, max_level: int, min_blob_area: int):
    """Mean-shift-filter the image, then connected-component the (quantized)
    result to recover one blob per color region -- same trick freq_image_fitting.py
    uses for its low-freq blobs, but keyed on mean-shift color rather than a
    frequency-map threshold.
    """
    img_u8 = (np.clip(gt_image, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    shifted = cv2.pyrMeanShiftFiltering(bgr, sp, sr, maxLevel=max_level)
    shifted_rgb = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)

    quant = (shifted_rgb.astype(np.int32) // 4)
    flat_id = quant[..., 0] * 10000 + quant[..., 1] * 100 + quant[..., 2]

    means, half_extents, colors = [], [], []
    for uid in np.unique(flat_id):
        region = (flat_id == uid).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
        for lbl in range(1, n_labels):  # skip background label 0
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_blob_area:
                continue
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[lbl]
            color = gt_image[labels == lbl].mean(axis=0)
            means.append([cx, cy])
            half_extents.append([max(w, 4) / 2.0, max(h, 4) / 2.0])
            colors.append(color)

    return (
        np.array(means, dtype=np.float32).reshape(-1, 2),
        np.array(half_extents, dtype=np.float32).reshape(-1, 2),
        np.array(colors, dtype=np.float32).reshape(-1, 3),
    )


def sample_fine_points(gt_image: np.ndarray, density_map: np.ndarray, num_points: int, base_density: float = 0.05):
    """Sample pixel coords with density biased toward high-residual regions."""
    H, W = density_map.shape
    prob = density_map.flatten().astype(np.float64) + base_density
    prob /= prob.sum()
    idx = np.random.choice(H * W, size=num_points, replace=True, p=prob)
    ys, xs = np.unravel_index(idx, (H, W))
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    colors = gt_image[ys, xs]
    return coords, colors


def main(
    img_path: Path,
    output_dir: Path = Path("results/meanshift_image_fitting"),
    coarse_method: Literal["meanshift", "freq"] = "meanshift",
    coarse_z_offset: float = 1.0,
    sp: float = 10.0,
    sr: float = 40.0,
    max_level: int = 1,
    min_blob_area: int = 30,
    freq_map_path: Optional[Path] = None,
    low_freq_percentile: float = 40.0,
    gray_levels: int = 16,
    coarse_iters: int = 300,
    num_fine_points: int = 20000,
    fine_iters: int = 1200,
    lr: float = 0.01,
    eval_every: int = 25,
    curve_output: Path = Path("results/meanshift_image_fitting/curve.npz"),
    ply_output: Path = Path("results/meanshift_image_fitting/meanshift_guided.ply"),
) -> dict:
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    np.random.seed(0)

    gt_image = image_to_tensor(img_path)
    H, W = gt_image.shape[0], gt_image.shape[1]
    gt_np = gt_image.numpy()

    trainer = FreqGuidedTrainer(gt_image, freq_map=None, device=device)

    steps_log, psnr_log = [], []

    def full_image_psnr(render_img: Tensor) -> float:
        mse = F.mse_loss(render_img, trainer.gt_image).item()
        return 10.0 * np.log10(1.0 / max(mse, 1e-12))

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def full_image_ssim(render_img: Tensor) -> float:
        pred = render_img.clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
        gt = trainer.gt_image.permute(2, 0, 1).unsqueeze(0)
        return ssim_metric(pred, gt).item()

    torch.cuda.synchronize()
    t_start = time.time()

    # ── Stage 1: coarse init from flat color blocks, pushed back in z ───────────
    if coarse_method == "meanshift":
        px_means, px_half_ext, colors = meanshift_blobs(gt_np, sp, sr, max_level, min_blob_area)
        print(f"Mean-shift blobs detected: {len(px_means)}  (sp={sp} sr={sr})")
    else:
        if freq_map_path is None:
            raise ValueError("--freq_map_path is required when --coarse_method=freq")
        freq_map = load_freq_map(freq_map_path, H, W)
        threshold = np.percentile(freq_map, low_freq_percentile)
        px_means, px_half_ext, colors, areas = detect_lowfreq_blobs(
            gt_np, freq_map, threshold, gray_levels, min_blob_area
        )
        print(f"Freq-map blobs detected: {len(px_means)}  (p{low_freq_percentile:.0f} threshold={threshold:.4f})")

    means_w = trainer.px_to_world(px_means)
    means_w[:, 2] = coarse_z_offset  # push coarse layer back so fine splats (z=0) render in front
    scales_w = trainer.px_extent_to_world_scale(px_half_ext)
    c_means, c_quats, c_scales, c_opac, c_colors = make_params(
        means_w, scales_w, colors, len(px_means), device
    )

    optimizer = optim.Adam([c_means, c_quats, c_scales, c_opac, c_colors], lr=lr)
    for it in range(coarse_iters):
        render = trainer.render(c_means, c_quats, c_scales, c_opac, c_colors)
        loss = F.mse_loss(render, trainer.gt_image)
        optimizer.zero_grad()
        loss.backward()
        c_means.grad[:, 2].zero_()  # pin depth: only x,y move, z stays at coarse_z_offset
        optimizer.step()
        if it == 0 or (it + 1) % eval_every == 0:
            steps_log.append(it + 1)
            psnr_log.append(full_image_psnr(render.detach()))
        if (it + 1) % 50 == 0 or it == 0:
            print(f"[Stage1 coarse] iter {it+1}/{coarse_iters}  loss={loss.item():.6f}  full_psnr={psnr_log[-1]:.2f}")

    with torch.no_grad():
        stage1_render = trainer.render(c_means, c_quats, c_scales, c_opac, c_colors)
    trainer.save_png(stage1_render, output_dir / "stage1_coarse.png")

    # ── Stage 2: freeze coarse, add + train fine splats (residual-guided) ───────
    c_means_frozen = c_means.detach().clone()
    c_quats_frozen = c_quats.detach().clone()
    c_scales_frozen = c_scales.detach().clone()
    c_opac_frozen = c_opac.detach().clone()
    c_colors_frozen = c_colors.detach().clone()

    residual = (stage1_render.detach() - trainer.gt_image).abs().mean(dim=-1).cpu().numpy()  # (H,W)
    px_fine, fine_colors = sample_fine_points(gt_np, residual, num_fine_points)
    fine_means_w = trainer.px_to_world(px_fine)
    fine_scales_w = trainer.px_extent_to_world_scale(np.full((num_fine_points, 2), 1.0, dtype=np.float32))
    f_means, f_quats, f_scales, f_opac, f_colors = make_params(
        fine_means_w, fine_scales_w, fine_colors, num_fine_points, device, scale_z=0.005
    )

    optimizer = optim.Adam([f_means, f_quats, f_scales, f_opac, f_colors], lr=lr)
    for it in range(fine_iters):
        all_means = torch.cat([c_means_frozen, f_means], dim=0)
        all_quats = torch.cat([c_quats_frozen, f_quats], dim=0)
        all_scales = torch.cat([c_scales_frozen, f_scales], dim=0)
        all_opac = torch.cat([c_opac_frozen, f_opac], dim=0)
        all_colors = torch.cat([c_colors_frozen, f_colors], dim=0)

        render = trainer.render(all_means, all_quats, all_scales, all_opac, all_colors)
        loss = F.mse_loss(render, trainer.gt_image)
        optimizer.zero_grad()
        loss.backward()
        f_means.grad[:, 2].zero_()  # pin depth: fine splats stay at z=0, strictly in front of coarse
        optimizer.step()
        cum_step = coarse_iters + it + 1
        if it == 0 or (it + 1) % eval_every == 0:
            steps_log.append(cum_step)
            psnr_log.append(full_image_psnr(render.detach()))
        if (it + 1) % 100 == 0 or it == 0:
            print(f"[Stage2 fine] iter {it+1}/{fine_iters}  loss={loss.item():.6f}  full_psnr={psnr_log[-1]:.2f}")

    torch.cuda.synchronize()
    train_time = time.time() - t_start
    total_iters = coarse_iters + fine_iters
    print(f"Training time (coarse+fine): {train_time:.2f}s  ({total_iters/train_time:.1f} it/s)")

    with torch.no_grad():
        all_means = torch.cat([c_means_frozen, f_means], dim=0)
        all_quats = torch.cat([c_quats_frozen, f_quats], dim=0)
        all_scales = torch.cat([c_scales_frozen, f_scales], dim=0)
        all_opac = torch.cat([c_opac_frozen, f_opac], dim=0)
        all_colors = torch.cat([c_colors_frozen, f_colors], dim=0)
        stage2_render = trainer.render(all_means, all_quats, all_scales, all_opac, all_colors)
    trainer.save_png(stage2_render, output_dir / "stage2_fine.png")

    save_2dgs_ply(
        ply_output, all_means, all_quats, all_scales, all_opac, torch.sigmoid(all_colors)
    )
    print(f"Saved PLY: {ply_output}  ({ply_output.stat().st_size/1024:.1f} KB)")

    curve_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        curve_output, steps=np.array(steps_log), psnr=np.array(psnr_log),
        coarse_iters=coarse_iters, train_time=train_time,
    )
    print(f"Saved PSNR curve: {curve_output}")

    print(f"\nDone. Coarse splats: {len(px_means)}  Fine splats: {num_fine_points}")
    print(f"Outputs in: {output_dir}")

    return {
        "coarse_method": coarse_method,
        "coarse_z_offset": coarse_z_offset,
        "n_coarse": len(px_means),
        "n_fine": num_fine_points,
        "psnr_coarse": psnr_log[steps_log.index(coarse_iters)],
        "ssim_coarse": full_image_ssim(stage1_render.detach()),
        "psnr_final": full_image_psnr(stage2_render.detach()),
        "ssim_final": full_image_ssim(stage2_render.detach()),
        "train_time": train_time,
        "ply_path": ply_output,
        "ply_size_kb": ply_output.stat().st_size / 1024,
        "final_render": (stage2_render.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
    }


if __name__ == "__main__":
    tyro.cli(main)
