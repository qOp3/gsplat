"""
Frequency-map-guided 2D Gaussian image fitting (coarse / mid / fine).

Stage 1 (coarse): detect low-frequency, flat-color blobs via the wavelet
frequency map and cover each with one big 2D Gaussian (same-color-block
init). Only these coarse splats are optimized, with the loss masked to the
low-frequency pixels so they specialize on broad smooth regions.

Stage 2 (mid): freeze coarse. Detect regions where wavelet level L4 (the
coarsest *detail* band — rendered purple in `*_wl_dominant.png`) dominates,
and place one medium Gaussian at the centroid of each such region. Only
these mid splats are optimized, with the loss masked to the L4-dominant
pixels.

Stage 3 (fine): freeze coarse + mid, add many small Gaussians (sampled with
density proportional to the frequency map, so high-frequency regions get
more of them). Only the fine splats are optimized, with the loss computed
over the full image, to recover residual / high-frequency detail.

A PNG render is exported after each stage.

Usage (from gsplat/ repo root):
    python examples/freq_image_fitting.py \
        --img_path       ../white_model.jpg \
        --freq_map_path  ../frequency_maps/white_model_wavelet/white_model_wl_combined.png \
        --output_dir     ../results/freq_image_fitting
"""

import math
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization_2dgs
from splat_ply_io import save_2dgs_ply

SH_C0 = 0.28209479177387814


def image_to_tensor(path: Path) -> Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)  # (H, W, 3)


def load_freq_map(path: Path, H: int, W: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read frequency map: {path}")
    if img.shape != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def inv_sigmoid(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def detect_lowfreq_blobs(
    gt_image: np.ndarray,  # (H,W,3) float32 [0,1]
    freq_map: np.ndarray,  # (H,W) float32 [0,1]
    low_freq_threshold: float,
    gray_levels: int,
    min_blob_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find flat-color, low-frequency connected regions.

    Returns pixel-space (cx, cy), half-extents (half_w, half_h), RGB color,
    and area for each blob — one big Gaussian per blob.
    """
    H, W = freq_map.shape
    low_mask = (freq_map <= low_freq_threshold).astype(np.uint8)

    gray = (gt_image.mean(axis=-1) * 255).astype(np.uint8)
    bin_size = max(1, 256 // gray_levels)
    quant = (gray.astype(np.int32) // bin_size).astype(np.uint8)

    means, half_extents, colors, areas = [], [], [], []
    for level in range(gray_levels):
        level_mask = ((quant == level).astype(np.uint8)) * low_mask
        if level_mask.sum() == 0:
            continue
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            level_mask, connectivity=8
        )
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
            areas.append(area)

    return (
        np.array(means, dtype=np.float32).reshape(-1, 2),
        np.array(half_extents, dtype=np.float32).reshape(-1, 2),
        np.array(colors, dtype=np.float32).reshape(-1, 3),
        np.array(areas, dtype=np.float32).reshape(-1),
    )


def load_level_maps(freq_map_path: Path, H: int, W: int, num_levels: int = 4) -> list:
    """Load sibling per-level wavelet maps (L1..Ln) next to a `*_wl_combined.png`."""
    stem = freq_map_path.name.replace("_wl_combined.png", "")
    parent = freq_map_path.parent
    maps = []
    for i in range(1, num_levels + 1):
        p = parent / f"{stem}_wl_L{i}.png"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read level map: {p}")
        if img.shape != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        maps.append(img.astype(np.float32) / 255.0)
    return maps


def detect_dominant_level_blobs(
    level_maps: list,       # list of (H,W) float32 [0,1], index 0 = L1 (finest)
    gt_image: np.ndarray,   # (H,W,3) float32 [0,1]
    level_index: int,       # 0-based; e.g. 3 = L4 ("purple" in *_wl_dominant.png)
    detail_threshold: float,
    min_blob_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find connected regions where `level_index` is the dominant wavelet level.

    Returns pixel-space (cx, cy), half-extents (half_w, half_h), RGB color
    per blob, and the boolean dominance mask (for loss masking).
    """
    stack = np.stack(level_maps, axis=0)  # (L,H,W)
    dom = np.argmax(stack, axis=0)
    total = stack.max(axis=0)
    mask = ((dom == level_index) & (total > detail_threshold)).astype(np.uint8)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    means, half_extents, colors = [], [], []
    for lbl in range(1, n_labels):  # skip background label 0
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_blob_area:
            continue
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[lbl]
        color = gt_image[labels == lbl].mean(axis=0)
        means.append([cx, cy])
        half_extents.append([max(w, 3) / 2.0, max(h, 3) / 2.0])
        colors.append(color)

    return (
        np.array(means, dtype=np.float32).reshape(-1, 2),
        np.array(half_extents, dtype=np.float32).reshape(-1, 2),
        np.array(colors, dtype=np.float32).reshape(-1, 3),
        mask.astype(np.float32),
    )


def sample_fine_points(
    gt_image: np.ndarray,  # (H,W,3)
    freq_map: np.ndarray,  # (H,W)
    num_points: int,
    base_density: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample pixel coords with density biased toward high-frequency regions."""
    H, W = freq_map.shape
    prob = freq_map.flatten().astype(np.float64) + base_density
    prob /= prob.sum()
    idx = np.random.choice(H * W, size=num_points, replace=True, p=prob)
    ys, xs = np.unravel_index(idx, (H, W))
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    colors = gt_image[ys, xs]
    return coords, colors


class FreqGuidedTrainer:
    def __init__(self, gt_image: Tensor, freq_map: np.ndarray, device: torch.device):
        self.device = device
        self.gt_image = gt_image.to(device)
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.freq_map = freq_map

        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.depth = 8.0  # fixed camera distance; all splats initialized at world z=0

        self.K = torch.tensor(
            [[self.focal, 0, self.W / 2], [0, self.focal, self.H / 2], [0, 0, 1]],
            device=device,
        )
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, self.depth],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        self.background = torch.zeros(3, device=device)

    def px_to_world(self, px: np.ndarray) -> np.ndarray:
        """(N,2) pixel coords -> (N,3) world coords at fixed world_z=0."""
        x = (px[:, 0] - self.W / 2) * self.depth / self.focal
        y = (px[:, 1] - self.H / 2) * self.depth / self.focal
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=1)

    def px_extent_to_world_scale(self, half_extent_px: np.ndarray) -> np.ndarray:
        """(N,2) pixel half-extents -> (N,2) world-space scale."""
        return half_extent_px * self.depth / self.focal

    def render(self, means, quats, scales, opacities, colors) -> Tensor:
        out, *_ = rasterization_2dgs(
            means,
            quats / quats.norm(dim=-1, keepdim=True),
            torch.exp(scales),
            torch.sigmoid(opacities),
            torch.sigmoid(colors)[None],  # add camera dim: (N,D) -> (C=1,N,D)
            self.viewmat[None],
            self.K[None],
            self.W,
            self.H,
            packed=False,
            backgrounds=self.background[None],
        )
        return out[0]

    def save_png(self, img: Tensor, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(str(path))
        print(f"Saved: {path}")


def make_params(means_w, scales_w, colors, n, device, scale_z=0.02):
    N = means_w.shape[0]
    means = torch.tensor(means_w, dtype=torch.float32, device=device, requires_grad=True)
    scale3 = np.concatenate([scales_w, np.full((N, 1), scale_z, dtype=np.float32)], axis=1)
    scales_raw = torch.tensor(np.log(np.clip(scale3, 1e-4, None)), dtype=torch.float32, device=device, requires_grad=True)
    colors_raw = torch.tensor(inv_sigmoid(colors), dtype=torch.float32, device=device, requires_grad=True)
    opacities_raw = torch.full((N,), 3.0, dtype=torch.float32, device=device, requires_grad=True)  # sigmoid(3)=0.95
    quats = torch.zeros((N, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity
    quats.requires_grad = True
    return means, quats, scales_raw, opacities_raw, colors_raw


def main(
    img_path: Path,
    freq_map_path: Path,
    output_dir: Path = Path("results/freq_image_fitting"),
    low_freq_percentile: float = 40.0,
    gray_levels: int = 16,
    min_blob_area: int = 50,
    mid_level_index: int = 3,
    mid_detail_threshold: float = 0.05,
    min_mid_blob_area: int = 20,
    mid_iters: int = 300,
    num_fine_points: int = 20000,
    coarse_iters: int = 300,
    fine_iters: int = 1200,
    lr: float = 0.01,
    eval_every: int = 25,
    curve_output: Path = Path("results/freq_image_fitting/freq_curve.npz"),
    ply_output: Path = Path("results/freq_image_fitting/freq_guided.ply"),
) -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    np.random.seed(0)

    gt_image = image_to_tensor(img_path)
    H, W = gt_image.shape[0], gt_image.shape[1]
    gt_np = gt_image.numpy()
    freq_map = load_freq_map(freq_map_path, H, W)

    trainer = FreqGuidedTrainer(gt_image, freq_map, device)

    threshold = np.percentile(freq_map, low_freq_percentile)
    print(f"Low-freq threshold (p{low_freq_percentile:.0f}): {threshold:.4f}")

    # ── Stage 1: coarse init from low-freq color blobs ──────────────────────────
    px_means, px_half_ext, colors, areas = detect_lowfreq_blobs(
        gt_np, freq_map, threshold, gray_levels, min_blob_area
    )
    print(f"Coarse blobs detected: {len(px_means)}  (total area {areas.sum():.0f}px / {H*W}px)")

    means_w = trainer.px_to_world(px_means)
    scales_w = trainer.px_extent_to_world_scale(px_half_ext)
    c_means, c_quats, c_scales, c_opac, c_colors = make_params(
        means_w, scales_w, colors, len(px_means), device
    )

    low_mask = torch.from_numpy((freq_map <= threshold).astype(np.float32)).to(device)
    low_mask = low_mask.unsqueeze(-1)  # (H,W,1)

    steps_log, psnr_log = [], []

    def full_image_psnr(render_img: Tensor) -> float:
        mse = F.mse_loss(render_img, trainer.gt_image).item()
        return 10.0 * np.log10(1.0 / max(mse, 1e-12))

    torch.cuda.synchronize()
    t_start = time.time()

    optimizer = optim.Adam([c_means, c_quats, c_scales, c_opac, c_colors], lr=lr)
    for it in range(coarse_iters):
        render = trainer.render(c_means, c_quats, c_scales, c_opac, c_colors)
        loss = (((render - trainer.gt_image) ** 2) * low_mask).sum() / low_mask.sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it == 0 or (it + 1) % eval_every == 0:
            steps_log.append(it + 1)
            psnr_log.append(full_image_psnr(render.detach()))
        if (it + 1) % 50 == 0 or it == 0:
            print(f"[Stage1 coarse] iter {it+1}/{coarse_iters}  loss={loss.item():.6f}  full_psnr={psnr_log[-1]:.2f}")

    with torch.no_grad():
        stage1_render = trainer.render(c_means, c_quats, c_scales, c_opac, c_colors)
    trainer.save_png(stage1_render, output_dir / "stage1_coarse.png")

    # ── Stage 2: freeze coarse, add + train mid splats (dominant-L4/"purple") ───
    c_means_frozen = c_means.detach().clone()
    c_quats_frozen = c_quats.detach().clone()
    c_scales_frozen = c_scales.detach().clone()
    c_opac_frozen = c_opac.detach().clone()
    c_colors_frozen = c_colors.detach().clone()

    level_maps = load_level_maps(freq_map_path, H, W)
    mid_px, mid_half_ext, mid_colors, mid_mask_np = detect_dominant_level_blobs(
        level_maps, gt_np, mid_level_index, mid_detail_threshold, min_mid_blob_area
    )
    print(f"Mid (L{mid_level_index+1}-dominant) blobs detected: {len(mid_px)}")

    mid_means_w = trainer.px_to_world(mid_px)
    mid_scales_w = trainer.px_extent_to_world_scale(mid_half_ext)
    m_means, m_quats, m_scales, m_opac, m_colors = make_params(
        mid_means_w, mid_scales_w, mid_colors, len(mid_px), device, scale_z=0.01
    )

    mid_mask = torch.from_numpy(mid_mask_np).to(device).unsqueeze(-1)  # (H,W,1)

    optimizer = optim.Adam([m_means, m_quats, m_scales, m_opac, m_colors], lr=lr)
    for it in range(mid_iters):
        all_means = torch.cat([c_means_frozen, m_means], dim=0)
        all_quats = torch.cat([c_quats_frozen, m_quats], dim=0)
        all_scales = torch.cat([c_scales_frozen, m_scales], dim=0)
        all_opac = torch.cat([c_opac_frozen, m_opac], dim=0)
        all_colors = torch.cat([c_colors_frozen, m_colors], dim=0)

        render = trainer.render(all_means, all_quats, all_scales, all_opac, all_colors)
        loss = (((render - trainer.gt_image) ** 2) * mid_mask).sum() / mid_mask.sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_step = coarse_iters + it + 1
        if it == 0 or (it + 1) % eval_every == 0:
            steps_log.append(cum_step)
            psnr_log.append(full_image_psnr(render.detach()))
        if (it + 1) % 50 == 0 or it == 0:
            print(f"[Stage2 mid] iter {it+1}/{mid_iters}  loss={loss.item():.6f}  full_psnr={psnr_log[-1]:.2f}")

    with torch.no_grad():
        all_means = torch.cat([c_means_frozen, m_means], dim=0)
        all_quats = torch.cat([c_quats_frozen, m_quats], dim=0)
        all_scales = torch.cat([c_scales_frozen, m_scales], dim=0)
        all_opac = torch.cat([c_opac_frozen, m_opac], dim=0)
        all_colors = torch.cat([c_colors_frozen, m_colors], dim=0)
        stage2_render = trainer.render(all_means, all_quats, all_scales, all_opac, all_colors)
    trainer.save_png(stage2_render, output_dir / "stage2_mid.png")

    # ── Stage 3: freeze coarse + mid, add + train fine splats ───────────────────
    m_means_frozen = m_means.detach().clone()
    m_quats_frozen = m_quats.detach().clone()
    m_scales_frozen = m_scales.detach().clone()
    m_opac_frozen = m_opac.detach().clone()
    m_colors_frozen = m_colors.detach().clone()

    cm_means = torch.cat([c_means_frozen, m_means_frozen], dim=0)
    cm_quats = torch.cat([c_quats_frozen, m_quats_frozen], dim=0)
    cm_scales = torch.cat([c_scales_frozen, m_scales_frozen], dim=0)
    cm_opac = torch.cat([c_opac_frozen, m_opac_frozen], dim=0)
    cm_colors = torch.cat([c_colors_frozen, m_colors_frozen], dim=0)

    px_fine, fine_colors = sample_fine_points(gt_np, freq_map, num_fine_points)
    fine_means_w = trainer.px_to_world(px_fine)
    # small init scale: ~1 pixel radius in world units
    fine_scales_w = np.full((num_fine_points, 2), 1.0, dtype=np.float32)
    fine_scales_w = trainer.px_extent_to_world_scale(fine_scales_w)
    f_means, f_quats, f_scales, f_opac, f_colors = make_params(
        fine_means_w, fine_scales_w, fine_colors, num_fine_points, device, scale_z=0.005
    )

    optimizer = optim.Adam([f_means, f_quats, f_scales, f_opac, f_colors], lr=lr)
    for it in range(fine_iters):
        all_means = torch.cat([cm_means, f_means], dim=0)
        all_quats = torch.cat([cm_quats, f_quats], dim=0)
        all_scales = torch.cat([cm_scales, f_scales], dim=0)
        all_opac = torch.cat([cm_opac, f_opac], dim=0)
        all_colors = torch.cat([cm_colors, f_colors], dim=0)

        render = trainer.render(all_means, all_quats, all_scales, all_opac, all_colors)
        loss = F.mse_loss(render, trainer.gt_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_step = coarse_iters + mid_iters + it + 1
        if it == 0 or (it + 1) % eval_every == 0:
            steps_log.append(cum_step)
            psnr_log.append(full_image_psnr(render.detach()))
        if (it + 1) % 100 == 0 or it == 0:
            print(f"[Stage3 fine] iter {it+1}/{fine_iters}  loss={loss.item():.6f}  full_psnr={psnr_log[-1]:.2f}")

    torch.cuda.synchronize()
    train_time = time.time() - t_start
    total_iters = coarse_iters + mid_iters + fine_iters
    print(f"Training time (coarse+mid+fine): {train_time:.2f}s  ({total_iters/train_time:.1f} it/s)")

    with torch.no_grad():
        all_means = torch.cat([cm_means, f_means], dim=0)
        all_quats = torch.cat([cm_quats, f_quats], dim=0)
        all_scales = torch.cat([cm_scales, f_scales], dim=0)
        all_opac = torch.cat([cm_opac, f_opac], dim=0)
        all_colors = torch.cat([cm_colors, f_colors], dim=0)
        stage3_render = trainer.render(all_means, all_quats, all_scales, all_opac, all_colors)
    trainer.save_png(stage3_render, output_dir / "stage3_fine.png")

    save_2dgs_ply(
        ply_output,
        all_means,
        all_quats,
        all_scales,
        all_opac,
        torch.sigmoid(all_colors),
    )
    print(f"Saved PLY: {ply_output}  ({ply_output.stat().st_size/1024:.1f} KB)")

    curve_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        curve_output,
        steps=np.array(steps_log),
        psnr=np.array(psnr_log),
        coarse_iters=coarse_iters,
        mid_iters=mid_iters,
        train_time=train_time,
    )
    print(f"Saved PSNR curve: {curve_output}")

    print(
        f"\nDone. Coarse splats: {len(px_means)}  Mid splats: {len(mid_px)}  Fine splats: {num_fine_points}"
    )
    print(f"Outputs in: {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
