#!/usr/bin/env python3
"""Downsample a COLMAP point cloud and export as PLY.

Reads the 3D points produced by COLMAP reconstruction, applies spatial
downsampling, and writes a point cloud PLY that can be passed to
simple_trainer.py via --coarse_init_ply.

Methods
-------
voxel  : Divide space into a uniform grid; keep one point per cell
         (the one closest to the cell centroid).  Fast, O(N).
fps    : Farthest-Point Sampling.  Best spatial coverage but O(N·K).
         Avoid for point clouds > 200 k points.
random : Uniform random subset.  Fast, not spatially uniform.

Examples
--------
# Voxel downsampling (recommended)
python tools/downsample_pointcloud.py \\
    --data_dir data/grape \\
    --output   data/grape/points_voxel.ply \\
    --method   voxel \\
    --voxel_size 0.02

# FPS to exactly 50 000 points
python tools/downsample_pointcloud.py \\
    --data_dir data/grape \\
    --output   data/grape/points_fps50k.ply \\
    --method   fps \\
    --n_points 50000

# Random subset of 100 000 points
python tools/downsample_pointcloud.py \\
    --data_dir data/grape \\
    --output   data/grape/points_rand100k.ply \\
    --method   random \\
    --n_points 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
for _p in (REPO_ROOT, EXAMPLES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Point cloud I/O (binary little-endian PLY, no extra dependencies)
# ---------------------------------------------------------------------------

def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write (N,3) float32 points and (N,3) uint8 colors to binary PLY."""
    n = len(points)
    dt = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
    ])
    data = np.empty(n, dtype=dt)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def read_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read binary PLY written by this tool. Returns (N,3) float32 and (N,3) uint8."""
    with open(path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n = int(next(l.split()[-1] for l in header if l.startswith("element vertex")))
        dt = np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
        ])
        raw = f.read(n * dt.itemsize)
        if len(raw) < n * dt.itemsize:
            raise ValueError(f"PLY file truncated: expected {n * dt.itemsize} bytes, got {len(raw)}")
        data = np.frombuffer(raw, dtype=dt)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).copy()
    colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).copy()
    return points, colors


# ---------------------------------------------------------------------------
# Downsampling methods
# ---------------------------------------------------------------------------

def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the point closest to each occupied voxel centroid."""
    voxel_ids = np.floor(points / voxel_size).astype(np.int64)
    # Encode (i,j,k) as a single integer for fast grouping
    lo = voxel_ids.min(axis=0)
    shifted = voxel_ids - lo
    dims = shifted.max(axis=0) + 1
    keys = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]

    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    points_sorted = points[order]
    colors_sorted = colors[order]

    # For each voxel, pick the point closest to the voxel centroid
    _, first = np.unique(keys_sorted, return_index=True)
    last = np.concatenate([first[1:], [len(keys_sorted)]])

    kept_pts, kept_rgb = [], []
    for s, e in zip(first, last):
        chunk_pts = points_sorted[s:e]
        centroid = chunk_pts.mean(axis=0)
        idx = np.argmin(np.linalg.norm(chunk_pts - centroid, axis=1))
        kept_pts.append(chunk_pts[idx])
        kept_rgb.append(colors_sorted[s + idx])

    return np.stack(kept_pts), np.stack(kept_rgb)


def fps_downsample(
    points: np.ndarray, colors: np.ndarray, n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Farthest-Point Sampling. O(N·K) — slow for large N."""
    n = len(points)
    if n_samples >= n:
        return points, colors
    selected = np.empty(n_samples, dtype=np.int64)
    dists = np.full(n, np.inf)
    selected[0] = np.random.randint(n)
    for i in range(1, n_samples):
        d = np.linalg.norm(points - points[selected[i - 1]], axis=1)
        np.minimum(dists, d, out=dists)
        selected[i] = np.argmax(dists)
        if i % 5000 == 0:
            print(f"  FPS {i}/{n_samples}", end="\r")
    print()
    return points[selected], colors[selected]


def random_downsample(
    points: np.ndarray, colors: np.ndarray, n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if n_samples >= n:
        return points, colors
    idx = np.random.choice(n, size=n_samples, replace=False)
    return points[idx], colors[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="COLMAP dataset root (must contain sparse/).")
    p.add_argument("--output", required=True, help="Output PLY path.")
    p.add_argument(
        "--method", choices=["voxel", "fps", "random"], default="voxel",
        help="Downsampling method (default: voxel).",
    )
    p.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size for --method voxel.")
    p.add_argument("--n_points", type=int, default=50_000, help="Target count for fps/random.")
    p.add_argument("--factor", type=int, default=1, help="Dataset downsample factor (passed to Parser).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    from datasets.colmap import Parser

    print(f"Loading COLMAP points from: {args.data_dir}")
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=False,
        test_every=8,
    )
    points: np.ndarray = parser.points.astype(np.float32)     # (N, 3)
    colors: np.ndarray = parser.points_rgb.astype(np.uint8)   # (N, 3)
    print(f"Original point cloud: {len(points):,} points")

    if args.method == "voxel":
        print(f"Voxel downsampling  voxel_size={args.voxel_size}")
        pts_out, rgb_out = voxel_downsample(points, colors, args.voxel_size)
    elif args.method == "fps":
        print(f"FPS downsampling  n_points={args.n_points:,}")
        pts_out, rgb_out = fps_downsample(points, colors, args.n_points)
    else:
        print(f"Random downsampling  n_points={args.n_points:,}")
        pts_out, rgb_out = random_downsample(points, colors, args.n_points)

    print(f"Downsampled: {len(pts_out):,} points  ({len(pts_out)/len(points)*100:.1f}% of original)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output, pts_out, rgb_out)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
