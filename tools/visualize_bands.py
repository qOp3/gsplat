#!/usr/bin/env python3
"""Export per-band PLY files and summary plots for progressive checkpoints.

Examples:
    python tools/visualize_bands.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --output-dir results/garden/bands

    python tools/visualize_bands.py --ckpt vanilla.pt --output-dir vanilla_bands

Output files:
    coarse.ply, mid.ply, fine.ply  -- one PLY per band
    summary.json                   -- per-band stats (count, scale, opacity)
    scale_histogram.png            -- scale distribution by band (requires matplotlib)
    opacity_histogram.png          -- opacity distribution by band (requires matplotlib)
    band_counts.png                -- splat count per band (requires matplotlib)

Args:
    --ckpt        checkpoint path (required); supports both progressive and vanilla checkpoints
    --output-dir  output directory (default: band_visualization)
    --skip-plots  skip matplotlib plots

Note: vanilla (non-progressive) checkpoints produce only coarse.ply containing all splats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (REPO_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


BAND_ORDER = ("coarse", "mid", "fine")


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", default="band_visualization")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_checkpoint_bands(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("progressive", False):
        return {
            band: {key: value.detach().cpu() for key, value in state.items()}
            for band, state in ckpt["bands"].items()
        }
    if "splats" not in ckpt:
        raise KeyError(f"{path} is neither a progressive nor vanilla splat checkpoint.")
    return {"coarse": {key: value.detach().cpu() for key, value in ckpt["splats"].items()}}


def sh_for_export(splats: Dict[str, torch.Tensor]):
    n = splats["means"].shape[0]
    if "sh0" in splats and "shN" in splats:
        return splats["sh0"], splats["shN"]
    if "colors" in splats:
        rgb = torch.sigmoid(splats["colors"]).unsqueeze(1)
        return rgb_to_sh(rgb), torch.empty((n, 0, 3), dtype=rgb.dtype)
    rgb = torch.full((n, 1, 3), 0.5, dtype=splats["means"].dtype)
    print("Warning: band has no SH/color tensors; exporting neutral gray colors.")
    return rgb_to_sh(rgb), torch.empty((n, 0, 3), dtype=rgb.dtype)


def band_summary(splats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    scales = torch.exp(splats["scales"].detach()).flatten()
    opacities = torch.sigmoid(splats["opacities"].detach()).flatten()
    return {
        "count": int(splats["means"].shape[0]),
        "scale_mean": float(scales.mean().item()),
        "scale_median": float(scales.median().item()),
        "scale_min": float(scales.min().item()),
        "scale_max": float(scales.max().item()),
        "opacity_mean": float(opacities.mean().item()),
    }


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib plots skipped: {exc}")
        return None
    return plt


def save_histogram(
    plt,
    values_by_band: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for band, values in values_by_band.items():
        if values.size:
            plt.hist(values, bins=80, alpha=0.45, label=band)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_band_counts(plt, counts: Dict[str, int], output_path: Path) -> None:
    names = list(counts.keys())
    values = [counts[name] for name in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, values)
    plt.title("Splat counts per band")
    plt.ylabel("splats")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fallback_export_ply(
    splats: Dict[str, torch.Tensor],
    sh0: torch.Tensor,
    output_path: Path,
) -> None:
    means = splats["means"].detach().cpu().numpy()
    scales = torch.exp(splats["scales"].detach()).cpu().numpy()
    opacities = torch.sigmoid(splats["opacities"].detach()).cpu().numpy()
    rgb = (sh0.squeeze(1) * 0.28209479177387814 + 0.5).clamp(0.0, 1.0)
    rgb = (rgb.cpu().numpy() * 255.0).astype(np.uint8)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {means.shape[0]}\n")
        for name in [
            "x",
            "y",
            "z",
            "scale_0",
            "scale_1",
            "scale_2",
            "opacity",
        ]:
            f.write(f"property float {name}\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, scale, opacity, color in zip(means, scales, opacities, rgb):
            f.write(
                f"{xyz[0]} {xyz[1]} {xyz[2]} "
                f"{scale[0]} {scale[1]} {scale[2]} {opacity} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def export_band_ply(band: str, splats: Dict[str, torch.Tensor], output_dir: Path) -> None:
    sh0, shN = sh_for_export(splats)
    output_path = output_dir / f"{band}.ply"
    try:
        from gsplat import export_splats

        export_splats(
            means=splats["means"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=str(output_path),
        )
    except Exception as exc:
        print(
            f"Warning: gsplat export_splats unavailable for {band} ({exc}); "
            "writing a simple ASCII PLY fallback."
        )
        fallback_export_ply(splats, sh0, output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bands = load_checkpoint_bands(Path(args.ckpt))
    ordered_bands = {
        band: bands[band]
        for band in (*BAND_ORDER, *sorted(set(bands) - set(BAND_ORDER)))
        if band in bands
    }

    summary = {}
    scale_values = {}
    opacity_values = {}
    for band, splats in ordered_bands.items():
        print(f"Exporting {band}: {splats['means'].shape[0]} splats")
        export_band_ply(band, splats, output_dir)
        summary[band] = band_summary(splats)
        scale_values[band] = torch.exp(splats["scales"]).flatten().numpy()
        opacity_values[band] = torch.sigmoid(splats["opacities"]).flatten().numpy()

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {output_dir / 'summary.json'}")

    if not args.skip_plots:
        plt = maybe_import_matplotlib()
        if plt is not None:
            save_histogram(
                plt,
                scale_values,
                "Scale distribution by band",
                "scale",
                output_dir / "scale_histogram.png",
            )
            save_histogram(
                plt,
                opacity_values,
                "Opacity distribution by band",
                "opacity",
                output_dir / "opacity_histogram.png",
            )
            save_band_counts(
                plt,
                {band: int(data["count"]) for band, data in summary.items()},
                output_dir / "band_counts.png",
            )
            print("Wrote matplotlib plots.")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
