#!/usr/bin/env python3
"""Evaluate vanilla or progressive 3DGS checkpoints.

Examples:
    python tools/eval_progressive.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --data-dir data/360_v2/garden --output-dir results/garden/progressive_eval

    python tools/eval_progressive.py --ckpt progressive.pt --data-dir data/scene \
        --render-mode coarse_mid --max-frames 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (REPO_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


BAND_ORDER = ("coarse", "mid", "fine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--data-dir", required=True, help="COLMAP dataset directory.")
    parser.add_argument("--output-dir", default="progressive_eval")
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--normalize-world-space", action="store_true", default=True)
    parser.add_argument(
        "--render-mode",
        choices=("coarse", "coarse_mid", "coarse_mid_fine", "all"),
        default="all",
        help="Which bands to render for progressive checkpoints.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all val views.")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--lpips-net", choices=("alex", "vgg"), default="alex")
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


def select_bands(
    bands: Dict[str, Dict[str, torch.Tensor]], render_mode: str
) -> Dict[str, Dict[str, torch.Tensor]]:
    if render_mode == "coarse":
        names = ("coarse",)
    elif render_mode == "coarse_mid":
        names = ("coarse", "mid")
    elif render_mode in {"coarse_mid_fine", "all"}:
        names = BAND_ORDER
    else:
        raise ValueError(render_mode)
    selected = {name: bands[name] for name in names if name in bands}
    if not selected:
        raise ValueError(f"No checkpoint bands match render mode {render_mode!r}.")
    return selected


def merge_bands(bands: Dict[str, Dict[str, torch.Tensor]], device: torch.device):
    keys = list(next(iter(bands.values())).keys())
    merged = {}
    for key in keys:
        tensors = [band[key] for band in bands.values() if key in band]
        if len(tensors) != len(bands):
            continue
        merged[key] = torch.cat(tensors, dim=0).to(device)
    return merged


def band_counts(bands: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, int]:
    counts = {name: int(state["means"].shape[0]) for name, state in bands.items()}
    counts["total"] = sum(counts.values())
    return counts


def prepare_colors(splats: Dict[str, torch.Tensor]):
    if "sh0" in splats and "shN" in splats:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        sh_degree = int(math.sqrt(colors.shape[1]) - 1)
        return colors, sh_degree
    if "colors" in splats:
        return torch.sigmoid(splats["colors"]), None
    raise ValueError(
        "Checkpoint does not contain SH colors or direct colors. "
        "App-opt feature-only checkpoints are not supported by this lightweight tool."
    )


def make_ssim_metric(device: torch.device):
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except Exception as exc:
        print(f"SSIM skipped: torchmetrics unavailable ({exc}).")
        return None
    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def make_lpips_metric(net_type: str, device: torch.device):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as exc:
        print(f"LPIPS skipped: torchmetrics LPIPS unavailable ({exc}).")
        return None
    try:
        return LearnedPerceptualImagePatchSimilarity(
            net_type=net_type, normalize=True
        ).to(device)
    except Exception as exc:
        print(f"LPIPS skipped: failed to initialize {net_type!r} ({exc}).")
        return None


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def image_to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    try:
        from datasets.colmap import Dataset, Parser
        from gsplat.rendering import rasterization
    except Exception as exc:
        raise ImportError(
            "Failed to import the local COLMAP parser or gsplat rasterizer. This "
            "tool currently evaluates COLMAP-style datasets and needs the same "
            "runtime dependencies as examples/simple_trainer.py."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "renders"
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    all_bands = load_checkpoint_bands(Path(args.ckpt))
    selected_bands = select_bands(all_bands, args.render_mode)
    splats = merge_bands(selected_bands, device)
    colors, sh_degree = prepare_colors(splats)

    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
        load_exposure=False,
    )
    valset = Dataset(parser, split="val")

    ssim_metric = make_ssim_metric(device)
    lpips_metric = make_lpips_metric(args.lpips_net, device)
    psnrs, ssims, lpips_values, render_times = [], [], [], []
    frame_count = len(valset) if args.max_frames <= 0 else min(args.max_frames, len(valset))

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Rendering bands: {list(selected_bands.keys())}")
    print(f"Splat counts: {band_counts(selected_bands)}")

    with torch.inference_mode():
        for frame_idx in range(frame_count):
            data = valset[frame_idx]
            pixels = data["image"].to(device).unsqueeze(0) / 255.0
            camtoworlds = data["camtoworld"].to(device).unsqueeze(0)
            Ks = data["K"].to(device).unsqueeze(0)
            height, width = pixels.shape[1:3]

            if device.type == "cuda":
                torch.cuda.synchronize()
            tic = time.time()
            rendered, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                packed=args.packed,
                sh_degree=sh_degree,
                render_mode="RGB",
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            render_times.append(time.time() - tic)
            rendered = rendered[..., :3].clamp(0.0, 1.0)

            psnrs.append(psnr(rendered, pixels).item())
            rendered_nchw = image_to_nchw(rendered)
            pixels_nchw = image_to_nchw(pixels)
            if ssim_metric is not None:
                ssims.append(ssim_metric(rendered_nchw, pixels_nchw).item())
            if lpips_metric is not None:
                try:
                    lpips_values.append(lpips_metric(rendered_nchw, pixels_nchw).item())
                except Exception as exc:
                    print(f"LPIPS skipped after evaluation error: {exc}")
                    lpips_metric = None

            if args.save_images:
                canvas = torch.cat([pixels, rendered], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(image_dir / f"val_{frame_idx:04d}.png", (canvas * 255).astype(np.uint8))

    stats = {
        "checkpoint": str(args.ckpt),
        "render_mode": args.render_mode,
        "bands": list(selected_bands.keys()),
        "splats": band_counts(selected_bands),
        "num_frames": frame_count,
        "psnr": mean_or_none(psnrs),
        "ssim": mean_or_none(ssims),
        "lpips": mean_or_none(lpips_values),
        "avg_render_time_sec": mean_or_none(render_times),
        "peak_cuda_mem_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else None
        ),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
