"""
Baseline (vanilla random-init) 2DGS image fitting, matched in total splat
count and total iterations to the frequency-guided pipeline, for a fair
side-by-side comparison.

Usage (from gsplat/ repo root):
    python examples/baseline_image_fit_eval.py \
        --img_path   ../white_model.jpg \
        --num_points 20069 \
        --iterations 1000 \
        --output     ../results/freq_image_fitting/baseline.png
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torch import optim

from gsplat import rasterization_2dgs
from image_fitting import SimpleTrainer, image_path_to_tensor
from splat_ply_io import save_2dgs_ply


def psnr_from_mse(mse: float) -> float:
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main(
    img_path: Path,
    num_points: int = 20069,
    iterations: int = 1500,
    lr: float = 0.01,
    eval_every: int = 25,
    output: Path = Path("results/freq_image_fitting/baseline.png"),
    curve_output: Path = Path("results/freq_image_fitting/baseline_curve.npz"),
    ply_output: Path = Path("results/freq_image_fitting/baseline.ply"),
) -> None:
    gt_image = image_path_to_tensor(img_path)
    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)

    K = torch.tensor(
        [[trainer.focal, 0, trainer.W / 2], [0, trainer.focal, trainer.H / 2], [0, 0, 1]],
        device=trainer.device,
    )

    def render():
        out, *_ = rasterization_2dgs(
            trainer.means,
            trainer.quats / trainer.quats.norm(dim=-1, keepdim=True),
            trainer.scales,
            torch.sigmoid(trainer.opacities),
            torch.sigmoid(trainer.rgbs)[None],  # add camera dim: (N,D) -> (C=1,N,D)
            trainer.viewmat[None],
            K[None],
            trainer.W,
            trainer.H,
            packed=False,
        )
        return out[0]

    optimizer = optim.Adam(
        [trainer.rgbs, trainer.means, trainer.scales, trainer.opacities, trainer.quats], lr
    )
    steps_log, psnr_log = [], []
    torch.cuda.synchronize()
    t_start = time.time()
    for it in range(iterations):
        out_img = render()
        loss = F.mse_loss(out_img, trainer.gt_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it == 0 or (it + 1) % eval_every == 0:
            p = psnr_from_mse(loss.item())
            steps_log.append(it + 1)
            psnr_log.append(p)
        if (it + 1) % 100 == 0 or it == 0:
            print(f"[Baseline] iter {it+1}/{iterations}  loss={loss.item():.6f}  psnr={psnr_log[-1]:.2f}")

    torch.cuda.synchronize()
    train_time = time.time() - t_start
    print(f"Training time: {train_time:.2f}s  ({iterations/train_time:.1f} it/s)")

    with torch.no_grad():
        final = render()

    output.parent.mkdir(parents=True, exist_ok=True)
    arr = (final.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(output))
    print(f"Saved baseline render: {output}")
    print(f"Splats: {num_points} (single uniform band, randomly initialized)")

    curve_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        curve_output, steps=np.array(steps_log), psnr=np.array(psnr_log), train_time=train_time
    )
    print(f"Saved PSNR curve: {curve_output}")

    # image_fitting.py's SimpleTrainer.scales is linear (passed directly to the
    # rasterizer, not exponentiated) — convert to log-scale for the PLY, which
    # follows the standard 3DGS convention of storing pre-exp scale.
    scales_log = torch.log(trainer.scales.detach().clamp_min(1e-6))
    save_2dgs_ply(
        ply_output,
        trainer.means,
        trainer.quats,
        scales_log,
        trainer.opacities,
        torch.sigmoid(trainer.rgbs),
    )
    print(f"Saved PLY: {ply_output}  ({ply_output.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    tyro.cli(main)
