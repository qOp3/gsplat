"""Minimal 3DGS/2DGS-compatible PLY writer (no SH higher-order terms)."""

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def inv_sigmoid(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def save_2dgs_ply(
    path: Path,
    means: torch.Tensor,       # (N,3) world coords, raw
    quats: torch.Tensor,       # (N,4) raw (unnormalized OK), wxyz
    scales_raw: torch.Tensor,  # (N,3) log-scale (pre-exp)
    opacities_raw: torch.Tensor,  # (N,) logit (pre-sigmoid)
    colors: torch.Tensor,      # (N,3) post-activation RGB in [0,1] (already sigmoid'd)
) -> None:
    N = means.shape[0]
    means = means.detach().cpu().numpy().astype(np.float32)
    quats = quats.detach().cpu().numpy().astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    scales_raw = scales_raw.detach().cpu().numpy().astype(np.float32)
    opacities_raw = opacities_raw.detach().cpu().numpy().astype(np.float32)
    colors = colors.detach().cpu().numpy().astype(np.float32)
    f_dc = (colors - 0.5) / SH_C0  # encode as SH DC term, consistent with standard 3DGS PLY

    names = (
        ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    dtype = [(n, "f4") for n in names]
    arr = np.empty(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = opacities_raw
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales_raw[:, 0], scales_raw[:, 1], scales_raw[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(path))
