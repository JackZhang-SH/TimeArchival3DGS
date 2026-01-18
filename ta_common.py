#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ta_common.py — Shared utilities for Time-Archival 3DGS scripts.

This module intentionally keeps dependencies light where possible:
- `parse_frames` / `find_latest_iteration` do not require torch.
- COLMAP camera loading and feature alignment import torch lazily.

Keeping these helpers in one place helps avoid subtle drift between
ta_render.py / ta_test.py / ta_train.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import os
import re


def parse_frames(frames_arg: Optional[str], dataset_root: Optional[Path] = None) -> List[int]:
    """
    Parse a frames spec into a sorted unique list of integers.

    Supported:
      - None / "" / "all" (requires dataset_root, will enumerate frame_* folders)
      - "1-10"
      - "1,2,5-7"
      - tokens may optionally include "frame_" prefix (e.g., "frame_3")

    Args:
        frames_arg: frames spec string.
        dataset_root: optional path to dataset root (used for "all").

    Returns:
        Sorted list of frame indices.

    Raises:
        ValueError if the spec is invalid or dataset_root missing for "all".
    """
    if frames_arg is None:
        frames_arg = ""
    frames_arg = str(frames_arg).strip()
    if frames_arg == "" or frames_arg.lower() == "all":
        if dataset_root is None:
            raise ValueError("frames='all' requires dataset_root to enumerate frame_* folders.")
        if not dataset_root.exists():
            raise ValueError(f"Dataset root does not exist: {dataset_root}")
        frames: List[int] = []
        for p in sorted(dataset_root.glob("frame_*")):
            if p.is_dir():
                m = re.match(r"frame_(\d+)$", p.name)
                if m:
                    frames.append(int(m.group(1)))
        return sorted(set(frames))

    out: List[int] = []
    for raw in frames_arg.split(","):
        part = raw.strip()
        if not part:
            continue
        if part.startswith("frame_"):
            part = part[len("frame_"):]
        # range
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip(); b = b.strip()
            if a.startswith("frame_"): a = a[len("frame_"):]
            if b.startswith("frame_"): b = b[len("frame_"):]
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid frame range token: '{raw}'")
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(list(range(lo, hi + 1)))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid frame token: '{raw}'")
            out.append(int(part))

    return sorted(set(out))


def find_latest_iteration(model_path: Path) -> int:
    """
    Find the largest 'iteration_xxx' folder under `model_path/point_cloud`.

    Returns:
        iteration number (int), or -1 if nothing found.
    """
    p = model_path / "point_cloud"
    if not p.exists():
        return -1
    iters: List[int] = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                iters.append(int(d.name.split("_", 1)[1]))
            except Exception:
                pass
    return max(iters) if iters else -1


def _try_read_colmap_extrinsics(images_bin: Path, images_txt: Path):
    """Return (extr_map, used_binary: bool)."""
    # Lazy import so that parse_frames works without COLMAP deps
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
    try:
        return read_extrinsics_binary(str(images_bin)), True
    except Exception:
        return read_extrinsics_text(str(images_txt)), False


def _try_read_colmap_intrinsics(cameras_bin: Path, cameras_txt: Path):
    """Return (intr_map, used_binary: bool)."""
    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
    try:
        return read_intrinsics_binary(str(cameras_bin)), True
    except Exception:
        return read_intrinsics_text(str(cameras_txt)), False


def load_cam_from_colmap(
    dataset_root: Path,
    image_name: str,
    sparse_id: int = 0,
    znear: float = 0.01,
    zfar: float = 100.0,
):
    """
    Load a COLMAP camera for a given image name and return a MiniCam.

    Matching strategy:
      1) exact match (COLMAP images entry equals image_name)
      2) basename match (os.path.basename)

    Note: This function mirrors the typical 3DGS COLMAP loading convention:
      - R = qvec2rotmat(qvec).T
      - T = tvec
      - world_view = getWorld2View2(R, T, ...)
    """
    sp = dataset_root / "sparse" / str(sparse_id)
    if not sp.exists():
        raise FileNotFoundError(f"COLMAP sparse folder not found: {sp}")

    images_bin = sp / "images.bin"
    images_txt = sp / "images.txt"
    cameras_bin = sp / "cameras.bin"
    cameras_txt = sp / "cameras.txt"

    extr_map, _ = _try_read_colmap_extrinsics(images_bin, images_txt)
    intr_map, _ = _try_read_colmap_intrinsics(cameras_bin, cameras_txt)

    target_key = None
    for k, extr in extr_map.items():
        if extr.name == image_name:
            target_key = k
            break
    if target_key is None:
        bn = os.path.basename(image_name)
        for k, extr in extr_map.items():
            if os.path.basename(extr.name) == bn:
                target_key = k
                break
    if target_key is None:
        raise KeyError(f"Image named '{image_name}' not found in COLMAP images.")

    extr = extr_map[target_key]
    intr = intr_map[extr.camera_id]

    width, height = int(intr.width), int(intr.height)
    # --- robust override by real image size (avoid bogus COLMAP W/H) ---
    try:
        from PIL import Image
        img_path = dataset_root / "images" / image_name
        if not img_path.exists():
            import os as _os
            img_path = dataset_root / "images" / _os.path.basename(image_name)
        if img_path.exists():
            W_img, H_img = Image.open(str(img_path)).size
            if (W_img != width) or (H_img != height):
                sx = W_img / max(1, width)
                sy = H_img / max(1, height)
                fx *= sx
                fy *= sy
                width, height = int(W_img), int(H_img)
    except Exception:
        pass
    # --- end override ---

    model = intr.model
    if model == "SIMPLE_PINHOLE":
        fx = float(intr.params[0]); fy = fx
    elif model == "PINHOLE":
        fx = float(intr.params[0]); fy = float(intr.params[1])
    else:
        raise AssertionError(
            f"Unsupported COLMAP camera model: {model}. Use PINHOLE/SIMPLE_PINHOLE."
        )

    # Lazy imports
    import numpy as np
    import torch
    from scene.colmap_loader import qvec2rotmat
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    R = np.transpose(qvec2rotmat(extr.qvec)).astype(np.float32)
    T = np.asarray(extr.tvec, dtype=np.float32)

    world_view = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1)
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1)
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    cam_center = world_view.inverse()[3, :3]
    # ensure cuda tensors for renderer extension
    world_view = world_view.to(device="cuda", dtype=torch.float32).contiguous()
    full_proj  = full_proj.to(device="cuda", dtype=torch.float32).contiguous()

    cam = MiniCam(
        width=width,
        height=height,
        fovy=float(fovy),
        fovx=float(fovx),
        znear=float(znear),
        zfar=float(zfar),
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )
    # MiniCam already computes camera_center internally from world_view_transform.
    # Keep image_name for debugging/printing if needed.
    cam.image_name = image_name
    return cam
def align_features_for_merge(A_feats, B_feats, align: str = "none"):
    """
    Align B's feature distribution to A (used for merge A+B in ta_render/ta_test).

    align:
      - "none"
      - "meanstd": match per-channel mean and std
      - "affine":  fit per-channel affine using least squares (B -> A)

    Returns:
        aligned B features (same shape as B_feats)
    """
    import numpy as np

    align = (align or "none").lower()
    if align == "none":
        return B_feats
    if A_feats.shape != B_feats.shape:
        raise ValueError(f"Feature shape mismatch: A{A_feats.shape} vs B{B_feats.shape}")

    A = A_feats.astype(np.float64)
    B = B_feats.astype(np.float64)

    if align == "meanstd":
        eps = 1e-6
        A_mean = A.mean(axis=0); A_std = A.std(axis=0) + eps
        B_mean = B.mean(axis=0); B_std = B.std(axis=0) + eps
        Bout = (B - B_mean) * (A_std / B_std) + A_mean
        return Bout.astype(B_feats.dtype)

    if align == "affine":
        Bout = np.empty_like(B)
        X = np.vstack([B, np.ones((B.shape[0], 1), dtype=np.float64)])  # [N, D+1]
        for c in range(B.shape[1]):
            y = A[:, c]
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            Bout[:, c] = X @ w
        return Bout.astype(B_feats.dtype)

    raise ValueError(f"Unknown --feature_align mode: {align}")
