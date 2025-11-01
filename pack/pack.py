#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack.py — Pack ONE specified 3DGS model folder (e.g., model_frame_1) into a single `.pt`
with Morton sort and FP16 attributes. No auto‑scan of multiple frames.

Examples:
  # 1) Pack latest iteration's point_cloud.ply under the given model folder
  #    and write to the default path: <model>/packed/iter_<it>.pt
  python pack.py --model ..\\output_seq\\soccer\\model_frame_1

  # 2) Force a specific iteration
  python pack.py --model ..\\output_seq\\soccer\\model_frame_1 --iter 8000

  # 3) Custom output file path
  python pack.py --model ..\\output_seq\\soccer\\model_frame_1 --out_pt ..\\output_seq_packed\\soccer\\model_frame_1\\iter_8000.pt
"""
import argparse, math
from pathlib import Path
import numpy as np
import torch

# ---- Dependencies ----
try:
    from plyfile import PlyData
except Exception as e:
    raise SystemExit("Please install dependency first: pip install plyfile") from e


# ========== Utilities ==========
def _find_latest_iteration(model_dir: Path) -> int:
    """Return the largest iteration number under <model_dir>/point_cloud/iteration_XXXX, or -1 if none."""
    pc = model_dir / "point_cloud"
    if not pc.exists():
        return -1
    best = -1
    for d in pc.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                it = int(d.name.split("_")[1])
                best = max(best, it)
            except Exception:
                pass
    return best


def _get_fields_safe(verts, names, allow_missing=False, default_val=0.0, dtype=np.float32, shape=None):
    """Fetch multiple named columns from a PLY structured array, with optional fill for missing fields."""
    outs = []
    missing = []
    for n in names:
        if n in verts.dtype.names:
            outs.append(np.asarray(verts[n], dtype=dtype))
        else:
            missing.append(n)
            if not allow_missing:
                raise KeyError(f"PLY missing required field '{n}'")
            outs.append(np.full((len(verts),), default_val, dtype=dtype))
    arr = np.stack(outs, axis=1)
    if shape is not None:
        arr = arr.reshape(shape)
    if missing:
        print(f"[pack] WARN missing fields: {missing} -> filled with {default_val}")
    return arr


def _gather_sh(verts):
    """
    Gather spherical harmonic features in a layout compatible with common 3DGS forks.
    Returns: (sh_dc [N,3,1], sh_rest [N,3,K], L) where L is SH degree.
    """
    # DC
    dc_names = [f"f_dc_{i}" for i in range(3)]
    dc = _get_fields_safe(verts, dc_names, allow_missing=False, dtype=np.float32).reshape(-1, 3, 1)

    # REST: try f_rest_*; fallback features_rest_*
    rest_cols = [n for n in (verts.dtype.names or []) if n.startswith("f_rest_")]
    if not rest_cols:
        rest_cols = [n for n in (verts.dtype.names or []) if n.startswith("features_rest_")]

    if not rest_cols:
        rest = np.zeros((len(verts), 3, 0), dtype=np.float32)
        L = 0
    else:
        def _idx(name): return int(name.split("_")[-1])
        rest_cols.sort(key=_idx)
        R = np.stack([np.asarray(verts[n], dtype=np.float32) for n in rest_cols], axis=1)  # [N,K_total]
        K_total = R.shape[1]
        if K_total % 3 == 0:
            K = K_total // 3
            rest = R.reshape(len(verts), 3, K)
        else:
            # Rare/export-variant case: replicate channels to 3
            rest = np.tile(R[:, None, :], (1, 3, 1))
        L = int(round(math.sqrt(rest.shape[2] + 1) - 1))
    return dc, rest, L


def _morton3D_order(xyz: np.ndarray, qbits: int = 10):
    """Compute Morton sort order for 3D points (xyz in float32)."""
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)
    norm = (xyz - mins) / span
    qmax = (1 << qbits) - 1
    qi = np.clip((norm * qmax + 0.5).astype(np.uint32), 0, qmax)

    def expand(v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8 )) & 0x0300F00F
        v = (v | (v << 4 )) & 0x030C30C3
        v = (v | (v << 2 )) & 0x09249249
        return v

    codes = (expand(qi[:, 0]) << 0) | (expand(qi[:, 1]) << 1) | (expand(qi[:, 2]) << 2)
    order = np.argsort(codes, kind="mergesort")
    return order, mins.astype(np.float32), maxs.astype(np.float32)


# ========== Core ==========
def pack_model(model_dir: Path, out_pt: Path = None, iter_override: int = None):
    """
    Pack the given <model_dir> (e.g., .../model_frame_1) to a .pt file.
    If out_pt is None -> write to <model_dir>/packed/iter_<it>.pt
    If iter_override is provided -> use that iteration; else use latest iteration.
    """
    if not model_dir.exists():
        raise SystemExit(f"[pack] model folder not found: {model_dir}")

    it = int(iter_override) if iter_override is not None else _find_latest_iteration(model_dir)
    if it < 0:
        raise SystemExit(f"[pack] no 'point_cloud/iteration_xxxx' found under: {model_dir}")

    ply_path = model_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    if not ply_path.exists():
        raise SystemExit(f"[pack] missing PLY: {ply_path}")

    # Output path
    if out_pt is None:
        out_pt = model_dir / "packed" / f"iter_{it}.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pack] model : {model_dir}")
    print(f"[pack] iter  : {it}")
    print(f"[pack] ply   : {ply_path}")
    print(f"[pack] out   : {out_pt}")

    # ---- Read PLY
    ply = PlyData.read(str(ply_path))
    verts = ply["vertex"].data  # structured array

    # ---- Required attributes
    xyz = np.stack([np.asarray(verts[n], dtype=np.float32) for n in ("x", "y", "z")], axis=1)  # [N,3]

    # Scaling (aka scale or scaling)
    scale_names = [n for n in ("scale_0", "scale_1", "scale_2",
                               "scaling_0", "scaling_1", "scaling_2") if n in verts.dtype.names]
    if len(scale_names) >= 3:
        scaling = _get_fields_safe(verts, scale_names[:3], allow_missing=False, dtype=np.float32)
    else:
        scaling = _get_fields_safe(verts, ["scale_0", "scale_1", "scale_2"], allow_missing=True, dtype=np.float32)

    # Rotation quaternion (x,y,z,w)
    rot_names = [n for n in ("rot_0", "rot_1", "rot_2", "rot_3",
                             "rotation_0", "rotation_1", "rotation_2", "rotation_3") if n in verts.dtype.names]
    if len(rot_names) >= 4:
        rotation = _get_fields_safe(verts, rot_names[:4], allow_missing=False, dtype=np.float32)
    else:
        rotation = _get_fields_safe(verts, ["rot_0", "rot_1", "rot_2", "rot_3"], allow_missing=True, dtype=np.float32)

    # Opacity
    if "opacity" in verts.dtype.names:
        opacity = np.asarray(verts["opacity"], dtype=np.float32)[:, None]
    else:
        opacity = np.ones((len(verts), 1), dtype=np.float32)

    # SH
    sh_dc, sh_rest, L = _gather_sh(verts)

    # ---- Morton sort (stable mergesort)
    order, aabb_min, aabb_max = _morton3D_order(xyz, qbits=10)
    xyz      = xyz[order]
    scaling  = scaling[order]
    rotation = rotation[order]
    opacity  = opacity[order]
    sh_dc    = sh_dc[order]
    sh_rest  = sh_rest[order]

    # ---- Package
    pkg = {
        "n": int(xyz.shape[0]),
        "sh_degree": int(L),
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "xyz": torch.from_numpy(xyz).to(torch.float32),
        "scaling": torch.from_numpy(scaling).to(torch.float16),
        "rotation": torch.from_numpy(rotation).to(torch.float16),
        "opacity": torch.from_numpy(opacity).to(torch.float16),
        "sh_dc": torch.from_numpy(sh_dc).to(torch.float16),
        "sh_rest": torch.from_numpy(sh_rest).to(torch.float16),
    }
    torch.save(pkg, out_pt)
    print(f"[pack] saved {out_pt} (N={pkg['n']}, L={pkg['sh_degree']})")


# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser("Pack ONE 3DGS model folder to a .pt (Morton-sorted, FP16 attrs)")
    ap.add_argument("--model", required=True, type=str,
                    help="Path to a single model folder (e.g., .../model_frame_1)")
    ap.add_argument("--iter", type=int, default=None,
                    help="Force a specific iteration (e.g., 8000). If omitted, use latest.")
    ap.add_argument("--out_pt", type=str, default=None,
                    help="Optional explicit output .pt path. If omitted, write to <model>/packed/iter_<it>.pt")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    out_pt = Path(args.out_pt).resolve() if args.out_pt else None

    pack_model(model_dir, out_pt=out_pt, iter_override=args.iter)


if __name__ == "__main__":
    main()
