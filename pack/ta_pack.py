#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ta_pack.py — Pack per-frame 3DGS PLY to contiguous `.pt` with Morton sort & FP16 attrs.

Examples:
  # 原有用法（无数据集名）
  python ta_pack.py -m ../output_seq --prefix model_frame_ --out ../output_seq_packed --autocreate

  # 带数据集名（输入：../output_seq/soccer，输出：../output_seq_packed/soccer）
  python ta_pack.py -m ../output_seq --name soccer --prefix model_frame_ --out ../output_seq_packed --autocreate
"""
import argparse, sys, math
from pathlib import Path
import numpy as np
import torch

# ---------- add project root to sys.path (go up to 5 levels) ----------
def _add_repo_root():
    here = Path(__file__).resolve().parent
    cur = here
    for _ in range(5):
        markers = ("output_seq", "server", "pack")
        if any((cur / m).exists() for m in markers):
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            return
        cur = cur.parent
_add_repo_root()

# ---------- PLY reader ----------
try:
    from plyfile import PlyData
except Exception as e:
    raise SystemExit("Please install plyfile: pip install plyfile") from e

def list_frames(models_root: Path, prefix: str):
    frames = []
    if not models_root.exists():
        raise SystemExit(f"models_root not found: {models_root}")
    for p in models_root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            try:
                frames.append(int(p.name.split("_")[-1]))
            except:
                pass
    return sorted(frames)

def find_latest_iteration(model_path: Path) -> int:
    pc = model_path / "point_cloud"
    if not pc.exists(): return -1
    best = -1
    for d in pc.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                it = int(d.name.split("_")[1]); best = max(best, it)
            except:
                pass
    return best

def _get_fields_safe(verts, names, allow_missing=False, default_val=0.0, shape=None, dtype=np.float32):
    out = []
    missing = []
    for n in names:
        if n in verts.dtype.names:
            out.append(np.asarray(verts[n], dtype=dtype))
        else:
            missing.append(n)
            if not allow_missing:
                raise KeyError(f"PLY missing required field '{n}'")
            out.append(np.full((len(verts),), default_val, dtype=dtype))
    arr = np.stack(out, axis=1) if shape is None else np.stack(out, axis=1).reshape(shape)
    if missing:
        print(f"[ta_pack] WARN missing fields: {missing} -> filled with {default_val}")
    return arr

def _gather_sh(verts):
    # f_dc_0..2 -> [N,3,1]
    dc_names = [f"f_dc_{i}" for i in range(3)]
    dc = _get_fields_safe(verts, dc_names, allow_missing=False, dtype=np.float32).reshape(-1,3,1)

    # f_rest_0..K-1 -> [N,3,K]; K 推断自存在的最大索引
    rest_cols = [n for n in (verts.dtype.names or []) if n.startswith("f_rest_")]
    if not rest_cols:
        # 某些导出可能写成 features_*；尝试兜底
        rest_cols = [n for n in (verts.dtype.names or []) if n.startswith("features_rest_")]
    if not rest_cols:
        # 没有 REST，也允许（相当于 L=0）
        rest = np.zeros((len(verts),3,0), dtype=np.float32)
        L = 0
    else:
        def _idx(n): return int(n.split("_")[-1])
        rest_cols.sort(key=_idx)
        R = np.stack([np.asarray(verts[n], dtype=np.float32) for n in rest_cols], axis=1)
        K_total = R.shape[1]
        if K_total % 3 == 0:
            K = K_total // 3
            rest = R.reshape(len(verts), 3, K)
        else:
            rest = np.tile(R[:,None,:], (1,3,1))
        L = int(round(math.sqrt(rest.shape[2]+1)-1))
    return dc, rest, L

def morton3D_order(xyz: np.ndarray, qbits: int = 10):
    mins = xyz.min(axis=0); maxs = xyz.max(axis=0)
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
    codes = (expand(qi[:,0]) << 0) | (expand(qi[:,1]) << 1) | (expand(qi[:,2]) << 2)
    order = np.argsort(codes, kind="mergesort")
    return order, mins.astype(np.float32), maxs.astype(np.float32)

def pack_one_frame(ply_path: Path, out_path: Path):
    ply = PlyData.read(str(ply_path))
    verts = ply["vertex"].data  # structured array

    xyz = np.stack([np.asarray(verts[n], dtype=np.float32) for n in ("x","y","z")], axis=1)  # [N,3]
    # scaling
    scale_names = [n for n in ("scale_0","scale_1","scale_2","scaling_0","scaling_1","scaling_2") if n in verts.dtype.names]
    if len(scale_names) >= 3:
        scaling = _get_fields_safe(verts, scale_names[:3], allow_missing=False, dtype=np.float32)
    else:
        scaling = _get_fields_safe(verts, ["scale_0","scale_1","scale_2"], allow_missing=True, dtype=np.float32)
    # rotation (quaternion x,y,z,w)
    rot_names = [n for n in ("rot_0","rot_1","rot_2","rot_3","rotation_0","rotation_1","rotation_2","rotation_3") if n in verts.dtype.names]
    if len(rot_names) >= 4:
        rotation = _get_fields_safe(verts, rot_names[:4], allow_missing=False, dtype=np.float32)
    else:
        rotation = _get_fields_safe(verts, ["rot_0","rot_1","rot_2","rot_3"], allow_missing=True, dtype=np.float32)
    # opacity
    if "opacity" in verts.dtype.names:
        opacity = np.asarray(verts["opacity"], dtype=np.float32)[:,None]
    else:
        opacity = np.ones((len(verts),1), dtype=np.float32)

    sh_dc, sh_rest, L = _gather_sh(verts)

    # Morton 排序
    order, aabb_min, aabb_max = morton3D_order(xyz, qbits=10)
    xyz      = xyz[order]
    scaling  = scaling[order]
    rotation = rotation[order]
    opacity  = opacity[order]
    sh_dc    = sh_dc[order]
    sh_rest  = sh_rest[order]

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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pkg, out_path)
    print(f"[ta_pack] saved {out_path} (N={pkg['n']}, L={pkg['sh_degree']})")

def main():
    ap = argparse.ArgumentParser("Pack 3DGS frames to .pt")
    # --- 新增：单文件模式 ---
    ap.add_argument("--single_ply", type=str, default=None,
                    help="直接打包这一个 PLY（跳过 models_root/frames 扫描）")
    ap.add_argument("--out_pt", type=str, default=None,
                    help="单文件模式下输出的 .pt 完整路径；若未提供则需 --frame 与 --iter 来构造路径")
    ap.add_argument("--frame", type=int, default=1,
                    help="单文件模式下用于构造输出路径的帧号（与 --prefix 组合）")
    ap.add_argument("--iter", type=int, default=8000,
                    help="单文件模式下用于构造输出路径的迭代号（生成 iter_xxxx.pt）")

    # --- 目录模式（原有逻辑） ---
    ap.add_argument("-m","--models_root", required=False, type=str,
                    help="root folder that may contain dataset subfolders")
    ap.add_argument("--name","--dataset", dest="dataset", default=None, type=str, help="dataset name, e.g., 'soccer'")
    ap.add_argument("--prefix", default="model_frame_", type=str)
    ap.add_argument("--out", required=False, type=str,
                    help="打包根目录（目录模式必需；单文件模式若未给 --out_pt 也需要）")
    ap.add_argument("--autocreate", action="store_true")
    args = ap.parse_args()
    # -------- 单文件模式：直接把 --single_ply 打成 .pt --------
    if args.single_ply:
        ply_path = Path(args.single_ply)
        if not ply_path.exists():
            raise SystemExit(f"[ta_pack] single_ply not found: {ply_path}")
        # 解析输出路径
        if args.out_pt:
            out_path = Path(args.out_pt)
        else:
            if not args.out:
                raise SystemExit("[ta_pack] need --out or --out_pt in single-file mode")
            out_root = Path(args.out)
            out_path = out_root / f"{args.prefix}{args.frame}" / f"iter_{args.iter}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ta_pack] single-file mode")
        print(f"[ta_pack]  ply : {ply_path}")
        print(f"[ta_pack]  out : {out_path}")
        pack_one_frame(ply_path, out_path)
        return
    # resolve input/output with optional dataset name
    if not args.models_root or not args.out:
        raise SystemExit("[ta_pack] directory mode requires --models_root and --out (no --single_ply).")
    models_base = Path(args.models_root)
    out_base = Path(args.out)
    if args.dataset:
        models_root = models_base / args.dataset
        out_root = out_base / args.dataset
        print(f"[ta_pack] dataset='{args.dataset}'")
    else:
        models_root = models_base
        out_root = out_base

    if args.autocreate:
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"[ta_pack] input root : {models_root}")
    print(f"[ta_pack] output root: {out_root}")

    frames = list_frames(models_root, args.prefix)
    if not frames:
        raise SystemExit("no frames found under: " + str(models_root))

    for f in frames:
        mp = models_root / f"{args.prefix}{f}"
        it = find_latest_iteration(mp)
        if it < 0:
            print(f"[ta_pack] skip {f}: no iteration")
            continue
        ply = mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if not ply.exists():
            print(f"[ta_pack] skip {f}: missing {ply}")
            continue
        out_path = out_root / f"{args.prefix}{f}" / f"iter_{it}.pt"
        if out_path.exists():
            print(f"[ta_pack] exists: {out_path}, skip")
            continue
        pack_one_frame(ply, out_path)

if __name__ == "__main__":
    main()
