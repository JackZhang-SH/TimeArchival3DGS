#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ta_pack.py — Pack per-frame 3DGS PLY to contiguous `.pt` with Morton sort & FP16 attrs.

Overwrite-enabled version:
  • Always (re)create per-frame output folders before writing (removes old files).
  • Never skip existing outputs; new packs overwrite prior results.

New in the original version (retained here):
  • --merged_root MODE: pack merged outputs produced by merge_A_B_batch.py
    - Input layout: <merged_root>/frame_n/point_cloud_merged.ply  (or model_frame_n)
    - Output layout: <out>/<name>/model_frame_n/model_frame_n.pt

Examples:
  # 原有用法（无数据集名，扫描 models_root 下的 model_frame_* 并选最新 iteration_*）
  python ta_pack.py -m ../output_seq --prefix model_frame_ --out ../output_seq_packed --autocreate

  # 带数据集名（输入：../output_seq/soccer，输出：../output_seq_packed/soccer）
  python ta_pack.py -m ../output_seq --name soccer --prefix model_frame_ --out ../output_seq_packed --autocreate

  # merged-root 模式（输入 merged/frame_n/point_cloud_merged.ply，输出 name/model_frame_n/model_frame_n.pt）
  python ta_pack.py --merged_root ../output_seq/merged --name soccer_merged --out ../output_seq_packed --autocreate

  # 单文件模式（直接指定一个 PLY 并强制覆盖目标目录）
  python ta_pack.py --single_ply path/to/point_cloud.ply --out ../output_seq_packed --prefix model_frame_ --frame 1 --iter 8000
"""
import argparse, sys, math, shutil
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple
def _find_merged_input_ply(frame_dir: Path) -> Tuple[Optional[Path], Optional[int], str]:
    """
    返回 (ply_path, iteration, layout_tag)
      layout_tag in {"flat", "folder", "iter"}
    优先级：flat > folder > iter(max)
    """
    # 1) 扁平：.../point_cloud_merged.ply
    p_flat = frame_dir / "point_cloud_merged.ply"
    if p_flat.exists():
        return p_flat, None, "flat"

    # 2) 子目录：.../point_cloud_merged/point_cloud.ply
    p_folder = frame_dir / "point_cloud_merged" / "point_cloud.ply"
    if p_folder.exists():
        return p_folder, None, "folder"

    # 3) 迭代结构：.../point_cloud/iteration_*/point_cloud.ply（取最大 iter）
    pc = frame_dir / "point_cloud"
    best_it, best_ply = -1, None
    if pc.exists():
        for d in pc.iterdir():
            if d.is_dir() and d.name.startswith("iteration_"):
                try:
                    it = int(d.name.split("_")[1])
                except:
                    continue
                p = d / "point_cloud.ply"
                if p.exists() and it > best_it:
                    best_it, best_ply = it, p
    if best_ply is not None:
        return best_ply, best_it, "iter"

    return None, None, "none"

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

def list_merged_frames(merged_root: Path):
    """Find frames under merged_root supporting both 'frame_n' and 'model_frame_n'."""
    frames = []
    if not merged_root.exists():
        raise SystemExit(f"merged_root not found: {merged_root}")
    for p in merged_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("frame_") or name.startswith("model_frame_"):
            try:
                n = int(name.split("_")[-1])
                frames.append((n, p))
            except:
                pass
    frames.sort(key=lambda x: x[0])
    return frames

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
            # 容错：若不是 3 的倍数，重复铺成 (3,K_total) 的形状
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
    torch.save(pkg, out_path)  # overwrite if exists
    print(f"[ta_pack] saved {out_path} (N={pkg['n']}, L={pkg['sh_degree']})")

# -------- helpers --------
def ensure_clean_dir(p: Path):
    """Remove directory `p` if it exists, then recreate it empty."""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# ========================= main =========================
def main():
    ap = argparse.ArgumentParser("Pack 3DGS frames to .pt (overwrite version)")
    # --- 单文件模式 ---
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

    # --- merged-root 模式 ---
    ap.add_argument("--merged_root", type=str, default=None,
                    help="合并后的根目录（包含 frame_n/point_cloud_merged.ply 或 model_frame_n）")
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
            out_dir = out_root / f"{args.prefix}{args.frame}"
            ensure_clean_dir(out_dir)
            out_path = out_dir / f"iter_{args.iter}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ta_pack] single-file mode")
        print(f"[ta_pack]  ply : {ply_path}")
        print(f"[ta_pack]  out : {out_path}")
        pack_one_frame(ply_path, out_path)
        return

    # -------- merged-root 模式：frame_n/point_cloud_merged.ply -> name/model_frame_n/model_frame_n.pt --------
    if args.merged_root:
        if not args.out or not args.dataset:
            raise SystemExit("[ta_pack] merged-root mode requires --out and --name/--dataset")
        merged_root = Path(args.merged_root)
        out_base = Path(args.out)
        out_root = out_base / args.dataset
        if args.autocreate:
            out_root.mkdir(parents=True, exist_ok=True)

        frames = list_merged_frames(merged_root)
        if not frames:
            raise SystemExit(f"[ta_pack] no frame_* or model_frame_* found under: {merged_root}")

        print(f"[ta_pack] merged-root mode")
        print(f"[ta_pack]  merged_root : {merged_root}")
        print(f"[ta_pack]  out_root    : {out_root}")
        for n, fdir in frames:
            in_ply, it_opt, kind = _find_merged_input_ply(fdir)
            if not in_ply:
                print(f"[ta_pack] skip frame {n}: no merged ply found under {fdir}")
                continue

            out_dir = out_root / f"model_frame_{n}"
            ensure_clean_dir(out_dir)
            out_pt  = out_dir / f"model_frame_{n}.pt"

            extra = f" (iter={it_opt})" if it_opt is not None else ""
            print(f"[ta_pack] pack frame {n}: [{kind}]{extra}")
            print(f"          in  = {in_ply}")
            print(f"          out = {out_pt}")
            pack_one_frame(in_ply, out_pt)
        return

    # -------- 目录模式（原有逻辑，改为覆盖写） --------
    if not args.models_root or not args.out:
        raise SystemExit("[ta_pack] directory mode requires --models_root and --out (no --single_ply / --merged_root).")
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
        out_dir = out_root / f"{args.prefix}{f}"
        ensure_clean_dir(out_dir)  # <— always clean & recreate per-frame output
        out_path = out_dir / f"iter_{it}.pt"
        pack_one_frame(ply, out_path)

if __name__ == "__main__":
    main()
