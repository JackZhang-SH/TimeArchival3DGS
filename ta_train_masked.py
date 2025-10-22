#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Archival 3DGS — Mask-then-Train Orchestrator
=================================================

Batch runner that, for each COLMAP dataset under a source root (frame_*),
1) builds foreground masks using one of two strategies:
   - aabb_only      → uses ray–box (AABB) projection only (no depth).
   - from_B_only    → uses B point cloud splatting (constant or adaptive radius).
   The first time masks are computed for a frame, they are cached and reused.
2) constructs a *working* dataset with masked images (no modification to the
   original dataset) so that training consumes already-masked GT.
3) calls the stock train.py (same as ta_train) for each frame, in isolation.

Why build a working dataset?
----------------------------
- It guarantees train.py uses masked images regardless of internal alpha-mask
  conventions. We rewrite images.txt to point to the copied PNG masks.

Outputs (per frame_i):
- <work_root>/frame_i_work/ : COLMAP structure with PNG masked images + patched images.txt
- <mask_cache_root>/frame_i/<mode>/ : cached masks/masked_gt/*.png + masks_raw/*.npy
- Model: <output_root>/<prefix><i>[/per-frame-subdir]

Usage Example (PowerShell): see the message body in ChatGPT.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# ---------- tiny COLMAP text readers (same as in your scripts) ----------
import re
import numpy as np

def read_cameras_text(path: Path):
    cams = {}
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            if not ln.strip() or ln.startswith('#'): continue
            toks = ln.split()
            cam_id = int(toks[0]); model = toks[1]
            w, h = int(toks[2]), int(toks[3])
            params = list(map(float, toks[4:]))
            cams[cam_id] = (model, w, h, params)
    return cams


def read_images_text(path: Path):
    recs = []  # list of (line_idx, parsed_dict)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1; continue
        toks = line.split()
        if len(toks) < 10:
            i += 1; continue
        img_id = int(toks[0])
        # q(4), t(3), cam_id, name
        recs.append({
            'line_index': i,
            'img_id': img_id,
            'qvec': list(map(float, toks[1:5])),
            'tvec': list(map(float, toks[5:8])),
            'camera_id': int(toks[8]),
            'name': toks[9]
        })
        i += 2  # skip point lines
    return lines, recs

_digit_re = re.compile(r"^\d+$")

def idx_str_for(name: str, img_id: int) -> str:
    stem = Path(name).stem
    return stem if _digit_re.match(stem) else f"{img_id:04d}"

# ---------- core helpers ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def masks_already_cached(cache_dir: Path) -> bool:
    # consider cached if masked_gt has at least 1 png
    mg = cache_dir / 'masked_gt'
    if not mg.exists():
        # older naming in B-only script
        mg = cache_dir / 'masked_gt_Bonly'
    if not mg.exists():
        return False
    pngs = list(mg.glob('*.png'))
    return len(pngs) > 0


def run_mask_step(mode: str, frame_dir: Path, aabb_json: Path, pc_name: str,
                  cache_dir: Path, extra_mask_flags: List[str]) -> Path:
    """Dispatch to masking script. Returns the cache_dir actually used."""
    ensure_dir(cache_dir)
    if mode == 'aabb_only':
        # make_mask_aabb_only.py --aabb AABB --colmap frame_dir --out cache_dir ...
        cmd = [sys.executable, '-u', 'make_mask_aabb_only.py',
               '--aabb', str(aabb_json),
               '--colmap', str(frame_dir),
               '--out', str(cache_dir)] + extra_mask_flags
    elif mode == 'from_B_only':
        pc_path = frame_dir / pc_name
        cmd = [sys.executable, '-u', 'make_mask_from_B_only.py',
               '--pc', str(pc_path),
               '--aabb', str(aabb_json),
               '--colmap', str(frame_dir),
               '--out', str(cache_dir)] + extra_mask_flags
    else:
        raise ValueError('mode must be aabb_only or from_B_only')

    print('[Mask][spawn]', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Mask step failed (rc={proc.returncode}) for {frame_dir}")
    return cache_dir


def build_work_dataset(frame_dir: Path, cache_dir: Path, work_dir: Path) -> None:
    """
    Copy sparse/ to work_dir, and create images/ filled with PNG masked images.
    Also rewrite images.txt so that each image line points to the PNG we created.
    We map each images.txt entry to the mask filename produced by masking
    scripts (idx_str.png). If B-only script was used, it stores under
    masked_gt_Bonly/; AABB-only under masked_gt/.
    """
    # 1) clone sparse — preserve "0" level if it exists, because train.py expects sparse/0/*
    src_sparse0 = frame_dir / 'sparse' / '0'
    has_level0 = src_sparse0.exists()
    if has_level0:
        src_sparse = src_sparse0
        dst_sparse_root = ensure_dir(work_dir / 'sparse' / '0')
    else:
        src_sparse = frame_dir / 'sparse'
        dst_sparse_root = ensure_dir(work_dir / 'sparse')

    if not src_sparse.exists():
        raise FileNotFoundError(f"Missing sparse/ in {frame_dir}")

    # copy the chosen source tree into the corresponding destination
    shutil.copytree(src_sparse, dst_sparse_root, dirs_exist_ok=True)

    # 2) decide masked_gt folder name
    mg = cache_dir / 'masked_gt'
    if not mg.exists():
        mg = cache_dir / 'masked_gt_Bonly'
    if not mg.exists():
        raise FileNotFoundError(f"No masked_gt found in {cache_dir}")

    # 3) read images.txt and patch names → copy pngs
    images_txt_path = dst_sparse_root / 'images.txt'
    lines, recs = read_images_text(images_txt_path)
    out_images = ensure_dir(work_dir / 'images')

    for rec in recs:
        img_id = rec['img_id']
        orig_name = rec['name']
        out_png_name = idx_str_for(orig_name, img_id) + '.png'
        src_png = mg / out_png_name
        if not src_png.exists():
            alt = mg / (Path(orig_name).stem + '.png')
            if alt.exists():
                src_png = alt
            else:
                raise FileNotFoundError(f"Masked PNG not found for {orig_name} at {src_png}")
        shutil.copy2(src_png, out_images / out_png_name)
        # patch the image line's name token to png
        li = rec['line_index']
        toks = lines[li].split()
        toks[9] = out_png_name
        lines[li] = ' '.join(toks) + '\n'

    # 4) write patched images.txt
    with open(images_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 5) cameras.txt and points3D.txt remain the same; no need to touch


@dataclass
class FramePlan:
    frame: int
    src_dir: Path
    cache_dir: Path
    work_dir: Path
    model_dir: Path


# ---------- main ----------

def parse_frames(frames_expr: str, dataset_root: Path) -> List[int]:
    def scan_all() -> List[int]:
        found = []
        for p in sorted(dataset_root.glob('frame_*')):
            if p.is_dir():
                try:
                    found.append(int(p.name.split('_')[1]))
                except Exception:
                    pass
        return found
    s = frames_expr.strip().lower()
    if s == 'all':
        return scan_all()
    if s == 'even' or s == 'odd':
        all_frames = scan_all()
        return [i for i in all_frames if (i % 2 == 0) == (s == 'even')]
    if s.startswith('frame_'):
        return [int(s.split('_')[1])]
    if '-' in s:
        a, b = s.split('-'); return list(range(int(a), int(b) + 1))
    if ',' in s:
        return [int(x) for x in s.split(',')]
    return [int(s)]


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Mask-then-Train batch runner')
    ap.add_argument('-s', '--source_root', required=True, help='Root with frame_* COLMAP datasets')
    ap.add_argument('-o', '--output_root', required=True, help='Root for model outputs')
    ap.add_argument('--frames', default='all', help='e.g., "all" | "1-10" | "1,5" | "frame_3"')
    ap.add_argument('--prefix', default='model_frame_', help='Model dir prefix')
    ap.add_argument('--per-frame-subdir', default=None)
    ap.add_argument('--resume-if-exists', action='store_true')
    ap.add_argument('--dry-run', action='store_true')

    # masking options
    ap.add_argument('--mask_mode', choices=['aabb_only','from_B_only'], required=True)
    ap.add_argument('--aabb', required=True, help='Path to aabb_B.json')
    ap.add_argument('--pc_name', default='fused_points.ply', help='Relative point cloud path inside each frame')
    ap.add_argument('--mask_cache_root', default=None, help='Where to cache masks (default: <source_root>/_ta_mask_cache)')
    # typical flags forwarded to mask scripts; keep them generic
    ap.add_argument('--save_mask_png', action='store_true')
    ap.add_argument('--aabb_close_px', type=int, default=0)
    ap.add_argument('--close_px', type=int, default=0)
    ap.add_argument('--dilate_px', type=int, default=0)
    # B-only specific
    ap.add_argument('--adaptive_splat', action='store_true')
    ap.add_argument('--s_world_scale', type=float, default=1.0)
    ap.add_argument('--splat_px_min', type=int, default=1)
    ap.add_argument('--splat_px_max', type=int, default=6)
    ap.add_argument('--splat_radius_px', type=int, default=1)

    # sentinel to split args to train.py
    ap.add_argument('--', dest='dashdash', action='store_true', help=argparse.SUPPRESS)

    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv:
        dd = argv.index('--')
        wrapper_args = argv[:dd]
        train_flags = argv[dd+1:]
    else:
        wrapper_args = argv
        train_flags = []

    args = ap.parse_args(wrapper_args)

    src_root = Path(args.source_root).resolve()
    out_root = Path(args.output_root).resolve(); ensure_dir(out_root)
    mask_cache_root = Path(args.mask_cache_root).resolve() if args.mask_cache_root else ensure_dir(src_root / '_ta_mask_cache')
    work_root = ensure_dir(Path.cwd() / '_work_datasets')

    frames = parse_frames(args.frames, src_root)
    print(f"[MaskTrain] Frames: {frames}")

    # auto-disable viewer unless explicitly provided
    if '--disable_viewer' not in train_flags:
        train_flags = train_flags + ['--disable_viewer']

    plans: List[FramePlan] = []
    for i in frames:
        fdir = src_root / f'frame_{i}'
        if not fdir.exists():
            print(f"[skip] frame_{i} missing: {fdir}")
            continue
        cache_dir = mask_cache_root / f'frame_{i}' / args.mask_mode
        work_dir  = work_root / f'frame_{i}_work'
        model_dir = out_root / f"{args.prefix}{i}"
        if args.per_frame_subdir:
            model_dir = model_dir / args.per_frame_subdir
        ensure_dir(model_dir)
        plans.append(FramePlan(i, fdir, cache_dir, work_dir, model_dir))

    # Per-frame pipeline
    aabb_json = Path(args.aabb).resolve()

    for p in plans:
        print(f"\n==== [frame_{p.frame}] ====")
        if args.resume_if_exists and (p.model_dir / 'chkpnt').exists():
            print('[resume-skip] trained artifacts exist ->', p.model_dir)
            continue

        # 1) Mask cache check + compute if needed
        if args.mask_mode == 'aabb_only' and masks_already_cached(p.cache_dir):
            print('[cache] reuse AABB-only masks at', p.cache_dir)
        else:
            mask_flags: List[str] = []
            if args.save_mask_png: mask_flags.append('--save_mask_png')
            if args.aabb_close_px: mask_flags += ['--aabb_close_px', str(args.aabb_close_px)]
            if args.close_px:      mask_flags += ['--close_px',      str(args.close_px)]
            if args.dilate_px:     mask_flags += ['--dilate_px',     str(args.dilate_px)]

            if args.mask_mode == 'from_B_only':
                if args.adaptive_splat:
                    mask_flags += ['--adaptive_splat', '--s_world_scale', str(args.s_world_scale),
                                   '--splat_px_min', str(args.splat_px_min), '--splat_px_max', str(args.splat_px_max)]
                else:
                    mask_flags += ['--splat_radius_px', str(args.splat_radius_px)]

            run_mask_step(args.mask_mode, p.src_dir, aabb_json, args.pc_name, p.cache_dir, mask_flags)

        # 2) Build working dataset
        if p.work_dir.exists() and any((p.work_dir / 'images').glob('*.png')):
            print('[work] reuse existing working dataset at', p.work_dir)
        else:
            if p.work_dir.exists():
                shutil.rmtree(p.work_dir)
            ensure_dir(p.work_dir)
            build_work_dataset(p.src_dir, p.cache_dir, p.work_dir)

        # 3) Train
        # 训练阶段不再传 mask/prune 类参数，改为仅用工作数据集 + 环境变量 AABB_JSON
        cmd = [sys.executable, '-u', 'train.py',
               '--source_path', str(p.work_dir),
               '--model_path', str(p.model_dir)] + train_flags

        print('[train][spawn]', ' '.join(cmd))
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.setdefault('PYTHONHASHSEED', '0')
        # 关键：让 colmap_loader 只在 AABB 内加载 COLMAP 点云
        env['AABB_JSON'] = str(aabb_json)
        # 可选：把“使用的 AABB”打印出来便于排查
        print(f"[train][env] AABB_JSON={env['AABB_JSON']}")
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), env=env, check=False)
        if proc.returncode != 0:
            print(f"[train][FAIL] frame_{p.frame} rc={proc.returncode}")
        else:
            print(f"[train][OK] frame_{p.frame} -> {p.model_dir}")


if __name__ == '__main__':
    main()
