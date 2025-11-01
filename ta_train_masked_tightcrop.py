#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Archival 3DGS — Mask-then-Train Orchestrator (with optional tight crop)
============================================================================

This is a drop-in replacement for ta_train_masked.py that adds two optional flags:
  --tight_crop        : When set, each masked GT is cropped to its tight ROI.
  --pad_px N          : Padding (in pixels) around the tight ROI when cropping (default: 2).

Default behavior (no --tight_crop) remains 100% compatible with your current pipeline:
- We still build a working dataset with masked PNGs and unmodified camera intrinsics.

When --tight_crop is enabled:
- We compute each image's tight ROI from masks_raw/<stem>.npy and crop the masked PNG.
- We duplicate the corresponding camera in cameras.txt, shifting cx,cy by the crop offset
  and updating W,H to the new dimensions.
- We rewrite images.txt to point to the cropped PNG and the new camera id.
- We also crop masks_raw to keep train.py's --masked ROI loss working in the smaller image.

Usage Example (PowerShell):
---------------------------
python .\\ta_train_masked_tightcrop.py `
  -s .\\dataset\\soccer_dynamic_player_B `
  -o .\\output_seq\\soccer_dynamic_player_aabb_only `
  --frames 1 `
  --mask_mode aabb_only `
  --aabb .\\aabb_B.json `
  --save_mask_png `
  --aabb_close_px 2 --close_px 1 --dilate_px 1 `
  --tight_crop --pad_px 2 `
  -- `
  --iterations 8000 -r 1 --sh_degree 3 `
  --densify_from_iter 2500 --densify_until_iter 4500 `
  --densification_interval 600 --densify_grad_threshold 1e-3 `
  --opacity_reset_interval 10000 --lambda_dssim 0.15 `
  --masked --disable_viewer
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import re

# ---------- tiny COLMAP text readers ----------

def read_cameras_text_with_lines(path: Path):
    """
    Return (lines, cams) where:
      - lines: original file lines (list[str])
      - cams : dict[id] = (model, w, h, params(list[float]), line_index)
    """
    cams = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (not s) or s.startswith('#'):
            continue
        toks = s.split()
        cam_id = int(toks[0]); model = toks[1]
        w, h = int(toks[2]), int(toks[3])
        params = list(map(float, toks[4:]))
        cams[cam_id] = (model, w, h, params, i)
    return lines, cams


def read_images_text(path: Path):
    """
    Parse COLMAP images.txt returning (lines, records).
    Each record is a dict with: line_index, img_id, camera_id, name.

    We only need the image header lines (not the 3D point lines).
    """
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith('#'):
            i += 1; continue
        toks = s.split()
        if len(toks) < 10:
            i += 1; continue
        img_id = int(toks[0])
        recs.append({
            'line_index': i,
            'img_id': img_id,
            'camera_id': int(toks[8]),
            'name': toks[9]
        })
        i += 2  # skip the following 3D points line
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
    mg = cache_dir / 'masked_gt'
    if not mg.exists():
        mg = cache_dir / 'masked_gt_Bonly'
    if not mg.exists():
        return False
    return any(mg.glob('*.png'))


def run_mask_step(mode: str, frame_dir: Path, aabb_json: Path, pc_name: str,
                  cache_dir: Path, extra_mask_flags: List[str]) -> Path:
    """Dispatch to masking script. Returns the cache_dir actually used."""
    ensure_dir(cache_dir)
    if mode == 'aabb_only':
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


def build_work_dataset(frame_dir: Path, cache_dir: Path, work_dir: Path,
                       tight_crop: bool=False, pad_px: int=2) -> None:
    """
    Clone sparse/ → work_dir, fill images/ with masked/cropped PNG,
    rewrite images.txt; optionally duplicate cameras with tight crop updates.
    """
    # 1) locate source sparse tree
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

    shutil.copytree(src_sparse, dst_sparse_root, dirs_exist_ok=True)

    # 2) resolve masked_gt and masks_raw
    mg = cache_dir / 'masked_gt'
    if not mg.exists():
        mg = cache_dir / 'masked_gt_Bonly'
    if not mg.exists():
        raise FileNotFoundError(f"No masked_gt found in {cache_dir}")
    masks_raw = cache_dir / 'masks_raw'

    # 3) read/prepare images & cameras
    images_txt_path = dst_sparse_root / 'images.txt'
    cams_txt_path = dst_sparse_root / 'cameras.txt'
    lines, recs = read_images_text(images_txt_path)
    cams_lines, cams = read_cameras_text_with_lines(cams_txt_path)
    next_cam_id = (max(cams.keys()) + 1) if len(cams) else 1

    out_images = ensure_dir(work_dir / 'images')
    out_masks_raw = ensure_dir(work_dir / 'masks_raw')

    def _shift_cxcy(model: str, params: List[float], dx: float, dy: float) -> List[float]:
        p = params.copy()
        # Model schemas (COLMAP):
        # SIMPLE_PINHOLE: [f, cx, cy]
        # PINHOLE      : [fx, fy, cx, cy]
        # SIMPLE_RADIAL: [f, cx, cy, k]
        # RADIAL       : [f, cx, cy, k1, k2]
        # OPENCV       : [fx, fy, cx, cy, k1, k2, p1, p2]
        if model == 'SIMPLE_PINHOLE':
            p[1] -= dx; p[2] -= dy
        elif model == 'PINHOLE':
            p[2] -= dx; p[3] -= dy
        elif model == 'SIMPLE_RADIAL':
            p[1] -= dx; p[2] -= dy
        elif model == 'RADIAL':
            p[1] -= dx; p[2] -= dy
        elif model == 'OPENCV':
            p[2] -= dx; p[3] -= dy
        # Unknown models: keep as-is (safe fallback).
        return p

    for rec in recs:
        img_id = rec['img_id']
        orig_name = rec['name']
        cam_id = rec['camera_id']

        out_png_name = idx_str_for(orig_name, img_id) + '.png'
        src_png = mg / out_png_name
        if not src_png.exists():
            alt = mg / (Path(orig_name).stem + '.png')
            if alt.exists():
                src_png = alt
            else:
                raise FileNotFoundError(f"Masked PNG not found for {orig_name} at {src_png}")

        # Default path if not cropping
        dst_png_path = out_images / out_png_name

        # Patch images.txt line to point to PNG filename
        li = rec['line_index']
        toks = lines[li].split()
        toks[9] = out_png_name  # filename

        # Handle mask npy
        src_npy = masks_raw / (Path(out_png_name).stem + '.npy')
        y0=y1=x0=x1=None  # ROI for this image

        if tight_crop and src_npy.exists():
            mask_np = np.load(str(src_npy))
            if mask_np.dtype != np.uint8 and mask_np.dtype != np.bool_:
                mask_np = (mask_np > 0).astype(np.uint8)
            ys, xs = np.nonzero(mask_np)
            if ys.size > 0 and xs.size > 0:
                # Read original size from PNG
                with Image.open(src_png) as im_tmp:
                    W0, H0 = im_tmp.size  # PIL returns (W, H)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                if pad_px > 0:
                    y0 = max(0, y0 - pad_px); x0 = max(0, x0 - pad_px)
                    y1 = min(H0, y1 + pad_px); x1 = min(W0, x1 + pad_px)
                # Crop and save PNG
                with Image.open(src_png) as im:
                    im_c = im.crop((x0, y0, x1, y1))
                    im_c.save(dst_png_path)
                # Crop and save mask npy (uint8 {0,1})
                mask_crop = (mask_np[y0:y1, x0:x1] > 0).astype(np.uint8)
                np.save(str(out_masks_raw / (Path(out_png_name).stem + '.npy')), mask_crop)

                # Duplicate camera with shifted cx,cy and new W,H
                if cam_id not in cams:
                    raise KeyError(f"camera_id {cam_id} not found in cameras.txt")
                model, Worig, Horig, params, _ = cams[cam_id]
                newW = int(x1 - x0)
                newH = int(y1 - y0)
                new_params = _shift_cxcy(model, params, dx=float(x0), dy=float(y0))
                new_cam_id = next_cam_id; next_cam_id += 1
                cams_lines.append(
                    f"{new_cam_id} {model} {newW} {newH} " + " ".join(f"{v:.6f}" for v in new_params) + "\n"
                )

                toks[8] = str(new_cam_id)  # switch to new camera
            else:
                # No positive mask; fall back to copy-as-is
                shutil.copy2(src_png, dst_png_path)
                if src_npy.exists():
                    shutil.copy2(src_npy, out_masks_raw / src_npy.name)
        else:
            # Not cropping → copy as-is
            shutil.copy2(src_png, dst_png_path)
            if src_npy.exists():
                shutil.copy2(src_npy, out_masks_raw / src_npy.name)

        # Write back images.txt header line
        lines[li] = ' '.join(toks) + '\n'


    # Persist updated text files
    with open(images_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    with open(cams_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(cams_lines)


@dataclass
class FramePlan:
    frame: int
    src_dir: Path
    cache_dir: Path
    work_dir: Path
    model_dir: Path


# ---------- CLI ----------

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
    if s in ('even', 'odd'):
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
    ap = argparse.ArgumentParser(description='Mask-then-Train batch runner (tight-crop optional)')
    ap.add_argument('-s', '--source_root', required=True, help='Root with frame_* COLMAP datasets')
    ap.add_argument('-o', '--output_root', required=True, help='Root for model outputs')
    ap.add_argument('--frames', default='all', help='e.g., "all" | "1-10" | "1,5" | "frame_3"')
    ap.add_argument('--prefix', default='model_frame_', help='Model dir prefix')
    ap.add_argument('--per-frame-subdir', default=None)
    ap.add_argument('--resume-if-exists', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--masked', action='store_true', help='Enable ROI crop + masked loss inside train.py')

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

    # NEW: tight-crop controls
    ap.add_argument('--tight_crop', action='store_true', help='Crop masked PNGs to tight ROI and duplicate cameras with shifted cx,cy')
    ap.add_argument('--pad_px', type=int, default=2, help='Padding around tight ROI when --tight_crop is on')

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

    aabb_json = Path(args.aabb).resolve()

    for p in plans:
        print(f"\\n==== [frame_{p.frame}] ====")
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

        # 2) Build working dataset (with or without cropping)
        if p.work_dir.exists() and any((p.work_dir / 'images').glob('*.png')):
            print('[work] reuse existing working dataset at', p.work_dir)
        else:
            if p.work_dir.exists():
                shutil.rmtree(p.work_dir)
            ensure_dir(p.work_dir)
            build_work_dataset(p.src_dir, p.cache_dir, p.work_dir, tight_crop=args.tight_crop, pad_px=args.pad_px)

        # 3) Forward --masked flag if requested
        if args.masked and ('--masked' not in train_flags):
            train_flags = train_flags + ['--masked']

        # 4) Train on the working dataset
        cmd = [sys.executable, '-u', 'train.py',
               '--source_path', str(p.work_dir),
               '--model_path', str(p.model_dir)] + train_flags

        print('[train][spawn]', ' '.join(cmd))
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.setdefault('PYTHONHASHSEED', '0')
        env['AABB_JSON'] = str(aabb_json)  # for colmap_loader to optionally use AABB
        print(f"[train][env] AABB_JSON={env['AABB_JSON']}")
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), env=env, check=False)
        if proc.returncode != 0:
            print(f"[train][FAIL] frame_{p.frame} rc={proc.returncode}")
        else:
            print(f"[train][OK] frame_{p.frame} -> {p.model_dir}")


if __name__ == '__main__':
    main()
