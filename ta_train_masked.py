#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Archival 3DGS — Train on Black-Masked GT (no --masked)

- 在 work dataset 中为当前帧生成“掩码区域纯黑的三通道 PNG”（无 alpha），并重写 images.txt 指向这些 PNG；
- 训练直接以黑底 GT 进行（无需 --masked / masks_raw）；
- work dataset 仅包含当前帧 images/ 与 sparse/；前一帧自动清理；
- 全部运行结束后清空 `_work_datasets/`；
- 掩码缓存为空时，按 AABB 为所选帧自动生成后规范为 <stem>.npy 并复用。
"""
from __future__ import annotations
import argparse, os, sys, shutil, subprocess, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import time
# ------------------------------- COLMAP helpers -------------------------------
def read_images_text(path: Path) -> Tuple[List[str], List[dict]]:
    recs: List[dict] = []
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
        try:
            img_id = int(toks[0])
        except Exception:
            i += 1; continue
        name = toks[9]
        recs.append({'line_index': i, 'img_id': img_id, 'name': name})
        i += 2
    return lines, recs

_digit_re = re.compile(r"^\d+$")
def idx_str_for(name: str, img_id: int) -> str:
    stem = Path(name).stem
    return stem if _digit_re.match(stem) else f"{img_id:04d}"

# --------------------------------- FS utils -----------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p
def clean_dir(p: Path) -> None:
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# ------------------------- Mask cache normalization ---------------------------
def find_masks_in_dir(root: Path) -> List[Path]:
    return sorted([q for q in root.glob("*.npy") if q.is_file()])
def masks_already_cached(cache_dir: Path) -> bool:
    return len(find_masks_in_dir(cache_dir)) > 0

def png_to_npy_mask(png_path: Path, out_npy: Path) -> None:
    import cv2
    m = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(str(png_path))
    np.save(str(out_npy), (m > 127).astype(np.uint8))

def gt_to_mask_npy(gt_png: Path, out_npy: Path) -> None:
    import cv2
    img = cv2.imread(str(gt_png), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(str(gt_png))
    if img.ndim == 3 and img.shape[2] == 4:
        mask = (img[:, :, 3] > 127).astype(np.uint8)
    elif img.ndim == 2:
        mask = (img > 0).astype(np.uint8)
    else:
        mask = (np.max(img[:, :, :3], axis=2) > 0).astype(np.uint8)
    np.save(str(out_npy), mask)

def normalize_cache_outputs(cache_dir: Path) -> None:
    existing = {p.name for p in cache_dir.glob("*.npy")}
    subdir_names = [
        "masks","masks_aabb","masks_Bonly","masks_residual",
        "masked_gt","masked_gt_Bonly","masks_raw","masks_raw_Bonly"
    ]
    subdirs = [d for d in (cache_dir / n for n in subdir_names) if d.exists()]
    for d in subdirs:
        if not d.name.startswith("masks") or d.name.startswith("masks_raw"): continue
        for p in sorted(d.glob("*.png")):
            dst = cache_dir / (p.stem + ".npy")
            if dst.name not in existing and not dst.exists():
                png_to_npy_mask(p, dst); existing.add(dst.name)
    for d in subdirs:
        if not d.name.startswith("masks_raw"): continue
        for p in sorted(d.glob("*.npy")):
            dst = cache_dir / p.name
            if dst.name not in existing and not dst.exists():
                shutil.copy2(p, dst); existing.add(dst.name)
    for d in subdirs:
        if not d.name.startswith("masked_gt"): continue
        for p in sorted(d.glob("*.png")):
            dst = cache_dir / (p.stem + ".npy")
            if dst.name not in existing and not dst.exists():
                gt_to_mask_npy(p, dst); existing.add(dst.name)
    for p in sorted(cache_dir.glob("*.png")):
        dst = cache_dir / (p.stem + ".npy")
        if dst.name not in existing and not dst.exists(): png_to_npy_mask(p, dst)
        try: p.unlink()
        except Exception: pass
    for d in subdirs:
        try: shutil.rmtree(d)
        except Exception: pass
    if not any(cache_dir.glob("*.npy")):
        raise RuntimeError(f"normalize_cache_outputs: no masks could be collected under {cache_dir}")

# ---------------------- Auto-generation of AABB masks --------------------------
def run_make_aabb_for_frame(aabb_json: Path, frame_root: Path, out_dir: Path,
                            aabb_close_px: int, close_px: int, dilate_px: int) -> None:
    ensure_dir(out_dir)
    cmd = [
        sys.executable, str(Path(__file__).with_name('make_mask_aabb_only.py')),
        '--aabb', str(aabb_json), '--colmap', str(frame_root),
        '--gt_dir', str(frame_root / 'images'),
        '--out', str(out_dir),
        '--aabb_close_px', str(aabb_close_px),
        '--close_px', str(close_px),
        '--dilate_px', str(dilate_px),
        '--save_mask_png'
    ]
    print('[maskgen][spawn]', ' '.join(cmd))
    rc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=False).returncode
    if rc != 0: raise RuntimeError(f"mask generation failed for {frame_root} (rc={rc})")

def ensure_cache_masks(aabb_json: Path, src_root: Path, frames: List[int], cache_dir: Path,
                       aabb_close_px: int, close_px: int, dilate_px: int,
                       ref_frame: int | None = None) -> None:
    """当缓存为空时，只用一个参考帧生成一次掩码并写入 cache_dir 根目录。"""
    if masks_already_cached(cache_dir):
        print(f"[MaskTrain] cache already has NPYs: {cache_dir}")
        normalize_cache_outputs(cache_dir)
        return

    if not frames:
        raise RuntimeError("ensure_cache_masks: no frames to generate from")

    f = ref_frame if ref_frame is not None else frames[0]
    frame_dir = src_root / f'frame_{f}'
    if not frame_dir.exists():
        raise FileNotFoundError(f"reference frame dir not found: {frame_dir}")

    print(f"[MaskTrain] cache empty → generate masks once from frame_{f} into {cache_dir}")
    # 关键：直接把输出写到 cache_dir 根目录，而不是 _gen_frame_* 临时子目录
    run_make_aabb_for_frame(aabb_json, frame_dir, cache_dir, aabb_close_px, close_px, dilate_px)

    # 将可能写入的 masked_gt/、masks_raw/ 等规整为根目录 *.npy，并清理子目录/残留 png
    normalize_cache_outputs(cache_dir)

# ------------------------- Mask application (RGB black) ------------------------
def apply_mask_black_rgb(src_img: Path, mask_path: Path, out_png: Path, invert: bool=False) -> None:
    import cv2
    img = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(str(src_img))
    if mask_path.suffix.lower() == '.npy':
        m = np.load(str(mask_path)); m = (m > 0).astype(np.uint8)
    else:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None: raise FileNotFoundError(str(mask_path))
        m = (m > 127).astype(np.uint8)
    H, W = img.shape[:2]
    if (m.shape[0] != H) or (m.shape[1] != W):
        import cv2
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    fg = m if not invert else (1 - m)
    if img.ndim == 2: rgb = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 3: rgb = img
    else: rgb = img[:, :, :3]
    out = rgb.copy()
    if out.dtype != np.uint8: out = np.clip(out, 0, 255).astype(np.uint8)
    out[fg == 0] = 0
    ensure_dir(out_png.parent)
    import cv2
    cv2.imwrite(str(out_png), out)

# ------------------------------ Workdir construction ---------------------------
@dataclass
class FramePlan:
    frame: int; src_dir: Path; work_dir: Path; model_dir: Path

def parse_frames(frames_arg: str, src_root: Path) -> List[int]:
    if frames_arg == 'all':
        frames: List[int] = []
        for d in sorted(src_root.glob('frame_*')):
            try: frames.append(int(d.name.split('_')[-1]))
            except Exception: pass
        return frames
    out: List[int] = []
    for part in frames_arg.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            a, b = part.split('-', 1); out.extend(list(range(int(a), int(b) + 1)))
        else: out.append(int(part))
    return sorted(list(dict.fromkeys(out)))

def _locate_dst_images_txt(work_dir: Path) -> Path:
    cand = work_dir / 'sparse' / '0' / 'images.txt'
    if cand.exists(): return cand
    cand = work_dir / 'sparse' / 'images.txt'
    if cand.exists(): return cand
    raise FileNotFoundError(f"images.txt not found under {work_dir}/sparse[/0]")

def build_work_dataset(src_frame_dir: Path, cache_dir: Path, work_dir: Path, invert_masks: bool=False) -> None:
    clean_dir(work_dir)
    src_sparse = src_frame_dir / 'sparse'
    if not src_sparse.exists(): raise FileNotFoundError(f"Missing sparse/: {src_sparse}")
    shutil.copytree(src_sparse, work_dir / 'sparse')

    dst_images_txt = _locate_dst_images_txt(work_dir)
    lines, recs = read_images_text(dst_images_txt)

    src_images_dir = src_frame_dir / 'images'
    if not src_images_dir.exists(): raise FileNotFoundError(f"{src_images_dir} not found")
    out_images = ensure_dir(work_dir / 'images')

    cache_map = {p.stem: p for p in cache_dir.glob('*.npy')}
    if not cache_map: raise FileNotFoundError(f"No mask .npy found in cache_dir: {cache_dir}")

    new_lines = list(lines)
    for rec in recs:
        img_name = rec['name']; img_id = rec['img_id']; idx = idx_str_for(img_name, img_id)
        src_img = src_images_dir / img_name
        if not src_img.exists(): src_img = src_images_dir / Path(img_name).name
        if not src_img.exists():
            print(f"[warn] missing GT: {src_img}"); continue
        mask_path = cache_map.get(Path(img_name).stem)
        if mask_path is None:
            raise FileNotFoundError(
                f"[mask-match] no mask by stem for {img_name}; "
                f"cache contains keys: {len(cache_map)}. "
                f"Refuse to fall back to numeric img_id to avoid cross-frame mismatch."
            )
        if mask_path is None:
            raise FileNotFoundError(f"mask not found for {img_name} (stem={Path(img_name).stem} / idx={idx}) under {cache_dir}")
        out_png = out_images / f"{idx}.png"
        apply_mask_black_rgb(src_img, mask_path, out_png, invert=invert_masks)

        li = rec['line_index']; toks = new_lines[li].split()
        if len(toks) >= 10:
            toks[9] = out_png.name
            new_lines[li] = ' '.join(toks) + '\n'

    with open(dst_images_txt, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# ------------------------------------- main ------------------------------------
def main(argv: List[str] | None = None) -> None:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser("Mask-then-Train (black-masked GT; single-workdir; no --masked)")
    ap.add_argument('-s','--source_root', required=True)
    ap.add_argument('-o','--output_root', required=True)
    ap.add_argument('--frames', default='all')
    ap.add_argument('--prefix', default='model_frame_')
    ap.add_argument('--per-frame-subdir', default=None)
    ap.add_argument('--resume-if-exists', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--mask_mode', choices=['aabb_only'], required=True)
    ap.add_argument('--aabb', required=True)
    ap.add_argument('--mask_cache_root', default=None)
    ap.add_argument('--invert_mask', action='store_true')
    ap.add_argument('--aabb_close_px', type=int, default=0)
    ap.add_argument('--close_px', type=int, default=0)
    ap.add_argument('--dilate_px', type=int, default=0)
    ap.add_argument('--', dest='dashdash', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--cache_ref_frame', type=int, default=None,
                help='仅使用该参考帧生成一次 AABB 掩码并复用到所有帧（不产生 _gen_frame_*）')
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv:
        dd = argv.index('--'); wrapper_args = argv[:dd]; train_flags = argv[dd+1:]
    else:
        wrapper_args = argv; train_flags = []
    args = ap.parse_args(wrapper_args)

    src_root = Path(args.source_root).resolve()
    out_root = Path(args.output_root).resolve(); ensure_dir(out_root)

    work_root = ensure_dir(Path.cwd() / '_work_datasets')
    cache_dir = ensure_dir((Path(args.mask_cache_root).resolve() if args.mask_cache_root else work_root / '_mask_cache' / args.mask_mode))

    frames = parse_frames(args.frames, src_root)
    print(f"[MaskTrain] Frames: {frames}")
    # for average-per-frame over the requested range
    frame_min = min(frames) if frames else 0
    frame_max = max(frames) if frames else -1
    frame_count_range = (frame_max - frame_min + 1) if frames else 0
    if not frames:
        print("[MaskTrain][ERROR] No frames selected or found under source_root."); sys.exit(1)

    if '--disable_viewer' not in train_flags:
        train_flags = train_flags + ['--disable_viewer']

    ensure_cache_masks(Path(args.aabb).resolve(), src_root, frames, cache_dir,
                    args.aabb_close_px, args.close_px, args.dilate_px,
                    ref_frame=args.cache_ref_frame)
    print(f"[MaskTrain] using mask cache: {cache_dir}")

    fixed_work = work_root / 'frame_work'

    @dataclass
    class _FramePlan(FramePlan): pass
    plans: List[FramePlan] = []
    for i in frames:
        fdir = src_root / f'frame_{i}'
        if not fdir.exists():
            print(f"[skip] frame_{i} missing: {fdir}"); continue
        model_dir = out_root / f"{args.prefix}{i}"
        if args.per_frame_subdir: model_dir = model_dir / args.per_frame_subdir
        ensure_dir(model_dir)
        plans.append(_FramePlan(i, fdir, fixed_work, model_dir))

    aabb_json = str(Path(args.aabb).resolve())

    exit_code = 0
    try:
        failed = 0
        ok_count = 0
        for p in plans:
            print(f"\n==== [frame_{p.frame}] ====")
            if args.resume_if_exists and (p.model_dir / 'chkpnt').exists():
                print('[resume-skip] trained artifacts exist ->', p.model_dir); continue
            try:
                build_work_dataset(p.src_dir, cache_dir, p.work_dir, invert_masks=args.invert_mask)
            except Exception as e:
                print(f"[train][FAIL] frame_{p.frame} build_work_dataset: {e}"); failed = 1; continue

            cmd = [sys.executable, '-u', 'train.py',
                   '--source_path', str(p.work_dir),
                   '--model_path', str(p.model_dir)] + train_flags
            print('[train][spawn]', ' '.join(cmd))
            if args.dry_run: continue
            env = os.environ.copy()
            env.setdefault('PYTHONHASHSEED','0')
            env['AABB_JSON'] = aabb_json
            print(f"[train][env] AABB_JSON={env['AABB_JSON']}")
            rc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), env=env, check=False).returncode
            if rc != 0:
                print(f"[train][FAIL] frame_{p.frame} rc={rc}"); failed = 1; continue
            print(f"[train][OK] frame_{p.frame} -> {p.model_dir}")
            ok_count += 1
        exit_code = failed
    finally:
        try:
            if work_root.exists(): shutil.rmtree(work_root)
            work_root.mkdir(parents=True, exist_ok=True)
            print(f"[cleanup] cleared {work_root}")
        except Exception as e:
            print(f"[cleanup][warn] failed to clear {work_root}: {e}")

    # ---- timing end & summary (runs AFTER cleanup) ----
    total_sec = time.perf_counter() - t0
    def _fmt(sec: float) -> str:
        m, s = divmod(sec, 60.0); h, m = divmod(m, 60.0)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    print("\n[time] total runtime (incl. cleanup):", _fmt(total_sec), f"({total_sec:.3f} s)")
    if frame_count_range > 0:
        avg_range = total_sec / frame_count_range
        print(f"[time] frames range: {frame_min}-{frame_max} (count={frame_count_range})")
        print("[time] avg per frame (range):", _fmt(avg_range), f"({avg_range:.3f} s)")
    if len(plans) > 0:
        avg_actual = total_sec / len(plans)
        print(f"[time] frames actually processed: {len(plans)}")
        print("[time] avg per frame (processed):", _fmt(avg_actual), f"({avg_actual:.3f} s)")

    sys.exit(exit_code)
        
if __name__ == '__main__':
    main()
