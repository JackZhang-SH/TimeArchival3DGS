#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AABB-only masking (single-stage)
--------------------------------
This script creates per-view foreground masks using **only** the AABB's
ray–box intersection against each camera ray (no depth comparison).

It mirrors the AABB projection logic you already use, and saves:
  out/
    masked_gt/   (GT image with background blacked out)
    masks/       (binary PNGs, if --save_mask_png)
    masks_raw/   (float32 .npy masks, 0/1)
    masks_aabb/  (alias of masks/ when --save_aabb_png)

Notes
-----
- Flags like --pc and --A_depth_dir are accepted for CLI *compatibility* but
  are ignored in this script.
- Optional morphology is provided to close tiny gaps or slightly expand masks.

Example (PowerShell multi-line with backticks is provided in chat).
"""

import argparse
from pathlib import Path
import json
import re
import numpy as np
import cv2

# ---------------- AABB utils ----------------

def parse_aabb(aabb_json_path):
    data = json.loads(Path(aabb_json_path).read_text(encoding="utf-8"))
    def as_np(v):
        return np.array(v, dtype=float)
    if "min" in data and "max" in data:
        lo, hi = as_np(data["min"]), as_np(data["max"])
    elif "center" in data and ("size" in data or "extent" in data or "half_size" in data):
        c = as_np(data["center"])
        if "size" in data:
            s = as_np(data["size"]); lo, hi = c - 0.5 * s, c + 0.5 * s
        elif "extent" in data:
            s = as_np(data["extent"]); lo, hi = c - 0.5 * s, c + 0.5 * s
        else:
            hs = as_np(data["half_size"]); lo, hi = c - hs, c + hs
    elif "aabb" in data or "bbox" in data:
        arr = as_np(data.get("aabb", data.get("bbox"))); lo, hi = arr[:3], arr[3:]
    elif all(k in data for k in ("xmin","xmax","ymin","ymax","zmin","zmax")):
        lo = np.array([data["xmin"], data["ymin"], data["zmin"]], float)
        hi = np.array([data["xmax"], data["ymax"], data["zmax"]], float)
    else:
        raise ValueError("Unsupported AABB JSON structure")
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return lo, hi

# ---------------- COLMAP text IO ----------------

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def get_fx_fy_cx_cy(model, params):
    m = model.upper()
    if m == "SIMPLE_PINHOLE":
        fx = fy = params[0]; cx = params[1]; cy = params[2]
    elif m == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif m in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = params[0]; cx = params[1]; cy = params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return float(fx), float(fy), float(cx), float(cy)


def read_cameras_text(path):
    cams = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            toks = ln.split()
            cam_id = int(toks[0]); model = toks[1]
            w, h = int(toks[2]), int(toks[3])
            params = np.array(list(map(float, toks[4:])), dtype=np.float64)
            cams[cam_id] = (model, w, h, params)
    return cams


def read_images_text(path):
    imgs = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if not l or l.startswith("#"):
            i += 1; continue
        toks = l.split()
        if len(toks) < 10:
            i += 1; continue
        img_id = int(toks[0])
        qvec = np.array(list(map(float, toks[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, toks[5:8])), dtype=np.float64)
        cam_id = int(toks[8]); name = toks[9]
        imgs[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name}
        i += 2
    return imgs


def load_colmap_model(root: Path):
    base = root / "sparse" / "0"
    if not base.exists():
        base = root / "sparse"
    cams = read_cameras_text(base / "cameras.txt")
    imgs = read_images_text(base / "images.txt")
    return cams, imgs

_digit_re = re.compile(r"^\d+$")

def index_from_image_name(name: str, image_id: int) -> str:
    stem = Path(name).stem
    return stem if _digit_re.match(stem) else f"{image_id:04d}"

# ---------------- Core: AABB → per-pixel mask ----------------

def aabb_projection_mask(lo, hi, q, tvec, intr, wh):
    """
    Ray–box (slab) intersection in world space:
    For each pixel, cast a ray from the camera center and set mask=True if the
    ray intersects the AABB with t_max >= max(t_min, t_near) and the hit lies
    in front of the camera.
    """
    fx, fy, cx, cy = intr
    w, h = wh

    R = qvec2rotmat(q)
    Rt = R.T

    # camera center in world
    Cw = -Rt @ tvec

    # pixel grid → ray direction in camera
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    dc_x = (uu - cx) / fx
    dc_y = (vv - cy) / fy
    dc_z = np.ones_like(dc_x)

    # ray dir in world (normalize not required for slab method)
    dir_w_x = Rt[0, 0] * dc_x + Rt[0, 1] * dc_y + Rt[0, 2] * dc_z
    dir_w_y = Rt[1, 0] * dc_x + Rt[1, 1] * dc_y + Rt[1, 2] * dc_z
    dir_w_z = Rt[2, 0] * dc_x + Rt[2, 1] * dc_y + Rt[2, 2] * dc_z

    eps = 1e-12
    inv_x = 1.0 / np.where(np.abs(dir_w_x) < eps, np.sign(dir_w_x) * eps + eps, dir_w_x)
    inv_y = 1.0 / np.where(np.abs(dir_w_y) < eps, np.sign(dir_w_y) * eps + eps, dir_w_y)
    inv_z = 1.0 / np.where(np.abs(dir_w_z) < eps, np.sign(dir_w_z) * eps + eps, dir_w_z)

    t1x = (lo[0] - Cw[0]) * inv_x; t2x = (hi[0] - Cw[0]) * inv_x
    tminx = np.minimum(t1x, t2x);  tmaxx = np.maximum(t1x, t2x)

    t1y = (lo[1] - Cw[1]) * inv_y; t2y = (hi[1] - Cw[1]) * inv_y
    tminy = np.minimum(t1y, t2y);  tmaxy = np.maximum(t1y, t2y)

    t1z = (lo[2] - Cw[2]) * inv_z; t2z = (hi[2] - Cw[2]) * inv_z
    tminz = np.minimum(t1z, t2z);  tmaxz = np.maximum(t1z, t2z)

    tmin = np.maximum(np.maximum(tminx, tminy), tminz)
    tmax = np.minimum(np.minimum(tmaxx, tmaxy), tmaxz)

    t_near = 1e-6
    valid = (tmax >= np.maximum(tmin, t_near))
    return valid.astype(bool)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="AABB-only masking (single-stage)")
    ap.add_argument("--pc", default=None, help="(ignored) for CLI compatibility")
    ap.add_argument("--A_depth_dir", default=None, help="(ignored) for CLI compatibility")

    ap.add_argument("--aabb", required=True, help="Path to aabb_B.json")
    ap.add_argument("--colmap", required=True, help="COLMAP root (contains sparse/ or sparse/0 and images/")
    ap.add_argument("--gt_dir", default=None, help="GT images dir (default: <colmap>/images)")
    ap.add_argument("--out", required=True, help="Output root directory")

    # Optional morphology
    ap.add_argument("--aabb_close_px", type=int, default=0, help="Close radius applied to raw AABB mask")
    ap.add_argument("--close_px", type=int, default=0, help="Final closing before dilation")
    ap.add_argument("--dilate_px", type=int, default=0, help="Final dilation after closing")

    # Optional image saves
    ap.add_argument("--save_aabb_png", action="store_true", help="Save raw AABB masks (PNG)")
    ap.add_argument("--save_mask_png", action="store_true", help="Save final masks (PNG)")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_masked = out_dir / "masked_gt"; out_masked.mkdir(parents=True, exist_ok=True)
    out_masks_npy = out_dir / "masks_raw"; out_masks_npy.mkdir(parents=True, exist_ok=True)
    out_aabb_png = out_dir / "masks_aabb"
    out_mask_png = out_dir / "masks"
    if args.save_aabb_png: out_aabb_png.mkdir(parents=True, exist_ok=True)
    if args.save_mask_png: out_mask_png.mkdir(parents=True, exist_ok=True)

    # Load AABB
    lo, hi = parse_aabb(args.aabb)

    # Load COLMAP model + images
    colmap_root = Path(args.colmap)
    cams, imgs = load_colmap_model(colmap_root)
    gt_dir = Path(args.gt_dir) if args.gt_dir else (colmap_root / "images")

    for img_id, rec in imgs.items():
        name = rec["name"]
        idx_str = index_from_image_name(name, img_id)
        cam_id = rec["camera_id"]
        q, t = rec["qvec"], rec["tvec"]
        model, w, h, params = cams[cam_id]
        fx, fy, cx, cy = get_fx_fy_cx_cy(model, params)
        intr = (fx, fy, cx, cy)
        wh = (w, h)

        # ----- AABB projection → mask -----
        mask = aabb_projection_mask(lo, hi, q, t, intr, wh)

        # Optional pre-close (tighten boxes & remove pinholes)
        if args.aabb_close_px > 0:
            k = 2 * int(args.aabb_close_px) + 1
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)

        # Optional final smooth
        if args.close_px > 0:
            k = 2 * int(args.close_px) + 1
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)
        if args.dilate_px > 0:
            k = 2 * int(args.dilate_px) + 1
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)

        # Save raw mask images if requested
        if args.save_aabb_png:
            cv2.imwrite(str(out_aabb_png / f"{idx_str}.png"), (mask.astype(np.uint8) * 255))
        if args.save_mask_png:
            cv2.imwrite(str(out_mask_png / f"{idx_str}.png"), (mask.astype(np.uint8) * 255))

        # Apply mask on GT image
        gt_path1 = gt_dir / name
        gt_path2 = colmap_root / name
        gt_path = gt_path1 if gt_path1.exists() else gt_path2
        if not gt_path.exists():
            print(f"[warn] Missing GT image: {gt_path1} / {gt_path2} (skip)")
            continue

        img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] Failed to read GT image: {gt_path} (skip)")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        if img.ndim == 2:
            img = img[:, :, None]
        mask3 = np.repeat(mask[:, :, None], img.shape[2], axis=2)
        masked = (img.astype(np.float32) * mask3).astype(img.dtype)

        # Save masked result + raw npy
        (out_masked / f"{idx_str}.png").parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_masked / f"{idx_str}.png"), masked)
        np.save(str(out_masks_npy / f"{idx_str}.npy"), mask.astype(np.float32))

    print("[done] AABB-only masked GT saved to:", out_masked)


if __name__ == "__main__":
    main()
