#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

# ---------- AABB ----------
def parse_aabb(aabb_json_path):
    data = json.loads(Path(aabb_json_path).read_text(encoding="utf-8"))
    def as_np(v): return np.array(v, dtype=float)
    if "min" in data and "max" in data:
        lo, hi = as_np(data["min"]), as_np(data["max"])
    elif "center" in data and ("size" in data or "extent" in data or "half_size" in data):
        c = as_np(data["center"])
        if "size" in data:
            s = as_np(data["size"]); lo, hi = c - 0.5*s, c + 0.5*s
        elif "extent" in data:
            s = as_np(data["extent"]); lo, hi = c - 0.5*s, c + 0.5*s
        else:
            hs = as_np(data["half_size"]); lo, hi = c - hs, c + hs
    elif "aabb" in data or "bbox" in data:
        arr = as_np(data.get("aabb", data.get("bbox"))); lo, hi = arr[:3], arr[3:]
    elif all(k in data for k in ("xmin","xmax","ymin","ymax","zmin","zmax")):
        lo = np.array([data["xmin"], data["ymin"], data["zmin"]], dtype=float)
        hi = np.array([data["xmax"], data["ymax"], data["zmax"]], dtype=float)
    else:
        raise ValueError("Unsupported AABB JSON structure")
    return np.minimum(lo, hi), np.maximum(lo, hi)

def crop_points_by_aabb(pts_xyz, lo, hi):
    return np.all((pts_xyz >= lo) & (pts_xyz <= hi), axis=1)

# ---------- COLMAP (text) ----------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def get_fx_fy_cx_cy(model, params):
    m = model.upper()
    if m == "SIMPLE_PINHOLE": fx = fy = params[0]; cx = params[1]; cy = params[2]
    elif m == "PINHOLE": fx, fy, cx, cy = params[:4]
    elif m in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = params[0]; cx = params[1]; cy = params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return float(fx), float(fy), float(cx), float(cy)

def read_cameras_text(path):
    cams = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"): continue
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
        if not l or l.startswith("#"): i += 1; continue
        toks = l.split()
        if len(toks) < 10: i += 1; continue
        img_id = int(toks[0])
        qvec = np.array(list(map(float, toks[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, toks[5:8])), dtype=np.float64)
        cam_id = int(toks[8])
        name = toks[9]
        imgs[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name}
        i += 2
    return imgs

def load_colmap_model(root: Path):
    base = root / "sparse" / "0"
    if not base.exists(): base = root / "sparse"
    cams = read_cameras_text(base / "cameras.txt")
    imgs = read_images_text(base / "images.txt")
    return cams, imgs

# ---------- index from image filename ----------
_digit_re = re.compile(r"^\d+$")
def index_from_image_name(name: str, image_id: int) -> str:
    stem = Path(name).stem
    return stem if _digit_re.match(stem) else f"{image_id:04d}"

# ---------- z-buffer w/ optional splat ----------
def zbuffer_rasterize(u, v, z, w, h, radius_px=0):
    depth = np.full((h, w), np.nan, dtype=np.float32)
    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)
    valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z = ui[valid], vi[valid], z[valid]

    if radius_px <= 0:
        flat_idx = vi * w + ui
        order = np.argsort(flat_idx, kind="mergesort")
        flat_idx, z = flat_idx[order], z[order]
        uniq, first = np.unique(flat_idx, return_index=True)
        mins = np.minimum.reduceat(z, first)
        depth.flat[uniq] = mins.astype(np.float32)
        return depth

    offsets = []
    R = int(radius_px)
    for dy in range(-R, R+1):
        for dx in range(-R, R+1):
            if dx*dx + dy*dy <= R*R:
                offsets.append((dy, dx))
    offsets = np.array(offsets, dtype=np.int32)

    for k in range(len(ui)):
        x = ui[k]; y = vi[k]; zz = z[k]
        ys = y + offsets[:,0]; xs = x + offsets[:,1]
        inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[inb]; ys = ys[inb]
        for yy, xx in zip(ys, xs):
            cur = depth[yy, xx]
            if np.isnan(cur) or zz < cur:
                depth[yy, xx] = zz
    return depth

# ---------- B depth rendering ----------
def render_B_depth_for_image(pts, q, t, intr, wh, splat_radius_px=0):
    (fx, fy, cx, cy) = intr
    (w, h) = wh
    R = qvec2rotmat(q)
    Xc = (R @ pts.T + t.reshape(3,1)).T
    mask = Xc[:,2] > 1e-6
    if not np.any(mask):
        return np.full((h, w), np.nan, dtype=np.float32)
    X = Xc[mask]
    u = fx * X[:,0]/X[:,2] + cx
    v = fy * X[:,1]/X[:,2] + cy
    return zbuffer_rasterize(u, v, X[:,2], w, h, radius_px=splat_radius_px)

# ---------- A-depth smoothness (for grass consistency) ----------
def depth_grad_mag(depth, valid):
    """
    仅在四邻域都有效的像素上计算中心差分梯度，其余为 NaN。
    返回与 depth 同形的梯度幅值（米/像素）。
    """
    h, w = depth.shape
    gx = np.full((h, w), np.nan, dtype=np.float32)
    gy = np.full((h, w), np.nan, dtype=np.float32)

    valid_lr = valid & np.roll(valid, 1, axis=1) & np.roll(valid, -1, axis=1)
    valid_ud = valid & np.roll(valid, 1, axis=0) & np.roll(valid, -1, axis=0)

    gx_vals = (np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1)) * 0.5
    gy_vals = (np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0)) * 0.5

    gx[valid_lr] = gx_vals[valid_lr]
    gy[valid_ud] = gy_vals[valid_ud]

    grad = np.sqrt(np.square(gx) + np.square(gy))
    return grad  # NaN 表示无法可靠估计

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Make masked GT by A-vs-B depth comparison inside AABB.")
    ap.add_argument("--pc", required=True, help="B fused_points.ply")
    ap.add_argument("--aabb", required=True, help="aabb_B.json")
    ap.add_argument("--colmap", required=True, help="COLMAP root containing sparse/ (B cameras)")
    ap.add_argument("--gt_dir", required=False, default=None,
                    help="Directory of GT RGB images. Default: <colmap>/images")
    ap.add_argument("--A_depth_dir", required=True,
                    help="Directory of A depth npy files (named depth_0000.npy etc.)")
    ap.add_argument("--out", required=True, help="Output root directory")

    # 差分 & 渲染参数
    ap.add_argument("--delta_thresh_m", type=float, default=0.03,
                    help="Foreground threshold in meters (A - B > τ)")
    ap.add_argument("--splat_radius_px", type=int, default=0,
                    help="Pixel radius for B depth splatting (0/1/2)")

    # 草地一致性抑制（可选，不做底部假设）
    ap.add_argument("--use_grass_depth_consistency", action="store_true",
                    help="Use A~B near-equality + low A-depth gradient to suppress grass-like pixels.")
    ap.add_argument("--grass_tol_m", type=float, default=0.03,
                    help="A and B depth difference tolerance (meters) for grass suppression.")
    ap.add_argument("--grass_grad_thresh", type=float, default=0.02,
                    help="Gradient magnitude threshold (m/pixel) in A-depth for grass suppression.")
    ap.add_argument("--save_grass_png", action="store_true",
                    help="Save the grass suppression mask as PNG for inspection.")

    # 形态学增强（差分之后）
    ap.add_argument("--dilate_px", type=int, default=0,
                    help="Post dilate mask by N pixels (expand players/ball).")
    ap.add_argument("--close_px", type=int, default=0,
                    help="Post closing with NxN kernel before dilate (fill small holes).")

    ap.add_argument("--save_mask_png", action="store_true", help="Also save mask PNG for inspection")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_masked = out_dir / "masked_gt"; out_masked.mkdir(parents=True, exist_ok=True)
    out_masks_png = out_dir / "masks"; 
    if args.save_mask_png: out_masks_png.mkdir(parents=True, exist_ok=True)
    out_masks_npy = out_dir / "masks_raw"; out_masks_npy.mkdir(parents=True, exist_ok=True)
    out_grass_png = out_dir / "grass_masks"; 
    if args.save_grass_png: out_grass_png.mkdir(parents=True, exist_ok=True)

    # 1) Load B point cloud and crop AABB
    pcd = o3d.io.read_point_cloud(args.pc)
    if pcd.is_empty(): raise RuntimeError(f"Empty point cloud: {args.pc}")
    lo, hi = parse_aabb(args.aabb)
    mask_pts = crop_points_by_aabb(np.asarray(pcd.points), lo, hi)
    idx = np.where(mask_pts)[0]
    if idx.size == 0: raise RuntimeError("AABB crop produced empty set. Check aabb_B.json.")
    pts_B = np.asarray(pcd.points)[idx]

    # 2) Load COLMAP model (B cameras) + figure out GT directory
    colmap_root = Path(args.colmap)
    cams, imgs = load_colmap_model(colmap_root)
    gt_dir = Path(args.gt_dir) if args.gt_dir else (colmap_root / "images")

    # 3) For each image: render B depth, load A depth, build mask, apply to GT
    for img_id, rec in imgs.items():  # keep original order / indexes
        name = rec["name"]
        idx_str = index_from_image_name(name, img_id)
        cam_id = rec["camera_id"]
        q, t = rec["qvec"], rec["tvec"]
        model, w, h, params = cams[cam_id]
        fx, fy, cx, cy = get_fx_fy_cx_cy(model, params)

        # B depth (from AABB-cropped B points)
        depth_B = render_B_depth_for_image(
            pts_B, q, t, (fx,fy,cx,cy), (w,h), splat_radius_px=args.splat_radius_px
        )

        # A depth: depth_0000.npy, index matching GT & camera
        A_path = Path(args.A_depth_dir) / f"depth_{idx_str}.npy"
        if not A_path.exists():
            print(f"[warn] Missing A depth: {A_path} (skip this view)")
            continue
        depth_A = np.load(str(A_path)).astype(np.float32)
        if depth_A.shape != (h, w):
            depth_A = cv2.resize(depth_A, (w,h), interpolation=cv2.INTER_NEAREST)

        # 基础差分：B 有效 且 (A 无效 或 A-B > τ)
        B_valid = np.isfinite(depth_B)
        A_valid = np.isfinite(depth_A)
        keep = B_valid & (~A_valid | ((depth_A - depth_B) > args.delta_thresh_m))

        # 可选：草地一致性抑制（A~B接近 且 A深度很平滑）
        if args.use_grass_depth_consistency:
            both_valid = A_valid & B_valid
            near_eq = np.zeros_like(depth_A, dtype=bool)
            near_eq[both_valid] = np.abs(depth_A[both_valid] - depth_B[both_valid]) <= args.grass_tol_m

            gradA = depth_grad_mag(depth_A, A_valid)  # NaN=未知
            low_grad = np.zeros_like(depth_A, dtype=bool)
            finite_grad = np.isfinite(gradA)
            low_grad[finite_grad] = gradA[finite_grad] <= args.grass_grad_thresh

            grass_mask = near_eq & low_grad
            keep[grass_mask] = False  # 抹掉草地样式的一致区域

            if args.save_grass_png:
                cv2.imwrite(str(out_grass_png / f"{idx_str}.png"), (grass_mask.astype(np.uint8)*255))

        # 形态学后处理
        if args.close_px > 0:
            k = 2 * int(args.close_px) + 1
            kernel = np.ones((k, k), np.uint8)
            keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        if args.dilate_px > 0:
            k = 2 * int(args.dilate_px) + 1
            kernel = np.ones((k, k), np.uint8)
            keep = cv2.dilate(keep.astype(np.uint8), kernel, iterations=1).astype(bool)

        mask_float = keep.astype(np.float32)

        # GT 颜色
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
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)

        # 应用掩码（背景→黑）
        if img.ndim == 2:
            img = img[:, :, None]
        mask_3 = np.repeat(mask_float[:, :, None], img.shape[2], axis=2)
        masked = (img.astype(np.float32) * mask_3).astype(img.dtype)

        # 保存
        cv2.imwrite(str(out_masked / f"{idx_str}.png"), masked)
        np.save(str(out_masks_npy / f"{idx_str}.npy"), mask_float)
        if args.save_mask_png:
            cv2.imwrite(str(out_masks_png / f"{idx_str}.png"), (keep.astype(np.uint8) * 255))

    print("[done] Masked GT images saved to:", out_masked)

if __name__ == "__main__":
    main()
