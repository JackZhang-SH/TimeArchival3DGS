#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from collections import defaultdict

# ---------- AABB ----------
def parse_aabb(aabb_json_path):
    data = json.loads(Path(aabb_json_path).read_text(encoding="utf-8"))
    def as_np(v): return np.array(v, dtype=float)
    if "min" in data and "max" in data:
        lo, hi = as_np(data["min"]), as_np(data["max"])
    elif "center" in data and ("size" in data or "extent" in data or "half_size" in data):
        c = as_np(data["center"])
        if "size" in data: s = as_np(data["size"]); lo, hi = c - 0.5*s, c + 0.5*s
        elif "extent" in data: s = as_np(data["extent"]); lo, hi = c - 0.5*s, c + 0.5*s
        else: hs = as_np(data["half_size"]); lo, hi = c - hs, c + hs
    elif "aabb" in data or "bbox" in data:
        arr = as_np(data.get("aabb", data.get("bbox"))); lo, hi = arr[:3], arr[3:]
    elif all(k in data for k in ("xmin","xmax","ymin","ymax","zmin","zmax")):
        lo = np.array([data["xmin"], data["ymin"], data["zmin"]], float)
        hi = np.array([data["xmax"], data["ymax"], data["zmax"]], float)
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
    elif m in ("SIMPLE_RADIAL","RADIAL","SIMPLE_RADIAL_FISHEYE","RADIAL_FISHEYE"):
        fx = fy = params[0]; cx = params[1]; cy = params[2]
    else:  # OPENCV/… — take first 4 as fx,fy,cx,cy
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

# ---------- world spacing estimation ----------
def estimate_world_spacing(pts, k_sample=10000, percentile=30):
    """
    估计点云在世界坐标中的“平均间距”。随机采样若干点，
    用 Open3D KDTree 查询最近邻距离，取给定分位数作为 s_world。
    """
    if pts.shape[0] < 2:
        return 0.02  # fallback: 2cm
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    n = min(k_sample, pts.shape[0])
    idxs = np.random.choice(pts.shape[0], n, replace=False)
    dists = []
    for i in idxs:
        # query K=2 to get the nearest neighbor excluding the point itself
        _, idx_nn, dist2 = kdt.search_knn_vector_3d(pcd.points[i], 2)
        if len(dist2) >= 2:
            d = np.sqrt(dist2[1])
            if np.isfinite(d) and d > 0:
                dists.append(d)
    if len(dists) == 0:
        return 0.02
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))

# ---------- z-buffer masks ----------
def zbuffer_valid_mask_constant(u, v, z, w, h, radius_px=0):
    """固定像素半径的命中掩码。"""
    depth = np.full((h, w), np.inf, dtype=np.float32)
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
        depth.flat[uniq] = np.minimum(depth.flat[uniq], mins.astype(np.float32))
    else:
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
            depth[ys, xs] = np.minimum(depth[ys, xs], zz)

    return np.isfinite(depth)

def zbuffer_valid_mask_adaptive(u, v, z, w, h, r_px_each, r_px_min=0, r_px_max=6):
    """
    深度自适应像素半径：
    r_px_each[i] 是第 i 个点的半径（整型）。为了效率，按半径分桶。
    """
    depth = np.full((h, w), np.inf, dtype=np.float32)

    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)
    r = np.asarray(r_px_each).astype(np.int32)

    valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (r >= r_px_min) & (r <= r_px_max)
    ui, vi, z, r = ui[valid], vi[valid], z[valid], r[valid]

    # 预生成各个半径的圆盘偏移
    unique_r = np.unique(r)
    offsets_dict = {}
    for R in unique_r:
        ofs = []
        for dy in range(-R, R+1):
            for dx in range(-R, R+1):
                if dx*dx + dy*dy <= R*R:
                    ofs.append((dy, dx))
        offsets_dict[int(R)] = np.array(ofs, dtype=np.int32)

    # 分桶更新
    buckets = defaultdict(list)
    for i in range(len(r)):
        buckets[int(r[i])].append(i)

    for R, idxs in buckets.items():
        idxs = np.array(idxs, dtype=np.int32)
        ofs = offsets_dict[R]
        for k in idxs:
            x = ui[k]; y = vi[k]; zz = z[k]
            ys = y + ofs[:,0]; xs = x + ofs[:,1]
            inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs = xs[inb]; ys = ys[inb]
            depth[ys, xs] = np.minimum(depth[ys, xs], zz)

    return np.isfinite(depth)

# ---------- per-view rendering ----------
def render_B_valid_mask_for_image(pts, q, t, intr, wh,
                                  adaptive=False, s_world=0.02,
                                  splat_scale=1.0, r_px_const=1,
                                  r_px_min=0, r_px_max=6):
    (fx, fy, cx, cy) = intr
    (w, h) = wh
    R = qvec2rotmat(q)
    Xc = (R @ pts.T + t.reshape(3,1)).T
    mask = Xc[:,2] > 1e-6
    if not np.any(mask):
        return np.zeros((h, w), dtype=bool)
    X = Xc[mask]
    u = fx * X[:,0]/X[:,2] + cx
    v = fy * X[:,1]/X[:,2] + cy
    z = X[:,2]

    if not adaptive:
        return zbuffer_valid_mask_constant(u, v, z, w, h, radius_px=int(r_px_const))

    # 自适应半径：r_px ≈ fx * (s_world * splat_scale) / z
    r_px_each = np.ceil((fx * (s_world * float(splat_scale))) / np.maximum(z, 1e-6)).astype(np.int32)
    r_px_each = np.clip(r_px_each, r_px_min, r_px_max)
    return zbuffer_valid_mask_adaptive(u, v, z, w, h, r_px_each,
                                       r_px_min=r_px_min, r_px_max=r_px_max)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Mask GT using ONLY B point hits inside AABB (no A depth).")
    ap.add_argument("--pc", required=True, help="B fused_points.ply")
    ap.add_argument("--aabb", required=True, help="aabb_B.json")
    ap.add_argument("--colmap", required=True, help="COLMAP root (contains sparse/ or sparse/0)")
    ap.add_argument("--gt_dir", default=None, help="GT images dir (default: <colmap>/images)")
    ap.add_argument("--out", required=True, help="Output root directory")

    # 常规（旧）固定半径 splat
    ap.add_argument("--splat_radius_px", type=int, default=1, help="Constant pixel radius for splatting (0/1/2).")

    # 新：自适应 splat（基于内参与深度）
    ap.add_argument("--adaptive_splat", action="store_true", help="Enable depth-adaptive splat radius.")
    ap.add_argument("--s_world_scale", type=float, default=1.0,
                    help="Scale for estimated world spacing when computing adaptive pixel radius.")
    ap.add_argument("--splat_px_min", type=int, default=1, help="Min pixel radius for adaptive splat.")
    ap.add_argument("--splat_px_max", type=int, default=6, help="Max pixel radius for adaptive splat.")

    # 形态学
    ap.add_argument("--close_px", type=int, default=0, help="Closing kernel radius in pixels (0=off).")
    ap.add_argument("--dilate_px", type=int, default=0, help="Final dilation radius in pixels (0=off).")

    ap.add_argument("--min_fg_frac", type=float, default=0.0,
                    help="If foreground fraction < this, skip writing (0=off).")
    ap.add_argument("--save_mask_png", action="store_true", help="Also save mask PNG for inspection.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_masked = out_dir / "masked_gt_Bonly"; out_masked.mkdir(parents=True, exist_ok=True)
    out_masks_png = out_dir / "masks_Bonly"
    if args.save_mask_png: out_masks_png.mkdir(parents=True, exist_ok=True)
    out_masks_npy = out_dir / "masks_raw_Bonly"; out_masks_npy.mkdir(parents=True, exist_ok=True)

    # 1) B 点云 → AABB 裁剪
    pcd = o3d.io.read_point_cloud(args.pc)
    if pcd.is_empty(): raise RuntimeError(f"Empty point cloud: {args.pc}")
    lo, hi = parse_aabb(args.aabb)
    mask_pts = crop_points_by_aabb(np.asarray(pcd.points), lo, hi)
    idx = np.where(mask_pts)[0]
    if idx.size == 0: raise RuntimeError("AABB crop produced empty set. Check aabb_B.json.")
    pts_B = np.asarray(pcd.points)[idx]

    # 若启用自适应，估计世界尺度 s_world
    s_world = None
    if args.adaptive_splat:
        s_world = estimate_world_spacing(pts_B, k_sample=10000, percentile=30)
        print(f"[info] estimated world spacing s_world ≈ {s_world:.4f} m (percentile=30%)")

    # 2) 读取相机
    colmap_root = Path(args.colmap)
    cams, imgs = load_colmap_model(colmap_root)
    gt_dir = Path(args.gt_dir) if args.gt_dir else (colmap_root / "images")

    # 3) 每张图：只要 B 命中就保留该像素
    for img_id, rec in imgs.items():
        name = rec["name"]
        idx_str = index_from_image_name(name, img_id)
        cam_id = rec["camera_id"]
        q, t = rec["qvec"], rec["tvec"]
        model, w, h, params = cams[cam_id]
        fx, fy, cx, cy = get_fx_fy_cx_cy(model, params)

        if args.adaptive_splat:
            Bmask = render_B_valid_mask_for_image(
                pts_B, q, t, (fx,fy,cx,cy), (w,h),
                adaptive=True, s_world=s_world, splat_scale=args.s_world_scale,
                r_px_const=args.splat_radius_px,  # ignored
                r_px_min=args.splat_px_min, r_px_max=args.splat_px_max
            )
        else:
            Bmask = render_B_valid_mask_for_image(
                pts_B, q, t, (fx,fy,cx,cy), (w,h),
                adaptive=False, s_world=0.02, splat_scale=1.0,
                r_px_const=args.splat_radius_px,
                r_px_min=0, r_px_max=0
            )

        # 形态学：先闭运算再膨胀（可选）
        if args.close_px > 0:
            k = 2*int(args.close_px) + 1
            Bmask = cv2.morphologyEx(Bmask.astype(np.uint8), cv2.MORPH_CLOSE,
                                     np.ones((k,k), np.uint8)).astype(bool)
        if args.dilate_px > 0:
            k = 2*int(args.dilate_px) + 1
            Bmask = cv2.dilate(Bmask.astype(np.uint8), np.ones((k,k), np.uint8), 1).astype(bool)

        fg_frac = float(np.count_nonzero(Bmask)) / (h*w)
        if args.min_fg_frac > 0 and fg_frac < args.min_fg_frac:
            print(f"[info] {idx_str}: FG {fg_frac:.4f} < {args.min_fg_frac}, skip.")
            continue

        # 读 GT 并应用掩码
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

        if img.ndim == 2: img = img[:, :, None]
        mask3 = np.repeat(Bmask[:, :, None], img.shape[2], axis=2)
        masked = (img.astype(np.float32) * mask3).astype(img.dtype)

        # 保存
        (out_dir / "masked_gt_Bonly").mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(out_dir / "masked_gt_Bonly" / f"{idx_str}.png"), masked)
        np.save(str(out_dir / "masks_raw_Bonly" / f"{idx_str}.npy"), Bmask.astype(np.float32))
        if args.save_mask_png:
            (out_dir / "masks_Bonly").mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(out_dir / "masks_Bonly" / f"{idx_str}.png"), (Bmask.astype(np.uint8)*255))

    print("[done] Masked GT (B-only) saved to:", out_dir / "masked_gt_Bonly")

if __name__ == "__main__":
    main()
