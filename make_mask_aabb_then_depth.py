#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AABB-first masking, then A-vs-B depth refinement.

Step-1: Project the 3D AABB onto each GT image and mask out pixels outside
        the AABB projection (no point cloud needed in this step).
Step-2: Within the AABB-projected region, refine by A-vs-B depth rules:
        Keep pixel if:
           (B_valid and ( (A_valid and (A - B > delta_far_m))  OR
                           (A_valid and abs(A - B) <= delta_ground_m) OR
                           (not A_valid and B_valid) ))
        Else mask it out.
This implements your "逐步删减法（AABB→深度差）" precisely.

Outputs:
  out/
    masked_gt/          # 最终masked图（背景→黑）
    masks_raw/          # 最终mask的npy（float32，0/1）
    masks_aabb/         # Step-1的AABB投影视图mask（可选保存）
    masks_depth/        # Step-2深度细化mask（可选保存）
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import cv2

# ---------------- AABB utils (reused style) ----------------
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
        lo = np.array([data["xmin"], data["ymin"], data["zmin"]], float)
        hi = np.array([data["xmax"], data["ymax"], data["zmax"]], float)
    else:
        raise ValueError("Unsupported AABB JSON structure")
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return lo, hi

# ---------------- COLMAP text IO (reused style) -------------
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
        cam_id = int(toks[8]); name = toks[9]
        imgs[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name}
        i += 2
    return imgs

def load_colmap_model(root: Path):
    base = root / "sparse" / "0"
    if not base.exists(): base = root / "sparse"
    cams = read_cameras_text(base / "cameras.txt")
    imgs = read_images_text(base / "images.txt")
    return cams, imgs

_digit_re = re.compile(r"^\d+$")
def index_from_image_name(name: str, image_id: int) -> str:
    stem = Path(name).stem
    return stem if _digit_re.match(stem) else f"{image_id:04d}"

# ---------------- AABB → image projection mask ----------------
def project_points_world_to_image(Xw, q, t, intr, wh):
    fx, fy, cx, cy = intr
    w, h = wh
    R = qvec2rotmat(q)
    Xc = (R @ Xw.T + t.reshape(3,1)).T
    z = Xc[:,2]
    # z<=0 直接标注为无效
    u = fx * Xc[:,0] / np.maximum(z, 1e-6) + cx
    v = fy * Xc[:,1] / np.maximum(z, 1e-6) + cy
    return u, v, z

def aabb_corners(lo, hi):
    x0,y0,z0 = lo; x1,y1,z1 = hi
    # 8 corners
    return np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float64)

def aabb_faces_idx():
    # 6 faces as quads (each will be filled as polygon)
    return [
        [0,1,2,3],  # z=z0
        [4,5,6,7],  # z=z1
        [0,1,5,4],  # y=y0
        [2,3,7,6],  # y=y1
        [1,2,6,5],  # x=x1
        [0,3,7,4],  # x=x0
    ]

def aabb_projection_mask(lo, hi, q, tvec, intr, wh):
    """
    Ray-based AABB mask: for each pixel, mark True if the camera ray
    intersects the AABB with t_max >= max(t_min, t_near).
    This is robust even when the camera is very close to / inside the box.
    """
    fx, fy, cx, cy = intr
    w, h = wh

    # R: world->cam, Rt: cam->world
    R = qvec2rotmat(q)
    Rt = R.T

    # camera center in world
    Cw = -Rt @ tvec

    # pixel grid (u, v) and ray direction in camera (dc)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    dc_x = (uu - cx) / fx
    dc_y = (vv - cy) / fy
    dc_z = np.ones_like(dc_x)  # pinhole => z = t

    # dir in world for each pixel
    dir_w_x = Rt[0,0]*dc_x + Rt[0,1]*dc_y + Rt[0,2]*dc_z
    dir_w_y = Rt[1,0]*dc_x + Rt[1,1]*dc_y + Rt[1,2]*dc_z
    dir_w_z = Rt[2,0]*dc_x + Rt[2,1]*dc_y + Rt[2,2]*dc_z

    # slab intersection (vectorized)
    eps = 1e-12
    inv_x = 1.0 / np.where(np.abs(dir_w_x) < eps, np.sign(dir_w_x)*eps + eps, dir_w_x)
    inv_y = 1.0 / np.where(np.abs(dir_w_y) < eps, np.sign(dir_w_y)*eps + eps, dir_w_y)
    inv_z = 1.0 / np.where(np.abs(dir_w_z) < eps, np.sign(dir_w_z)*eps + eps, dir_w_z)

    t1x = (lo[0] - Cw[0]) * inv_x; t2x = (hi[0] - Cw[0]) * inv_x
    tminx = np.minimum(t1x, t2x);  tmaxx = np.maximum(t1x, t2x)

    t1y = (lo[1] - Cw[1]) * inv_y; t2y = (hi[1] - Cw[1]) * inv_y
    tminy = np.minimum(t1y, t2y);  tmaxy = np.maximum(t1y, t2y)

    t1z = (lo[2] - Cw[2]) * inv_z; t2z = (hi[2] - Cw[2]) * inv_z
    tminz = np.minimum(t1z, t2z);  tmaxz = np.maximum(t1z, t2z)

    tmin = np.maximum(np.maximum(tminx, tminy), tminz)
    tmax = np.minimum(np.minimum(tmaxx, tmaxy), tmaxz)

    # 要求与可视方向一致且在相机前方（t > 0）
    # t_near 设一个很小的正数，防止数值粘连到相机中心
    t_near = 1e-6
    valid = (tmax >= np.maximum(tmin, t_near))

    return valid.astype(bool)


# ---------------- B depth (from B point cloud) ----------------
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

    # disk splat
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

def render_B_depth_for_image(pts, q, t, intr, wh, splat_radius_px=0):
    fx, fy, cx, cy = intr
    w, h = wh
    R = qvec2rotmat(q)
    Xc = (R @ pts.T + t.reshape(3,1)).T
    mask = Xc[:,2] > 1e-6
    if not np.any(mask):
        return np.full((h, w), np.nan, dtype=np.float32)
    X = Xc[mask]
    u = fx * X[:,0]/X[:,2] + cx
    v = fy * X[:,1]/X[:,2] + cy
    return zbuffer_rasterize(u, v, X[:,2], w, h, radius_px=splat_radius_px)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Two-stage masking: AABB projection then A-vs-B depth refinement.")
    ap.add_argument("--pc", required=True, help="B fused_points.ply")
    ap.add_argument("--aabb", required=True, help="aabb_B.json")
    ap.add_argument("--colmap", required=True, help="COLMAP root (contains sparse/ or sparse/0)")
    ap.add_argument("--gt_dir", default=None, help="GT images dir (default: <colmap>/images)")
    ap.add_argument("--A_depth_dir", required=True, help="Directory of A-depth npy (depth_0000.npy ...)")

    ap.add_argument("--out", required=True, help="Output root directory")

    # 渲染 / 形态学
    ap.add_argument("--splat_radius_px", type=int, default=0, help="B depth splat radius (0/1/2)")
    ap.add_argument("--aabb_close_px", type=int, default=0, help="Step-1 AABB mask closing radius (0=off)")
    ap.add_argument("--dilate_px", type=int, default=0, help="Final dilation after Step-2 (0=off)")
    ap.add_argument("--close_px", type=int, default=0, help="Final closing before dilation (0=off)")

    # 深度阈值
    ap.add_argument("--delta_far_m", type=float, default=0.80,
                    help="Threshold for 'B much nearer than A' (meters). Recommend: player~stands gap (e.g., 0.8~2.0 m).")
    ap.add_argument("--delta_ground_m", type=float, default=0.05,
                    help="Tolerance for 'A and B very close' (ground/flat).")

    # 额外输出
    ap.add_argument("--save_aabb_png", action="store_true", help="Save Step-1 AABB mask PNGs")
    ap.add_argument("--save_depth_png", action="store_true", help="Save Step-2 depth-refine mask PNGs")
    ap.add_argument("--save_mask_png", action="store_true", help="Save final mask PNGs")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_masked = out_dir / "masked_gt"; out_masked.mkdir(parents=True, exist_ok=True)
    out_masks_npy = out_dir / "masks_raw"; out_masks_npy.mkdir(parents=True, exist_ok=True)
    out_aabb_png = out_dir / "masks_aabb"
    out_depth_png = out_dir / "masks_depth"
    out_mask_png = out_dir / "masks"
    if args.save_aabb_png: out_aabb_png.mkdir(parents=True, exist_ok=True)
    if args.save_depth_png: out_depth_png.mkdir(parents=True, exist_ok=True)
    if args.save_mask_png: out_mask_png.mkdir(parents=True, exist_ok=True)

    # 读 B 点云（仅用于 Step-2 深度渲染）
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(args.pc)
    if pcd.is_empty(): raise RuntimeError(f"Empty point cloud: {args.pc}")
    # AABB 读取
    lo, hi = parse_aabb(args.aabb)

    # 仅保留 AABB 内点（加速 + 更干净的 B 深度）
    pts = np.asarray(pcd.points, dtype=np.float64)
    inb = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts_B = pts[inb]
    if pts_B.shape[0] == 0:
        raise RuntimeError("AABB crop produced empty point set. Check aabb_B.json.")

    # 载入 COLMAP
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

        # ---------- Step-1: AABB 投影 mask ----------
        mask_aabb = aabb_projection_mask(lo, hi, q, t, intr, wh)

        if args.aabb_close_px > 0:
            k = 2*int(args.aabb_close_px) + 1
            mask_aabb = cv2.morphologyEx(mask_aabb.astype(np.uint8), cv2.MORPH_CLOSE,
                                         np.ones((k,k), np.uint8)).astype(bool)

        if args.save_aabb_png:
            cv2.imwrite(str(out_aabb_png / f"{idx_str}.png"), (mask_aabb.astype(np.uint8) * 255))

        # ---------- Step-2: 深度细化 ----------
        depth_B = render_B_depth_for_image(pts_B, q, t, intr, wh, splat_radius_px=args.splat_radius_px)

        A_path = Path(args.A_depth_dir) / f"depth_{idx_str}.npy"
        if not A_path.exists():
            print(f"[warn] Missing A depth: {A_path} (skip view)")
            continue
        depth_A = np.load(str(A_path)).astype(np.float32)
        if depth_A.shape != (h, w):
            depth_A = cv2.resize(depth_A, (w, h), interpolation=cv2.INTER_NEAREST)

        A_valid = np.isfinite(depth_A)
        B_valid = np.isfinite(depth_B)

        # 规则：
        #  (1) 地面/非常接近： |A - B| <= delta_ground_m  => keep
        #  (2) B明显近于A：  (A - B) > delta_far_m      => keep
        #  (3) A无效但B有效（常见于背景A未重建处）：      => keep
        #  其他：                                                => mask
        near_ground = np.zeros_like(B_valid, dtype=bool)
        both_valid = A_valid & B_valid
        near_ground[both_valid] = np.abs(depth_A[both_valid] - depth_B[both_valid]) <= args.delta_ground_m

        b_much_nearer = np.zeros_like(B_valid, dtype=bool)
        b_much_nearer[both_valid] = (depth_A[both_valid] - depth_B[both_valid]) > args.delta_far_m

        keep_depth = (B_valid & (near_ground | b_much_nearer)) | (~A_valid & B_valid)

        if args.save_depth_png:
            cv2.imwrite(str(out_depth_png / f"{idx_str}.png"), (keep_depth.astype(np.uint8) * 255))

        # ---------- 合并 Step-1 & Step-2 ----------
        keep_final = mask_aabb & keep_depth

        # 可选：形态学平滑
        if args.close_px > 0:
            k = 2 * int(args.close_px) + 1
            keep_final = cv2.morphologyEx(keep_final.astype(np.uint8), cv2.MORPH_CLOSE,
                                          np.ones((k, k), np.uint8)).astype(bool)
        if args.dilate_px > 0:
            k = 2 * int(args.dilate_px) + 1
            keep_final = cv2.dilate(keep_final.astype(np.uint8),
                                    np.ones((k, k), np.uint8), iterations=1).astype(bool)

        # ---------- 应用到 GT ----------
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
        mask3 = np.repeat(keep_final[:, :, None], img.shape[2], axis=2)
        masked = (img.astype(np.float32) * mask3).astype(img.dtype)

        # 保存
        cv2.imwrite(str(out_masked / f"{idx_str}.png"), masked)
        np.save(str(out_masks_npy / f"{idx_str}.npy"), keep_final.astype(np.float32))
        if args.save_mask_png:
            cv2.imwrite(str(out_mask_png / f"{idx_str}.png"), (keep_final.astype(np.uint8) * 255))

    print("[done] Two-stage masked GT saved to:", out_masked)

if __name__ == "__main__":
    main()
