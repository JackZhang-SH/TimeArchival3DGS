
# merge_A_B_batch.py — Batch merge A (static) with all B frames (latest iteration per frame),
#                        using ONLY multiview mask voting for B before merging.
#
# Key behavior:
#   - For each frame under --b_root (supports "model_frame_n" or "frame_n"):
#       * Find the latest .../point_cloud/iteration_*/point_cloud.ply as B.
#       * Auto-set colmap_dir = <frame_dir>/sparse/0
#       * Auto-set masks_dir  = <frame_dir>/masks_residual
#       * Run multiview mask voting on B with those paths (no user flags for them).
#       * Feature-align A/B (pad or trunc) and write merged PLY to --out_root/frame_n/point_cloud_merged.ply
#
# Usage (PowerShell example):
#   python merge_A_B_batch.py `
#     --a_ply output_seq/static_stadium/model_frame_1/point_cloud/iteration_16000/point_cloud.ply `
#     --b_root output_seq/soccer_dynamic_player_masked_aabb `
#     --out_root output_seq/merged `
#     --feature_align pad `
#     --mask_ext .png --mask_dilate_px 0 --min_views 25 --subsample_cams 0
#
# Dependencies:
#   pip install plyfile numpy opencv-python
#

import os, re, sys, json, random
from argparse import ArgumentParser

def die(msg, code=1):
    print(f"[error] {msg}")
    sys.exit(code)

# ---------- PLY IO ----------
def read_ply_xyzcso(path):
    import numpy as np
    try:
        import plyfile
    except ImportError:
        die("Missing dependency: plyfile. pip install plyfile")
    if not os.path.isfile(path): die(f"file not found: {path}")
    ply = plyfile.PlyData.read(path)
    if "vertex" not in ply: die("PLY has no 'vertex' element.")
    v = ply["vertex"].data; names = v.dtype.names

    need = ["x","y","z","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"]
    for k in need:
        if k not in names:
            die(f"PLY field '{k}' missing. Got fields: {names}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("float32")
    op  = v["opacity"].astype("float32")[:,None]
    sc  = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype("float32")
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype("float32")
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype("float32")

    f_list = []; k = 0
    while True:
        name = f"f_rest_{k}"
        if name in names:
            f_list.append(v[name].astype("float32")[:,None]); k += 1
        else:
            break
    f_rest = np.concatenate(f_list, axis=1) if f_list else np.zeros((xyz.shape[0],0), "float32")
    return dict(xyz=xyz, opacity=op, scale=sc, rot=rot, f_dc=f_dc, f_rest=f_rest)

def write_ply_xyzcso(path, data):
    import numpy as np
    try:
        import plyfile
    except ImportError:
        die("Missing dependency: plyfile. pip install plyfile")
    xyz, op, sc, rot, f_dc, f_rest = [data[k] for k in ["xyz","opacity","scale","rot","f_dc","f_rest"]]
    n = xyz.shape[0]
    props = [("x","f4"),("y","f4"),("z","f4"),("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4")]
    for i in range(f_rest.shape[1]): props.append((f"f_rest_{i}", "f4"))
    props += [("opacity","f4"),
              ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
              ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    arr = np.zeros(n, dtype=props)
    arr["x"]=xyz[:,0]; arr["y"]=xyz[:,1]; arr["z"]=xyz[:,2]
    arr["f_dc_0"]=f_dc[:,0]; arr["f_dc_1"]=f_dc[:,1]; arr["f_dc_2"]=f_dc[:,2]
    for i in range(f_rest.shape[1]): arr[f"f_rest_{i}"]=f_rest[:,i]
    arr["opacity"]=op[:,0]
    arr["scale_0"]=sc[:,0]; arr["scale_1"]=sc[:,1]; arr["scale_2"]=sc[:,2]
    arr["rot_0"]=rot[:,0]; arr["rot_1"]=rot[:,1]; arr["rot_2"]=rot[:,2]; arr["rot_3"]=rot[:,3]
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plyfile.PlyData([plyfile.PlyElement.describe(arr, "vertex")], text=False).write(path)

# ---------- COLMAP + mask voting ----------
def load_colmap_simple(colmap_dir):
    cams_params = {}
    camfile = os.path.join(colmap_dir, "cameras.txt")
    imgfile = os.path.join(colmap_dir, "images.txt")
    if not (os.path.isfile(camfile) and os.path.isfile(imgfile)):
        die(f"COLMAP files not found in {colmap_dir}")
    with open(camfile, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            toks = line.strip().split()
            cam_id = int(toks[0]); model = toks[1]
            w = int(toks[2]); h = int(toks[3]); params = list(map(float, toks[4:]))
            if model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            cams_params[cam_id] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)

    cams = []
    with open(imgfile, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            toks = line.strip().split()
            if len(toks) < 10: continue
            img_id = int(toks[0])
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz = map(float, toks[5:8])
            cam_id = int(toks[8]); name = toks[9]
            import numpy as np
            q = np.array([qw,qx,qy,qz], dtype=np.float64)
            q = q / (np.linalg.norm(q) + 1e-12)
            w,x,y,z = q
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
                [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
            ], dtype=np.float64)
            t = np.array([tx,ty,tz], dtype=np.float64)
            intr = cams_params[cam_id]
            cams.append(dict(id=img_id, name=name, R=R, t=t, **intr))
    return cams

def project_points(P, cam):
    import numpy as np
    R, t = cam["R"], cam["t"]
    Xc = (R @ P.T + t.reshape(3,1)).T  # N,3
    zc = Xc[:,2]
    valid = zc > 1e-6
    u = cam["fx"] * (Xc[:,0]/zc) + cam["cx"]
    v = cam["fy"] * (Xc[:,1]/zc) + cam["cy"]
    return u, v, zc, valid

def build_mask_loader(masks_dir, mask_ext, dilate_px=0):
    from functools import lru_cache
    import numpy as np, cv2
    @lru_cache(maxsize=4096)
    def get_mask(img_name):
        base = os.path.splitext(os.path.basename(img_name))[0]
        path = os.path.join(masks_dir, base + mask_ext)
        if not os.path.isfile(path): return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None: return None
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
            m = cv2.dilate(m, k)
        return (m > 127)
    return get_mask

def filter_B_by_multiview_masks(B, cams, masks_dir, mask_ext=".png", mask_dilate_px=3,
                                min_views=2, subsample_cams=0):
    import numpy as np
    if subsample_cams and 0 < subsample_cams < len(cams):
        cams = random.sample(cams, subsample_cams)
    get_mask = build_mask_loader(masks_dir, mask_ext, mask_dilate_px)

    P = B["xyz"].astype(np.float64); N = P.shape[0]
    votes = np.zeros(N, dtype=np.int32)

    bs = 200000
    for i in range(0, N, bs):
        idx = slice(i, min(i+bs, N))
        Pblk = P[idx]
        vt = np.zeros(Pblk.shape[0], dtype=np.int32)
        for cam in cams:
            m = get_mask(cam["name"])
            if m is None: continue
            H, W = m.shape
            u, v, zc, valid = project_points(Pblk, cam)
            uu = np.round(u).astype(np.int64)
            vv = np.round(v).astype(np.int64)
            ok = valid & (uu>=0) & (uu<W) & (vv>=0) & (vv<H)
            if not ok.any(): continue
            hit = np.zeros_like(ok, dtype=bool)
            hit[ok] = m[vv[ok], uu[ok]]
            vt += hit.astype(np.int32)
        votes[idx] = vt

    keep = votes >= int(min_views)
    drop = int((~keep).sum())
    print(f"[mask vote] keep {int(keep.sum())} / {N}  (min_views={min_views}, subsample_cams={subsample_cams})")
    if drop > 0:
        for k in ("xyz","opacity","scale","rot","f_dc","f_rest"):
            B[k] = B[k][keep]

# ---------- Frame discovery ----------
def find_latest_B_ply(b_frame_dir):
    pc_root = os.path.join(b_frame_dir, "point_cloud")
    if not os.path.isdir(pc_root):
        return (None, None)
    best_iter = -1
    best_path = None
    it_pat = re.compile(r"^iteration_(\d+)$")
    for name in os.listdir(pc_root):
        m = it_pat.match(name)
        if not m: continue
        try:
            it = int(m.group(1))
        except Exception:
            continue
        ply_path = os.path.join(pc_root, name, "point_cloud.ply")
        if os.path.isfile(ply_path) and it > best_iter:
            best_iter = it
            best_path = ply_path
    return best_path, best_iter

def list_model_frames(b_root):
    frames = []
    pat = re.compile(r"^(model_frame_|frame_)(\d+)$")
    for name in os.listdir(b_root):
        path = os.path.join(b_root, name)
        if not os.path.isdir(path): continue
        m = pat.match(name)
        if not m: continue
        n = int(m.group(2))
        frames.append((n, path))
    frames.sort(key=lambda x: x[0])
    return frames

# ---------- Merge one frame ----------
def merge_one_frame(A, B_path, out_ply, cams, masks_dir, feature_align="pad",
                    mask_ext=".png", mask_dilate_px=0, min_views=2, subsample_cams=0):
    import numpy as np
    B = read_ply_xyzcso(B_path)
    print(f"[stat] B: {B['xyz'].shape[0]} points | f_rest={B['f_rest'].shape[1]} | {B_path}")

    # (only route) multiview mask voting on B
    filter_B_by_multiview_masks(
        B, cams,
        masks_dir=masks_dir, mask_ext=mask_ext, mask_dilate_px=mask_dilate_px,
        min_views=min_views, subsample_cams=subsample_cams
    )

    # Feature alignment between A and B
    ar, br = A["f_rest"].shape[1], B["f_rest"].shape[1]
    if ar != br:
        if feature_align == "pad":
            target = max(ar, br)
            if ar < target:
                A_pad = np.pad(A["f_rest"], ((0,0),(0,target-ar)), mode="constant")
            else:
                A_pad = A["f_rest"]
            if br < target:
                B["f_rest"] = np.pad(B["f_rest"], ((0,0),(0,target-br)), mode="constant")
            A_use = dict(A); A_use["f_rest"] = A_pad
            print(f"[align] PAD f_rest: A {ar} -> {target}, B {br} -> {target}")
        else:
            target = min(ar, br)
            A_use = dict(A); A_use["f_rest"] = A["f_rest"][:, :target]
            B["f_rest"] = B["f_rest"][:, :target]
            print(f"[align] TRUNC f_rest to {target}")
    else:
        A_use = A

    # Concat & write
    OUT = { k: np.concatenate([A_use[k], B[k]], axis=0) for k in ["xyz","opacity","scale","rot","f_dc","f_rest"] }
    write_ply_xyzcso(out_ply, OUT)
    print(f"[merge] wrote: {out_ply} | total N={OUT['xyz'].shape[0]} (A={A_use['xyz'].shape[0]} + B={B['xyz'].shape[0]})")

# ---------- main ----------
def main():
    ap = ArgumentParser("Batch merge fixed A with all B frames (latest iteration per frame) using ONLY multiview mask voting")
    ap.add_argument("--a_ply", required=True, help="Path to static A PLY")
    ap.add_argument("--b_root", required=True, help="Root dir containing model_frame_n for B")
    ap.add_argument("--out_root", required=True, help="Output root; we will create out_root/frame_{n}/point_cloud_merged.ply")
    ap.add_argument("--feature_align", choices=["pad","trunc"], default="pad")
    # voting controls
    ap.add_argument("--mask_ext", type=str, default=".png")
    ap.add_argument("--mask_dilate_px", type=int, default=0)
    ap.add_argument("--min_views", type=int, default=2)
    ap.add_argument("--subsample_cams", type=int, default=0)
    args = ap.parse_args()
    print("[args]", vars(args))

    # load A once
    A = read_ply_xyzcso(args.a_ply)
    print(f"[stat] A: N={A['xyz'].shape[0]}, f_rest={A['f_rest'].shape[1]} | {args.a_ply}")

    # iterate frames
    frames = list_model_frames(args.b_root)
    if not frames:
        die(f"No model_frame_n found under: {args.b_root}")
    for n, frame_dir in frames:
        ply_path, itN = find_latest_B_ply(frame_dir)
        if not ply_path:
            print(f"[skip] no B ply found in {os.path.relpath(frame_dir, args.b_root)}")
            continue
        # auto paths for this frame
        colmap_dir = os.path.join(frame_dir, "sparse", "0")
        masks_dir  = os.path.join(frame_dir, "masks_residual")
        if not os.path.isdir(colmap_dir):
            die(f"Missing COLMAP dir for frame {n}: {colmap_dir}")
        if not os.path.isdir(masks_dir):
            die(f"Missing masks_residual for frame {n}: {masks_dir}")

        cams = load_colmap_simple(colmap_dir)

        print(f"\n[frame {n}] latest iteration = {itN} | ply = {ply_path}")
        print(f"[frame {n}] colmap_dir = {colmap_dir}")
        print(f"[frame {n}] masks_dir  = {masks_dir}")

        out_dir = os.path.join(args.out_root, f"frame_{n}")
        os.makedirs(out_dir, exist_ok=True)
        out_ply = os.path.join(out_dir, "point_cloud_merged.ply")

        # Merge for this frame (multiview mask voting only)
        merge_one_frame(
            A, ply_path, out_ply, cams, masks_dir,
            feature_align=args.feature_align,
            mask_ext=args.mask_ext, mask_dilate_px=args.mask_dilate_px,
            min_views=args.min_views, subsample_cams=args.subsample_cams
        )

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        raise
    except Exception as e:
        import traceback
        print("[exception]", repr(e))
        traceback.print_exc()
        sys.exit(2)
