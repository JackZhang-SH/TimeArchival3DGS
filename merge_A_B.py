# merge_A_B.py — Merge A+B with shrink+feather on B, robust AABB parser, optional cull, and multiview mask voting
import os, sys, json, math, random
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

# ---------- AABB utils ----------
def smoothstep(edge0, edge1, x):
    import numpy as np
    t = np.clip((x - edge0) / max(1e-12, (edge1 - edge0)), 0.0, 1.0)
    return t*t*(3.0 - 2.0*t)

def _arr3(v):
    import numpy as np
    if isinstance(v, (list, tuple)) and len(v)==3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    if isinstance(v, dict) and all(k in v for k in ("x","y","z")):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float32)
    return None

def parse_aabb_any(obj):
    import numpy as np
    if isinstance(obj, dict):
        if "min" in obj and "max" in obj:
            mn, mx = _arr3(obj["min"]), _arr3(obj["max"])
            if mn is not None and mx is not None: return mn, mx
        for k in ("aabb","bounds","box"):
            if k in obj:
                sub = obj[k]
                if isinstance(sub, dict) and "min" in sub and "max" in sub:
                    mn, mx = _arr3(sub["min"]), _arr3(sub["max"])
                    if mn is not None and mx is not None: return mn, mx
                if isinstance(sub, (list, tuple)) and len(sub)==2:
                    mn, mx = _arr3(sub[0]), _arr3(sub[1])
                    if mn is not None and mx is not None: return mn, mx
        if "center" in obj:
            c = _arr3(obj["center"])
            if c is not None:
                for key in ("extent","half_size","size"):
                    if key in obj:
                        e = _arr3(obj[key])
                        if e is not None:
                            if key=="size": e = e*0.5
                            return c - e, c + e
        flat = ("xmin","xmax","ymin","ymax","zmin","zmax")
        if all(k in obj for k in flat):
            mn = [float(obj["xmin"]), float(obj["ymin"]), float(obj["zmin"])]
            mx = [float(obj["xmax"]), float(obj["ymax"]), float(obj["zmax"])]
            return np.array(mn, dtype=np.float32), np.array(mx, dtype=np.float32)
    if isinstance(obj, (list, tuple)) and len(obj)==2:
        mn, mx = _arr3(obj[0]), _arr3(obj[1])
        if mn is not None and mx is not None: return mn, mx
    die(f"AABB JSON not recognized.")

def feather_B_opacity_in_aabb(B, aabb_min, aabb_max, shrink_m=0.5, feather_m=0.5):
    import numpy as np
    xyz = B["xyz"]; op = B["opacity"]
    half_sizes = (aabb_max - aabb_min) * 0.5
    cap = float(np.maximum(0.0, np.min(half_sizes) - 1e-7))
    shrink_m = min(max(0.0, float(shrink_m)), cap)
    bmin_p = aabb_min + shrink_m
    bmax_p = aabb_max - shrink_m
    d = np.minimum.reduce([
        xyz[:,0]-bmin_p[0], bmax_p[0]-xyz[:,0],
        xyz[:,1]-bmin_p[1], bmax_p[1]-xyz[:,1],
        xyz[:,2]-bmin_p[2], bmax_p[2]-xyz[:,2]
    ])
    w = smoothstep(0.0, max(1e-6, float(feather_m)), d).astype("float32")
    B["opacity"] = (op[:,0] * w)[:,None]
    print(f"[feather B] shrink={shrink_m:.3f}, feather={feather_m:.3f} |"
          f" w=1:{int((w>=0.999).sum())}, 0<w<1:{int(((w>0)&(w<0.999)).sum())}, w=0:{int((w<=0).sum())}/{w.shape[0]}")

def cull_B_outside(B, aabb_min, aabb_max, shrink_m=0.0, mode="orig"):
    import numpy as np
    if mode not in ("orig","shrunken"): mode="orig"
    bmin = aabb_min.copy(); bmax = aabb_max.copy()
    if mode=="shrunken" and shrink_m>0: bmin += shrink_m; bmax -= shrink_m
    xyz = B["xyz"]
    inside = (
        (xyz[:,0] >= bmin[0]) & (xyz[:,0] <= bmax[0]) &
        (xyz[:,1] >= bmin[1]) & (xyz[:,1] <= bmax[1]) &
        (xyz[:,2] >= bmin[2]) & (xyz[:,2] <= bmax[2])
    )
    drop = int((~inside).sum())
    if drop>0:
        print(f"[cull] remove {drop} B pts outside {mode} box; keep {int(inside.sum())}.")
        for k in ("xyz","opacity","scale","rot","f_dc","f_rest"):
            B[k] = B[k][inside]
    else:
        print("[cull] no B points outside AABB to remove.")

# ---------- Multiview mask voting ----------
def load_colmap_simple(colmap_dir):
    cams_params = {}
    # cameras.txt
    with open(os.path.join(colmap_dir, "cameras.txt"), "r", encoding="utf-8") as f:
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
    with open(os.path.join(colmap_dir, "images.txt"), "r", encoding="utf-8") as f:
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

def build_depth_loader(depth_dir):
    from functools import lru_cache
    import numpy as np
    @lru_cache(maxsize=4096)
    def get_depth(img_name):
        base = os.path.splitext(os.path.basename(img_name))[0]
        path = os.path.join(depth_dir, base + ".npy")
        if not os.path.isfile(path): return None
        d = np.load(path)
        return d.astype(np.float32)
    return get_depth

def filter_B_by_multiview_masks(B, cams, masks_dir, mask_ext=".png", mask_dilate_px=3,
                                min_views=2, subsample_cams=0, depth_dir=None, depth_tol_m=0.5):
    import numpy as np
    if subsample_cams and 0 < subsample_cams < len(cams):
        cams = random.sample(cams, subsample_cams)
    get_mask = build_mask_loader(masks_dir, mask_ext, mask_dilate_px)
    get_depth = build_depth_loader(depth_dir) if depth_dir else (lambda name: None)

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
            if depth_dir:
                depth = get_depth(cam["name"])
                if depth is not None:
                    d_ok = np.zeros_like(ok, dtype=bool)
                    d_ok[ok] = np.abs(depth[vv[ok], uu[ok]] - zc[ok]) <= depth_tol_m
                    hit &= d_ok
            vt += hit.astype(np.int32)
        votes[idx] = vt

    keep = votes >= int(min_views)
    drop = int((~keep).sum())
    print(f"[mask vote] keep {int(keep.sum())} / {N}  (min_views={min_views}, subsample_cams={subsample_cams})")
    if drop > 0:
        for k in ("xyz","opacity","scale","rot","f_dc","f_rest"):
            B[k] = B[k][keep]

# ---------- main ----------
def main():
    ap = ArgumentParser("Merge fixed A and trained B (shrink+feather on B, optional cull & mask-vote)")
    ap.add_argument("--a_ply", required=True)
    ap.add_argument("--b_ply", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--feature_align", choices=["pad","trunc"], default="pad")
    ap.add_argument("--aabb_json", required=True)
    ap.add_argument("--shrink_m", type=float, default=0.5)
    ap.add_argument("--feather_m", type=float, default=0.5)
    ap.add_argument("--cull_outside", action="store_true")
    ap.add_argument("--cull_box", choices=["orig","shrunken"], default="orig")
    # new: multiview mask voting
    ap.add_argument("--keep_by_masks", action="store_true",
                    help="Enable multiview mask voting to keep only B points near person/ball.")
    ap.add_argument("--colmap_dir", type=str, default=None, help="Folder containing cameras.txt/images.txt")
    ap.add_argument("--masks_dir", type=str, default=None, help="Folder of per-view binary masks (same names).")
    ap.add_argument("--mask_ext", type=str, default=".png")
    ap.add_argument("--mask_dilate_px", type=int, default=0)
    ap.add_argument("--min_views", type=int, default=2)
    ap.add_argument("--subsample_cams", type=int, default=0)
    ap.add_argument("--depth_dir", type=str, default=None, help="Optional per-view depth .npy for consistency")
    ap.add_argument("--depth_tol_m", type=float, default=0.5)
    args = ap.parse_args()
    print("[args]", vars(args))

    import numpy as np, traceback
    try:
        A = read_ply_xyzcso(args.a_ply)
        B = read_ply_xyzcso(args.b_ply)
        print(f"[stat] A: N={A['xyz'].shape[0]}, f_rest={A['f_rest'].shape[1]} | "
              f"B: N={B['xyz'].shape[0]}, f_rest={B['f_rest'].shape[1]}")

        with open(args.aabb_json, "r", encoding="utf-8") as f:
            aabb_obj = json.load(f)
        aabb_min, aabb_max = parse_aabb_any(aabb_obj)
        print(f"[aabb] min={aabb_min.tolist()}  max={aabb_max.tolist()}")

        # (1) shrink+feather on B
        feather_B_opacity_in_aabb(B, aabb_min, aabb_max, args.shrink_m, args.feather_m)

        # (2) optional: physically cull B outside AABB
        if args.cull_outside:
            cull_B_outside(B, aabb_min, aabb_max, shrink_m=args.shrink_m, mode=args.cull_box)

        # (3) optional: multiview mask voting
        if args.keep_by_masks:
            if not args.colmap_dir or not args.masks_dir:
                die("--keep_by_masks needs both --colmap_dir and --masks_dir")
            cams = load_colmap_simple(args.colmap_dir)
            filter_B_by_multiview_masks(
                B, cams,
                masks_dir=args.masks_dir,
                mask_ext=args.mask_ext,
                mask_dilate_px=args.mask_dilate_px,
                min_views=args.min_views,
                subsample_cams=args.subsample_cams,
                depth_dir=args.depth_dir,
                depth_tol_m=args.depth_tol_m
            )

        # align f_rest columns
        ar, br = A["f_rest"].shape[1], B["f_rest"].shape[1]
        if ar != br:
            if args.feature_align == "pad":
                target = max(ar, br)
                import numpy as np
                if ar < target:
                    A["f_rest"] = np.pad(A["f_rest"], ((0,0),(0,target-ar)), mode="constant")
                    print(f"[align] PAD A.f_rest: {ar} -> {target}")
                if br < target:
                    B["f_rest"] = np.pad(B["f_rest"], ((0,0),(0,target-br)), mode="constant")
                    print(f"[align] PAD B.f_rest: {br} -> {target}")
            else:
                target = min(ar, br)
                A["f_rest"] = A["f_rest"][:, :target]
                B["f_rest"] = B["f_rest"][:, :target]
                print(f"[align] TRUNC to {target}")

        OUT = { k: np.concatenate([A[k], B[k]], axis=0) for k in ["xyz","opacity","scale","rot","f_dc","f_rest"] }
        write_ply_xyzcso(args.out_ply, OUT)
        print(f"[merge] wrote: {args.out_ply}")
        print(f"[merge] total N = {OUT['xyz'].shape[0]}  (A={A['xyz'].shape[0]} + B={B['xyz'].shape[0]})")
    except Exception as e:
        print("[exception]", repr(e))
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
