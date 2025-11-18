#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-merge A (static) with all B frames (latest iteration per frame),
replicating the behavior of:
  merge_A_B.py (shrink+feather, optional cull, multiview mask voting)
+ make_residual_masks.py (auto-generate masks_residual if missing)

B point cloud     : from B MODEL ROOT -> model_frame_n/point_cloud/iteration_*/point_cloud.ply (pick max iter)
COLMAP + GT imgs  : from B DATASET ROOT -> frame_n/sparse/0 and frame_n/images
A-only renders    : EITHER
    (A) --a_images_root   -> frame_n/images  (per-frame root)
 OR (B) --a_images_single -> a single images folder reused for all frames
 Masks directory  : frame_n/masks_residual (use if exists; otherwise auto-make if A images provided)
 Output           : out_root/<prefix>n/point_cloud_merged/point_cloud.ply  (default prefix='model_frame_')

Timing:
  - Per-frame breakdown is printed:
      [time][frame n] masks_residual = X.XXX s
      [time][frame n] merge+colmap   = Y.YYY s
      [time][frame n] frame total    = Z.ZZZ s
  - A final summary over all merged frames is printed at the end.
"""

import os, re, sys, json, random, subprocess, time
from argparse import ArgumentParser


def die(msg, code=1):
    print(f"[error] {msg}")
    sys.exit(code)

# ---------- PLY IO ----------
def _stackf(lst):
    import numpy as np
    return np.stack([x for x in lst], axis=1).astype("float32")

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

    need = ["x","y","z","opacity","scale_0","scale_1","scale_2",
            "rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"]
    for k in need:
        if k not in names:
            die(f"PLY field '{k}' missing. Got fields: {names}")

    xyz = _stackf([v["x"], v["y"], v["z"]])
    op  = v["opacity"].astype("float32")[:,None]
    sc  = _stackf([v["scale_0"], v["scale_1"], v["scale_2"]])
    rot = _stackf([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]])
    f_dc = _stackf([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]])
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
                    mn, mx = _arr3(sub["min"]), _arr3(sub["max"]); 
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
    die("AABB JSON not recognized.")

def feather_B_opacity_in_aabb(B, aabb_min, aabb_max, shrink_m=0.0, feather_m=0.0):
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
    print(f"[feather B] shrink={shrink_m:.3f}, feather={feather_m:.3f} | "
          f"w=1:{int((w>=0.999).sum())}, 0<w<1:{int(((w>0)&(w<0.999)).sum())}, w=0:{int((w<=0).sum())}/{w.shape[0]}")

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

# ---------- COLMAP + masks voting ----------
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
    import numpy as np
    with open(imgfile, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            toks = line.strip().split()
            if len(toks) < 10: continue
            img_id = int(toks[0])
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz = map(float, toks[5:8])
            cam_id = int(toks[8]); name = toks[9]
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
    Xc = (R @ P.T + t.reshape(3,1)).T
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

def filter_B_by_multiview_masks(B, cams, masks_dir, mask_ext=".png", mask_dilate_px=0,
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

# ---------- Frame discovery (MODEL root) ----------
def find_latest_B_ply(model_frame_dir):
    pc_root = os.path.join(model_frame_dir, "point_cloud")
    if not os.path.isdir(pc_root):
        return (None, None)
    best_iter = -1
    best_path = None
    it_pat = re.compile(r"^iteration_(\d+)$")
    for name in os.listdir(pc_root):
        m = it_pat.match(name)
        if not m: continue
        it = int(m.group(1))
        ply_path = os.path.join(pc_root, name, "point_cloud.ply")
        if os.path.isfile(ply_path) and it > best_iter:
            best_iter = it
            best_path = ply_path
    return best_path, best_iter

def list_model_frames(model_root):
    frames = []
    pat = re.compile(r"^(model_frame_|frame_)(\d+)$")
    for name in os.listdir(model_root):
        path = os.path.join(model_root, name)
        if not os.path.isdir(path): continue
        m = pat.match(name)
        if not m: continue
        n = int(m.group(2))
        frames.append((n, path))
    frames.sort(key=lambda x: x[0])
    return frames

# ---------- Auto make residual masks if missing ----------
def dir_empty_or_missing(p):
    return (not os.path.isdir(p)) or (len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))]) == 0)

def ensure_masks_residual(frame_n, b_dataset_root, a_images_root, a_images_single,
                          gt_ext=".png", a_ext=".png",
                          thr=25, blur_px=0, open_px=0, close_px=1, dilate_px=5):
    """Create masks_residual for this frame if missing.
       Priority:
         1) if masks_residual exists -> reuse it
         2) else if a_images_root provided -> use a_images_root/frame_n/images
         3) else if a_images_single provided -> use that single images folder for all frames
         4) else -> error
    """
    masks_dir = os.path.join(b_dataset_root, f"frame_{frame_n}", "masks_residual")
    if not dir_empty_or_missing(masks_dir):
        print(f"[auto-masks] frame {frame_n}: reuse existing masks_residual -> {masks_dir}")
        return masks_dir

    if a_images_root:
        gt_dir = os.path.join(b_dataset_root, f"frame_{frame_n}", "images")
        a_dir  = os.path.join(a_images_root,  f"frame_{frame_n}", "images")
    elif a_images_single:
        gt_dir = os.path.join(b_dataset_root, f"frame_{frame_n}", "images")
        a_dir  = a_images_single  # single directory reused for all frames
    else:
        die(f"masks_residual missing and no --a_images_root / --a_images_single provided for frame {frame_n}: {masks_dir}")

    os.makedirs(masks_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "make_residual_masks.py"),
        "--gt_dir", gt_dir, "--a_dir", a_dir, "--out_dir", masks_dir,
        "--gt_ext", gt_ext, "--a_ext", a_ext,
        "--thr", str(thr),
        "--blur_px", str(blur_px),
        "--open_px", str(open_px),
        "--close_px", str(close_px),
        "--dilate_px", str(dilate_px),
    ]
    print(f"[auto-masks][frame {frame_n}]", " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    t1 = time.perf_counter()
    if r.returncode != 0:
        die(f"make_residual_masks.py failed for frame {frame_n} (exit={r.returncode})")
    print(f"[auto-masks][frame {frame_n}] generated residual masks in {t1 - t0:.3f} s -> {masks_dir}")
    return masks_dir

def merge_one_frame(
    A,
    B_path,
    out_ply,
    cams,
    masks_dir,
    aabb_json=None,
    shrink_m=0.0,
    feather_m=0.0,
    cull_outside=False,
    cull_box="orig",
    feature_align="pad",
    mask_ext=".png",
    mask_dilate_px=0,
    min_views=2,
    subsample_cams=0,
    filtered_b_ply=None,
    write_merged=True,
):
    """
    Merge one frame of B with static A.

    Steps:
      1) Load B PLY.
      2) (Optional) AABB-based feathering / culling on B.
      3) Multiview mask voting on B.
      4) Feature alignment between A and B (pad or trunc f_rest).
      5) (Optional) Save filtered B-only PLY.
      6) Concatenate A + B and write merged PLY.
    """
    import numpy as np, json

    # ---- Load B ----
    B = read_ply_xyzcso(B_path)
    print(
        f"[stat] B: {B['xyz'].shape[0]} points | "
        f"f_rest={B['f_rest'].shape[1]} | {B_path}"
    )

    # ---- (1) AABB shrink + feather + optional cull ----
    if aabb_json is not None:
        with open(aabb_json, "r", encoding="utf-8") as f:
            box = json.load(f)
        aabb_min, aabb_max = parse_aabb_any(box)
        print(
            f"[AABB] min={aabb_min.tolist()} max={aabb_max.tolist()} "
            f"shrink_m={shrink_m} feather_m={feather_m}"
        )

        # feather opacity inside AABB
        if feather_m > 0.0 or shrink_m > 0.0:
            feather_B_opacity_in_aabb(
                B,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                shrink_m=shrink_m,
                feather_m=feather_m,
            )

        # optionally cull points outside the AABB
        if cull_outside:
            cull_B_outside(
                B,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                shrink_m=shrink_m,
                mode=cull_box,
            )

    # ---- (2) Multiview mask voting (masks_residual) ----
    if masks_dir is not None:
        filter_B_by_multiview_masks(
            B,
            cams,
            masks_dir,
            mask_ext=mask_ext,
            mask_dilate_px=mask_dilate_px,
            min_views=min_views,
            subsample_cams=subsample_cams,
        )
    else:
        print("[mask vote] WARNING: masks_dir is None, skip voting")

    # ---- (3) Feature alignment between A and B (f_rest) ----
    ar, br = A["f_rest"].shape[1], B["f_rest"].shape[1]
    print(f"[align] A.f_rest={ar}, B.f_rest={br}, mode={feature_align}")

    if ar == br:
        # No alignment needed
        A_use = A
    else:
        if feature_align == "pad":
            if ar < br:
                # Pad A to match B
                pad = np.zeros((A["xyz"].shape[0], br - ar), dtype=np.float32)
                A_use = A.copy()
                A_use["f_rest"] = np.concatenate([A["f_rest"], pad], axis=1)
                print(
                    f"[align] pad A.f_rest from {ar} -> {A_use['f_rest'].shape[1]}"
                )
            else:
                # Pad B to match A
                pad = np.zeros((B["xyz"].shape[0], ar - br), dtype=np.float32)
                B["f_rest"] = np.concatenate([B["f_rest"], pad], axis=1)
                A_use = A
                print(
                    f"[align] pad B.f_rest from {br} -> {B['f_rest'].shape[1]}"
                )
        elif feature_align == "trunc":
            m = min(ar, br)
            A_use = A.copy()
            A_use["f_rest"] = A["f_rest"][:, :m]
            B["f_rest"] = B["f_rest"][:, :m]
            print(f"[align] truncate f_rest to {m} (A={ar}, B={br})")
        else:
            die(f"Unknown feature_align mode: {feature_align}")

    # ---- (4) Optional: write filtered-B-only PLY ----
    if filtered_b_ply is not None:
        write_ply_xyzcso(filtered_b_ply, B)
        print(
            f"[filtered B] wrote: {filtered_b_ply} | "
            f"N={B['xyz'].shape[0]}"
        )

    # ---- (5) Concatenate A + B and write merged PLY (optional) ----
    if write_merged:
        OUT = {
            k: np.concatenate([A_use[k], B[k]], axis=0)
            for k in ["xyz", "opacity", "scale", "rot", "f_dc", "f_rest"]
        }
        write_ply_xyzcso(out_ply, OUT)
        print(
            f"[merge] wrote: {out_ply} | total N={OUT['xyz'].shape[0]} "
            f"(A={A_use['xyz'].shape[0]} + B={B['xyz'].shape[0]})"
        )
    else:
        print(
            f"[merge] skip writing merged PLY (write_merged=False). "
            f"Filtered B points: N={B['xyz'].shape[0]}"
        )


def parse_frames_arg(frames_arg: str):
    """
    Parse frame selection string.

    Supports:
      - "all"          : use all frames discovered under b_model_root
      - "1-5"          : range (inclusive)
      - "1,3,5"        : explicit list
      - "3"            : single frame
      - "1,3,5-8"      : mix of ranges and singles (e.g. 1,2,6,20-100)
      - Also accepts Chinese commas "，" in place of ",".

    Returns:
      []  -> means "all"
      [1,3,5] etc. for explicit selection
    """
    if not frames_arg:
        return []

    s = frames_arg.strip()
    s = s.replace("，", ",")  # allow Chinese comma

    if s.lower() == "all":
        return []

    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a)
            b = int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))

    # unique + sorted
    return sorted(dict.fromkeys(out))

# ---------- main ----------
def main():
    ap = ArgumentParser(
        "Batch merge A with all B frames (max-iter PLY per frame), "
        "shrink+feather/cull, and multiview mask voting; auto-make residual masks if missing"
    )
    ap.add_argument("--a_ply", required=True, help="Path to static A PLY")
    ap.add_argument("--b_model_root", required=True, help="Root containing model_frame_n for B (trained outputs)")
    ap.add_argument("--b_dataset_root", required=True, help="Dataset root containing frame_n with sparse/0 and images")
    ap.add_argument("--out_root", required=True, help="Output root; writes out_root/<prefix>n/point_cloud_merged/point_cloud.ply")
    ap.add_argument("--prefix", type=str, default="model_frame_", help="Subfolder prefix under out_root (e.g., 'model_frame_' or 'frame_').")
    # A-only renders (choose one):
    ap.add_argument("--a_images_root", type=str, default=None, help="Root containing frame_n/images of A-only renders (per-frame)")
    ap.add_argument("--a_images_single", type=str, default=None, help="Single images folder of A-only renders reused for all frames")
    # make_residual_masks options
    ap.add_argument("--gt_ext", type=str, default=".png")
    ap.add_argument("--a_ext", type=str, default=".png")
    ap.add_argument("--thr", type=float, default=25.0)
    ap.add_argument("--blur_px", type=int, default=0)
    ap.add_argument("--open_px", type=int, default=0)
    ap.add_argument("--close_px", type=int, default=1)
    ap.add_argument("--dilate_px", type=int, default=5)
    # AABB & cull & align
    ap.add_argument("--aabb_json", type=str, required=True)
    ap.add_argument("--shrink_m", type=float, default=0.0)
    ap.add_argument("--feather_m", type=float, default=0.0)
    ap.add_argument("--cull_outside", action="store_true")
    ap.add_argument("--cull_box", choices=["orig","shrunken"], default="orig")
    ap.add_argument("--feature_align", choices=["pad","trunc"], default="pad")
    # voting controls
    ap.add_argument("--mask_ext", type=str, default=".png")
    ap.add_argument("--mask_dilate_px", type=int, default=0)
    ap.add_argument("--min_views", type=int, default=25)
    ap.add_argument("--subsample_cams", type=int, default=0)
    ap.add_argument(
        "--merge_mode",
        type=str,
        choices=["merged", "filtered_only", "both"],
        default="merged",
        help=(
            "What to write out:\n"
            "  'merged'        : write only merged A+B PLY under --out_root (default)\n"
            "  'filtered_only' : write only filtered B-only PLYs under --filtered_b_root\n"
            "  'both'          : write both merged A+B and filtered B-only PLYs"
        ),
    )

    ap.add_argument(
        "--filtered_b_root",
        type=str,
        default=None,
        help=(
            "If provided, also write a filtered B-only PLY per frame under this root: "
            "<filtered_b_root>/<prefix>n/point_cloud/iteration_0/point_cloud.ply"
        ),
    )
    ap.add_argument(
        "--frames",
        type=str,
        default="all",
        help=(
            'Frame selection: "all" | "1-5" | "1,3,5" | "3" | "1,3,5-8". '
            'You can mix, e.g. "1,2,6,20-100". Chinese comma "，" is also supported.'
        ),
    )
    args = ap.parse_args()
    print("[args]", vars(args))
    if args.merge_mode in ("filtered_only", "both") and not args.filtered_b_root:
        die("--merge_mode filtered_only/both requires --filtered_b_root")

    write_merged = args.merge_mode in ("merged", "both")
    write_filtered = args.merge_mode in ("filtered_only", "both")
    if (not args.a_images_root) and (not args.a_images_single):
        print("[warn] no A-only images provided; will assume masks_residual already exist for every frame.")

    # Load A once
    A = read_ply_xyzcso(args.a_ply)
    print(f"[stat] A: N={A['xyz'].shape[0]}, f_rest={A['f_rest'].shape[1]} | {args.a_ply}")

    # Enumerate frames from MODEL root
    all_frames = list_model_frames(args.b_model_root)
    if not all_frames:
        die(f"No model_frame_n found under: {args.b_model_root}")

    # Apply --frames selection
    requested = parse_frames_arg(args.frames)
    if requested:
        frame_map = {n: path for n, path in all_frames}
        frames = []
        missing = []
        for n in requested:
            if n in frame_map:
                frames.append((n, frame_map[n]))
            else:
                missing.append(n)
        if missing:
            print(f"[warn] requested frames not found under b_model_root: {missing}")
        if not frames:
            die(f"No valid frames to merge after applying --frames={args.frames}")
    else:
        # "all" → use everything
        frames = all_frames

    print(f"[frames] will merge frames: {[n for n, _ in frames]}")

    total_frames_merged = 0
    total_mask_sec = 0.0
    total_merge_sec = 0.0
    total_frame_sec = 0.0

    t_global_start = time.perf_counter()

    for n, model_frame_dir in frames:
        ply_path, itN = find_latest_B_ply(model_frame_dir)
        if not ply_path:
            print(f"[skip] no B ply found in {os.path.relpath(model_frame_dir, args.b_model_root)}")
            continue

        t_frame_start = time.perf_counter()

        # Dataset-side paths
        dataset_frame_dir = os.path.join(args.b_dataset_root, f"frame_{n}")
        colmap_dir = os.path.join(dataset_frame_dir, "sparse", "0")
        if not os.path.isdir(colmap_dir):
            die(f"Missing COLMAP dir for frame {n}: {colmap_dir}")

        # masks_residual (use if exists; else auto-make if A images provided)
        t_mask_start = time.perf_counter()
        masks_dir = os.path.join(dataset_frame_dir, "masks_residual")
        if dir_empty_or_missing(masks_dir):
            masks_dir = ensure_masks_residual(
                n, args.b_dataset_root,
                a_images_root=args.a_images_root,
                a_images_single=args.a_images_single,
                gt_ext=args.gt_ext, a_ext=args.a_ext,
                thr=args.thr, blur_px=args.blur_px,
                open_px=args.open_px, close_px=args.close_px, dilate_px=args.dilate_px
            )
        else:
            # even when masks already exist, we include the (tiny) check time
            print(f"[auto-masks] frame {n}: masks_residual already present -> {masks_dir}")
        t_mask_end = time.perf_counter()

        t_merge_start = time.perf_counter()
        cams = load_colmap_simple(colmap_dir)

        print(f"\n[frame {n}] latest iteration = {itN} | ply = {ply_path}")
        print(f"[frame {n}] colmap_dir = {colmap_dir}")
        print(f"[frame {n}] masks_dir  = {masks_dir}")

        # merged A+B output
        out_ply = None
        out_dir = None
        if write_merged:
            out_dir = os.path.join(args.out_root, f"{args.prefix}{n}")
            out_pc_dir = os.path.join(out_dir, "point_cloud_merged")
            os.makedirs(out_pc_dir, exist_ok=True)
            out_ply = os.path.join(out_pc_dir, "point_cloud.ply")

        # filtered-B-only output
        filtered_b_ply = None
        if write_filtered:
            fb_dir = os.path.join(
                args.filtered_b_root,
                f"{args.prefix}{n}",
                "point_cloud",
                "iteration_0"
            )
            os.makedirs(fb_dir, exist_ok=True)
            filtered_b_ply = os.path.join(fb_dir, "point_cloud.ply")

        merge_one_frame(
            A, ply_path, out_ply, cams, masks_dir,
            aabb_json=args.aabb_json, shrink_m=args.shrink_m, feather_m=args.feather_m,
            cull_outside=args.cull_outside, cull_box=args.cull_box,
            feature_align=args.feature_align,
            mask_ext=args.mask_ext, mask_dilate_px=args.mask_dilate_px,
            min_views=args.min_views, subsample_cams=args.subsample_cams,
            filtered_b_ply=filtered_b_ply,
            write_merged=write_merged,
        )

        # propagate test list & cfg so ta_test can reuse the split
        try:
            import shutil

            # model_frame_n root under filtered_b_root
            filtered_frame_root = None
            if write_filtered and args.filtered_b_root:
                filtered_frame_root = os.path.join(
                    args.filtered_b_root, f"{args.prefix}{n}"
                )
                os.makedirs(filtered_frame_root, exist_ok=True)

            for name in ("test_images.txt", "cfg_args"):
                src = os.path.join(model_frame_dir, name)
                if not os.path.isfile(src):
                    continue

                # 1) Copy to merged A+B output (if it exists)
                if write_merged and out_dir is not None:
                    dst = os.path.join(out_dir, name)
                    shutil.copy2(src, dst)
                    print(f"[meta] copied {name} → {dst}")

                # 2) Copy to filtered B-only model root
                if filtered_frame_root is not None:
                    dst_fb = os.path.join(filtered_frame_root, name)
                    shutil.copy2(src, dst_fb)
                    print(f"[meta] copied {name} → {dst_fb}")

        except Exception as e:
            print(f"[meta][warn] failed to propagate test meta: {e}")


        t_merge_end = time.perf_counter()
        t_frame_end = t_merge_end

        mask_sec = t_mask_end - t_mask_start
        merge_sec = t_merge_end - t_merge_start
        frame_sec = t_frame_end - t_frame_start

        total_frames_merged += 1
        total_mask_sec += mask_sec
        total_merge_sec += merge_sec
        total_frame_sec += frame_sec

        print(
            f"[time][frame {n}] masks_residual = {mask_sec:.3f} s | "
            f"merge+colmap = {merge_sec:.3f} s | frame total = {frame_sec:.3f} s"
        )

    t_global_end = time.perf_counter()
    global_sec = t_global_end - t_global_start

    if total_frames_merged > 0:
        avg_frame_sec = total_frame_sec / total_frames_merged
    else:
        avg_frame_sec = 0.0

    print("\n[time][summary]")
    print(f"  frames merged      : {total_frames_merged}")
    print(f"  masks_residual sum : {total_mask_sec:.3f} s")
    print(f"  merge+colmap sum   : {total_merge_sec:.3f} s")
    print(f"  frame total sum    : {total_frame_sec:.3f} s")
    print(f"  avg frame time     : {avg_frame_sec:.3f} s")
    print(f"  end-to-end (loop)  : {global_sec:.3f} s")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print("[exception]", repr(e))
        traceback.print_exc()
        sys.exit(2)
