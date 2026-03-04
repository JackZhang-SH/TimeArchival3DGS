import os
import json
import numpy as np
import math
import argparse
from PIL import Image

def get_colmap_from_blender_matrix(c2w_blender_list):
    M_bl = np.array(c2w_blender_list, dtype=np.float64)

    # OpenCV-world -> Blender-world (A)  (与 core.py 完全一致)
    A = np.array([
        [1, 0,  0, 0],
        [0, 0,  1, 0],
        [0,-1,  0, 0],
        [0, 0,  0, 1],
    ], dtype=np.float64)

    # Blender-local -> OpenCV-local (T) (与 core.py 完全一致)
    T = np.array([
        [1, 0,  0, 0],
        [0,-1,  0, 0],
        [0, 0, -1, 0],
        [0, 0,  0, 1],
    ], dtype=np.float64)

    try:
        M_bl_inv = np.linalg.inv(M_bl)
    except np.linalg.LinAlgError:
        return None, None

    # ✅ 关键：要乘 A
    M_w2cv = T @ M_bl_inv @ A

    R = M_w2cv[:3, :3]
    tvec = M_w2cv[:3, 3]
    qvec = rotmat2qvec(R)
    return qvec, tvec

def rotmat2qvec(R):
    t = np.trace(R)
    if t > 0.0:
        r = np.sqrt(1.0 + t)
        s = 0.5 / r
        w = 0.5 * r
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        w = (R[2, 1] - R[1, 2]) * s
        x = 0.5 * r
        y = (R[0, 1] + R[1, 0]) * s
        z = (R[0, 2] + R[2, 0]) * s
    elif R[1, 1] > R[2, 2]:
        r = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        w = (R[0, 2] - R[2, 0]) * s
        x = (R[0, 1] + R[1, 0]) * s
        y = 0.5 * r
        z = (R[1, 2] + R[2, 1]) * s
    else:
        r = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        s = 0.5 / r
        w = (R[1, 0] - R[0, 1]) * s
        x = (R[0, 2] + R[2, 0]) * s
        y = (R[1, 2] + R[2, 1]) * s
        z = 0.5 * r
    return np.array([w, x, y, z], dtype=np.float64)

def _save_keep_alpha(img: Image.Image, dst_path: str):
    """
    Save image while preserving alpha if present.
    - If image is RGBA but dst is jpg/jpeg, we switch to PNG to preserve alpha.
    - Returns (actual_saved_path, actual_filename).
    """
    ext = os.path.splitext(dst_path)[1].lower()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if img.mode == "RGBA":
        if ext in [".jpg", ".jpeg"]:
            dst_path = os.path.splitext(dst_path)[0] + ".png"
            ext = ".png"
        # Save RGBA losslessly
        img.save(dst_path, compress_level=0)
        return dst_path, os.path.basename(dst_path)

    # Non-alpha images: keep as RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    if ext in [".jpg", ".jpeg"]:
        img.save(dst_path, quality=100, subsampling=0, optimize=True)
        return dst_path, os.path.basename(dst_path)
    elif ext == ".png":
        img.save(dst_path, compress_level=0)
        return dst_path, os.path.basename(dst_path)
    else:
        # fallback to PNG
        dst_path = os.path.splitext(dst_path)[0] + ".png"
        img.save(dst_path, compress_level=0)
        return dst_path, os.path.basename(dst_path)

def process_frame(in_frame_dir, out_frame_dir, frame_name):
    print(f"Processing {frame_name} ...")

    images_dir = os.path.join(out_frame_dir, "images")
    sparse_dir = os.path.join(out_frame_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    cameras = {}
    images = []

    global_params = {
        "w": None, "h": None, "fl_x": None, "fl_y": None,
        "cx": None, "cy": None, "k1": 0, "k2": 0, "p1": 0, "p2": 0
    }

    json_tasks = [
        ("transforms.json", "train_"),
        ("transforms_train.json", "train_"),
        ("transforms_test.json", "test_"),
        ("transforms_val.json", "test_")
    ]

    all_frames = []

    for jf_name, prefix in json_tasks:
        p = os.path.join(in_frame_dir, jf_name)
        if not os.path.exists(p):
            continue
        with open(p, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        if global_params["w"] is None:
            global_params["w"] = data.get("w")
            global_params["h"] = data.get("h")
            global_params["fl_x"] = data.get("fl_x")
            global_params["fl_y"] = data.get("fl_y")
            global_params["cx"] = data.get("cx")
            global_params["cy"] = data.get("cy")
            if global_params["fl_x"] is None and "camera_angle_x" in data and global_params["w"] is not None:
                global_params["fl_x"] = float(global_params["w"]) / (2 * math.tan(data["camera_angle_x"] / 2))
                global_params["fl_y"] = global_params["fl_x"]

        for frame in data.get("frames", []):
            frame["_output_prefix"] = prefix
            all_frames.append(frame)

    if not all_frames:
        print(f"  [Error] No frames found in {in_frame_dir}")
        return

    all_frames.sort(key=lambda x: x["file_path"])

    camera_id_counter = 1
    image_id_counter = 1

    for frame in all_frames:
        rel_path = frame["file_path"].replace("\\", "/")
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        src_path = os.path.join(in_frame_dir, rel_path)

        if not os.path.exists(src_path):
            basename = os.path.basename(rel_path)
            cand1 = os.path.join(in_frame_dir, basename)
            cand2 = os.path.join(in_frame_dir, "images", basename)
            if os.path.exists(cand1):
                src_path = cand1
            elif os.path.exists(cand2):
                src_path = cand2
            else:
                continue

        original_basename = os.path.basename(rel_path)
        prefix = frame["_output_prefix"]
        final_img_name = original_basename if original_basename.startswith(prefix) else f"{prefix}{original_basename}"
        dst_path = os.path.join(images_dir, final_img_name)

        # === 保留 alpha：RGBA 原样保存（必要时改为 PNG）===
        if not os.path.exists(dst_path):
            with Image.open(src_path) as img:
                saved_path, saved_name = _save_keep_alpha(img, dst_path)
                final_img_name = saved_name
                dst_path = saved_path
        else:
            # 目标文件已存在，但如果原本命名是 jpg 且实际应为 png（第一次生成时已修正），
            # 这里不强制再改，避免覆盖；若你想强制一致可自行清理输出目录后重跑。
            pass

        w = int(frame.get("w", global_params["w"] or 0))
        h = int(frame.get("h", global_params["h"] or 0))
        if w == 0:
            with Image.open(src_path) as pil_img:
                w, h = pil_img.size

        fl_x = frame.get("fl_x", global_params["fl_x"] or w)
        fl_y = frame.get("fl_y", global_params["fl_y"] or w)
        cx = frame.get("cx", w / 2.0)
        cy = frame.get("cy", h / 2.0)

        model_name = "PINHOLE"
        params_list = [fl_x, fl_y, cx, cy]
        params_str = " ".join(map(str, params_list))
        cam_key = (model_name, w, h, params_str)

        if cam_key not in cameras:
            cameras[cam_key] = camera_id_counter
            camera_id_counter += 1
        cam_id = cameras[cam_key]

        qvec, tvec = get_colmap_from_blender_matrix(frame["transform_matrix"])
        if qvec is None:
            continue

        images.append({
            "id": image_id_counter,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": final_img_name
        })
        image_id_counter += 1

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for (model, w, h, p_str), cid in sorted(cameras.items(), key=lambda x: x[1]):
            f.write(f"{cid} {model} {w} {h} {p_str}\n")

    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in images:
            q = img["qvec"]
            t = img["tvec"]
            f.write(f"{img['id']} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img['camera_id']} {img['name']}\n\n")

    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("1 0 0 0 128 128 128 0.0\n")

    print(f"  -> Done. {len(images)} images processed (Alpha preserved if present).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise SystemExit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    found = False
    for d in sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]):
        if "frame_" in d:
            found = True
            process_frame(os.path.join(args.input_dir, d), os.path.join(args.output_dir, d), d)

    if not found:
        if os.path.exists(os.path.join(args.input_dir, "transforms.json")):
            process_frame(args.input_dir, args.output_dir, os.path.basename(args.input_dir))