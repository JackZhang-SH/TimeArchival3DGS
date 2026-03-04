import os
import json
import numpy as np
import shutil
import math
import argparse
from PIL import Image

# =============================================================================
# 核心修复：复刻 RS Studio (core.py) 的矩阵变换逻辑 (Blender Z-up -> 3DGS)
# =============================================================================
def get_colmap_from_blender_matrix(c2w_blender_list):
    M_bl = np.array(c2w_blender_list)
    
    # 1. 定义局部坐标系修正矩阵 T
    # Blender Local: Right(X), Up(Y), Back(Z) -> View is -Z
    # COLMAP Local: Right(X), Down(Y), Fwd(Z) -> View is +Z
    # 变换: X->X, Y->-Y, Z->-Z
    T = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    
    # 2. 计算 w2c
    try:
        M_bl_inv = np.linalg.inv(M_bl)
    except np.linalg.LinAlgError:
        return None, None

    # 保持世界坐标系不变 (Z-up), 只修正相机朝向
    M_w2cv = T @ M_bl_inv
    
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
    return np.array([w, x, y, z])

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
    
    # 强制前缀，区分 Train/Test
    json_tasks = [
        ("transforms.json", "train_"),
        ("transforms_train.json", "train_"),
        ("transforms_test.json", "test_"),
        ("transforms_val.json", "test_")
    ]
    
    all_frames = []
    
    # 1. 读取 JSON
    for jf_name, prefix in json_tasks:
        p = os.path.join(in_frame_dir, jf_name)
        if not os.path.exists(p): continue
        with open(p, 'r') as f:
            try: data = json.load(f)
            except: continue
            
            if global_params["w"] is None:
                global_params["w"] = data.get("w")
                global_params["h"] = data.get("h")
                global_params["fl_x"] = data.get("fl_x")
                global_params["fl_y"] = data.get("fl_y")
                global_params["cx"] = data.get("cx")
                global_params["cy"] = data.get("cy")
                if global_params["fl_x"] is None and "camera_angle_x" in data:
                     global_params["fl_x"] = float(global_params["w"]) / (2 * math.tan(data["camera_angle_x"] / 2))
                     global_params["fl_y"] = global_params["fl_x"]

            for frame in data.get("frames", []):
                frame['_output_prefix'] = prefix
                all_frames.append(frame)

    if not all_frames:
        print(f"  [Error] No frames found in {in_frame_dir}")
        return

    all_frames.sort(key=lambda x: x["file_path"])

    camera_id_counter = 1
    image_id_counter = 1
    
    for i, frame in enumerate(all_frames):
        rel_path = frame["file_path"].replace("\\", "/")
        if rel_path.startswith("./"): rel_path = rel_path[2:]
        src_path = os.path.join(in_frame_dir, rel_path)
        
        # 路径容错
        if not os.path.exists(src_path):
            basename = os.path.basename(rel_path)
            if os.path.exists(os.path.join(in_frame_dir, basename)): src_path = os.path.join(in_frame_dir, basename)
            elif os.path.exists(os.path.join(in_frame_dir, "images", basename)): src_path = os.path.join(in_frame_dir, "images", basename)
            else: continue

        # 命名处理
        original_basename = os.path.basename(rel_path)
        prefix = frame['_output_prefix']
        final_img_name = original_basename if original_basename.startswith(prefix) else f"{prefix}{original_basename}"
        
        dst_path = os.path.join(images_dir, final_img_name)
        
        # === [新增功能] 强制处理透明背景 -> 白底 ===
        # 如果目标文件不存在，才进行处理
        if not os.path.exists(dst_path):
            with Image.open(src_path) as img:
                # 检查是否有透明通道
                if img.mode == 'RGBA':
                    # 创建纯白背景 (255, 255, 255)
                    # 如果想要黑底，改成 (0, 0, 0)
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    # 将图片粘贴到白底上，使用 Alpha 通道作为 Mask
                    background.paste(img, mask=img.split()[3])
                    background.save(dst_path, quality=100)
                else:
                    # 如果没有透明通道，直接复制或保存
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dst_path, quality=100)
        
        # 读取图片尺寸
        w = int(frame.get("w", global_params["w"] or 0))
        h = int(frame.get("h", global_params["h"] or 0))
        if w == 0:
            with Image.open(src_path) as pil_img: w, h = pil_img.size
            
        fl_x = frame.get("fl_x", global_params["fl_x"] or w)
        fl_y = frame.get("fl_y", global_params["fl_y"] or w)
        cx = frame.get("cx", w/2.0)
        cy = frame.get("cy", h/2.0)
        
        model_name = "PINHOLE"
        params_list = [fl_x, fl_y, cx, cy]
        params_str = " ".join(map(str, params_list))
        cam_key = (model_name, w, h, params_str)
        
        if cam_key not in cameras:
            cameras[cam_key] = camera_id_counter
            camera_id_counter += 1
        cam_id = cameras[cam_key]

        # 核心坐标转换
        qvec, tvec = get_colmap_from_blender_matrix(frame["transform_matrix"])
        if qvec is None: continue

        images.append({
            "id": image_id_counter,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": final_img_name
        })
        image_id_counter += 1

    # 写入 COLMAP 格式文件
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

    # 生成 Dummy points3D.txt (会被 run 脚本处理，但保留一个防止 COLMAP 报错)
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("1 0 0 0 128 128 128 0.0\n")

    print(f"  -> Done. {len(images)} images processed (White Background applied).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir): exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    
    found = False
    for d in sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]):
        if "frame_" in d:
            found = True
            process_frame(os.path.join(args.input_dir, d), os.path.join(args.output_dir, d), d)
    if not found:
        if os.path.exists(os.path.join(args.input_dir, "transforms.json")):
             process_frame(args.input_dir, args.output_dir, os.path.basename(args.input_dir))