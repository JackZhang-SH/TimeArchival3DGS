import os
import json
import numpy as np
import shutil
import math
import argparse
from PIL import Image

def rotmat2qvec(R):
    t = np.trace(R)
    if t > 0:
        r = np.sqrt(1 + t)
        s = 0.5 / r
        return np.array([0.5 * r, (R[2,1] - R[1,2]) * s, (R[0,2] - R[2,0]) * s, (R[1,0] - R[0,1]) * s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        r = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        s = 0.5 / r
        return np.array([(R[2,1] - R[1,2]) * s, 0.5 * r, (R[1,0] + R[0,1]) * s, (R[0,2] + R[2,0]) * s])
    elif R[1,1] > R[2,2]:
        r = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
        s = 0.5 / r
        return np.array([(R[0,2] - R[2,0]) * s, (R[1,0] + R[0,1]) * s, 0.5 * r, (R[2,1] + R[1,2]) * s])
    else:
        r = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
        s = 0.5 / r
        return np.array([(R[1,0] - R[0,1]) * s, (R[0,2] + R[2,0]) * s, (R[2,1] + R[1,2]) * s, 0.5 * r])

def process_frame(in_frame_dir, out_frame_dir, frame_name):
    print(f"Processing {frame_name} ...")
    
    images_dir = os.path.join(out_frame_dir, "images")
    sparse_dir = os.path.join(out_frame_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    cameras_dict = {}
    images_colmap = []
    image_id_counter = 1
    camera_id_counter = 1
    
    cam_centers = [] 
    
    json_tasks = [
        ("transforms.json", "train_"),
        ("transforms_test.json", "test_")
    ]
    
    for json_name, prefix in json_tasks:
        json_path = os.path.join(in_frame_dir, json_name)
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 提取全局参数（如果存在的话）
        global_params = data if isinstance(data, dict) else {}
        frames = data.get("frames", data) if isinstance(data, dict) else data
        
        for item in frames:
            # 兼容带有 Windows 双反斜杠的路径 "train\\render_1.png"
            raw_path = item["file_path"].replace("\\", "/").replace("./", "")
            orig_img_path = os.path.normpath(os.path.join(in_frame_dir, raw_path))
            
            if not os.path.exists(orig_img_path):
                print(f"  Warning: Image {orig_img_path} not found, skipping.")
                continue
                
            img_basename = os.path.basename(orig_img_path)
            new_img_name = f"{prefix}{img_basename}"
            new_img_path = os.path.join(images_dir, new_img_name)
            shutil.copy2(orig_img_path, new_img_path)
            
            # 优先从单个 frame 提取，如果找不到则去全局字典 (global_params) 提取
            w = item.get("w", global_params.get("w", None))
            h = item.get("h", global_params.get("h", None))
            
            if w is None or h is None:
                with Image.open(new_img_path) as img:
                    w, h = img.size
                    
            fl_x = item.get("fl_x", global_params.get("fl_x", None))
            fl_y = item.get("fl_y", global_params.get("fl_y", None))
            
            if fl_x is not None and fl_y is not None:
                fx, fy = float(fl_x), float(fl_y)
            else:
                camera_angle_x = item.get("camera_angle_x", global_params.get("camera_angle_x", None))
                fx = w / (2.0 * math.tan(float(camera_angle_x) / 2.0))
                fy = fx
                
            cx = item.get("cx", global_params.get("cx", w / 2.0))
            cy = item.get("cy", global_params.get("cy", h / 2.0))
            
            params = f"{w} {h} {fx} {fy} {cx} {cy}"
                
            if params not in cameras_dict:
                cameras_dict[params] = camera_id_counter
                camera_id_counter += 1
            cam_id = cameras_dict[params]
            
            c2w = np.array(item["transform_matrix"])
            cam_centers.append(c2w[:3, 3]) 
            
            c2w[:, 1:3] *= -1 # OpenCV 坐标系转换
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            qvec = rotmat2qvec(R)
            
            images_colmap.append({
                "id": image_id_counter,
                "qvec": qvec,
                "tvec": T,
                "camera_id": cam_id,
                "name": new_img_name
            })
            image_id_counter += 1

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for params, c_id in cameras_dict.items():
            f.write(f"{c_id} PINHOLE {params}\n")
            
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in images_colmap:
            q = img["qvec"]
            t = img["tvec"]
            f.write(f"{img['id']} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img['camera_id']} {img['name']}\n")
            f.write("\n")
            
    # 计算所有相机中心点包围盒，用于随机点云自适应缩放
    cam_centers = np.array(cam_centers)
    center_mean = np.mean(cam_centers, axis=0)
    avg_dist = np.mean(np.linalg.norm(cam_centers - center_mean, axis=1))
    radius = avg_dist * 0.5 
    if radius < 1.0:
        radius = 1.0

    num_points = 10000 
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(1, num_points + 1):
            pt = np.random.uniform(-radius, radius, 3) + center_mean
            c = np.random.randint(0, 255, 3)
            f.write(f"{i} {pt[0]} {pt[1]} {pt[2]} {c[0]} {c[1]} {c[2]} 0.0 \n")

    print(f"  -> Success! Generated {len(images_colmap)} images ({len(cameras_dict)} cameras) in {out_frame_dir}")
    print(f"  -> Scene radius estimated at {radius:.2f}, spawned 10000 points.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    frames_found = False
    for d in os.listdir(args.input_dir):
        in_frame_dir = os.path.join(args.input_dir, d)
        if os.path.isdir(in_frame_dir) and d.startswith("frame_"):
            frames_found = True
            out_frame_dir = os.path.join(args.output_dir, d)
            process_frame(in_frame_dir, out_frame_dir, d)
            
    if frames_found:
        print("\nAll frames converted successfully!")