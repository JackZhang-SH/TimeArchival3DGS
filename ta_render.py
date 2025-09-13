#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Archival 3DGS — Fixed-Camera Sequencer (Prefetch + Warmup)
- 预取下一帧 3DGS 到 GPU，和当前帧渲染/保存并行
- 首次调用前预热 render kernels
- 可选 JPEG 输出提速

用法举例：
python ta_render.py -m ./output_seq -o ./renders -c ./camera.json --frames all \
    --preload_depth 2 --warmup --save_format jpeg --jpeg_quality 90
"""
import argparse, json, math, os, sys, time, threading, queue
from pathlib import Path

import torch
import torchvision
import numpy as np
from PIL import Image

from gaussian_renderer import render  # noqa
from arguments import PipelineParams  # noqa
from scene.cameras import MiniCam  # noqa
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov  # noqa
from scene.gaussian_model import GaussianModel  # noqa
from scene.colmap_loader import (  # noqa
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text,
    qvec2rotmat
)

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def fov_from_fx(fx, w):
    return 2.0 * math.atan(w / (2.0 * fx))


def load_cam_from_json(cam_json_path: Path):
    with open(cam_json_path, "r") as f:
        C = json.load(f)
    W, H = int(C["width"]), int(C["height"])
    znear = float(C.get("znear", 0.01))
    zfar  = float(C.get("zfar", 100.0))

    if all(k in C for k in ["fx","fy","cx","cy"]):
        fx, fy = float(C["fx"]), float(C["fy"])
        fovx = fov_from_fx(fx, W); fovy = fov_from_fx(fy, H)
    else:
        fovx = float(C["FoVx"]); fovy = float(C["FoVy"])

    R = np.asarray(C["R"], dtype=np.float32); T = np.asarray(C["T"], dtype=np.float32)
    world_view = torch.tensor(getWorld2View2(R, T, np.array([0,0,0],dtype=np.float32), 1.0)).transpose(0,1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    view = MiniCam(W, H, fovy, fovx, znear, zfar, world_view, full)
    setattr(view, "image_name", "custom")
    return view


def _try_read_colmap_extrinsics(path_bin, path_txt):
    try: return read_extrinsics_binary(str(path_bin)), True
    except Exception: return read_extrinsics_text(str(path_txt)), False


def _try_read_colmap_intrinsics(path_bin, path_txt):
    try: return read_intrinsics_binary(str(path_bin)), True
    except Exception: return read_intrinsics_text(str(path_txt)), False


def load_cam_from_colmap(dataset_root: Path, image_name: str, sparse_id: int = 0, znear: float = 0.01, zfar: float = 100.0):
    sp = dataset_root / "sparse" / str(sparse_id)
    if not sp.exists():
        raise FileNotFoundError(f"COLMAP sparse folder not found: {sp}")

    images_bin  = sp / "images.bin";  images_txt  = sp / "images.txt"
    cameras_bin = sp / "cameras.bin"; cameras_txt = sp / "cameras.txt"
    extr_map, _ = _try_read_colmap_extrinsics(images_bin, images_txt)
    intr_map,  _ = _try_read_colmap_intrinsics(cameras_bin, cameras_txt)

    target_key = None
    for k, extr in extr_map.items():
        if extr.name == image_name: target_key=k; break
    if target_key is None:
        bn = os.path.basename(image_name)
        for k, extr in extr_map.items():
            if os.path.basename(extr.name) == bn: target_key=k; break
    if target_key is None:
        raise KeyError(f"Image named '{image_name}' not found in COLMAP images.")

    extr = extr_map[target_key]; intr = intr_map[extr.camera_id]
    width, height = int(intr.width), int(intr.height)
    model = intr.model
    if model == "SIMPLE_PINHOLE": fx = float(intr.params[0]); fy = fx
    elif model == "PINHOLE":      fx = float(intr.params[0]); fy = float(intr.params[1])
    else: raise AssertionError(f"Unsupported COLMAP camera model: {model}. Use PINHOLE/SIMPLE_PINHOLE.")

    fovx = focal2fov(fx, width); fovy = focal2fov(fy, height)
    R = np.transpose(qvec2rotmat(extr.qvec)).astype(np.float32)
    T = np.asarray(extr.tvec, dtype=np.float32)

    world_view = torch.tensor(getWorld2View2(R, T, np.array([0,0,0],dtype=np.float32), 1.0)).transpose(0,1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    view = MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full)
    setattr(view, "image_name", extr.name)
    return view


def find_latest_iteration(model_path: Path) -> int:
    p = model_path / "point_cloud"
    if not p.exists(): return -1
    iters = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try: iters.append(int(d.name.split("_")[1]))
            except: pass
    return max(iters) if iters else -1


def parse_frames(frames_str, models_root: Path):
    if frames_str == "all":
        dirs = [p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("model_frame_")]
        return sorted(int(p.name.split("_")[-1]) for p in dirs)
    if "-" in frames_str:
        a, b = frames_str.split("-"); return list(range(int(a), int(b)+1))
    if "," in frames_str: return [int(x) for x in frames_str.split(",")]
    return [int(frames_str)]


# --------------------- 预取线程 ---------------------
def prefetch_worker(frames, models_root, prefix, iteration, sh_degree, out_queue: queue.Queue, stop_flag):
    """
    依次把 (frame_idx, iter, GaussianModel, load_secs) 放到 out_queue
    """
    try:
        for i in frames:
            if stop_flag["stop"]: break
            mp = models_root / f"{prefix}{i}"
            it = iteration if iteration >= 0 else find_latest_iteration(mp)
            if it < 0:
                out_queue.put(("skip", i, "no_iter", 0.0))
                continue
            ply = mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
            if not ply.exists():
                out_queue.put(("skip", i, "no_ply", 0.0))
                continue
            t0 = time.perf_counter()
            gauss = GaussianModel(sh_degree)
            gauss.load_ply(str(ply), use_train_test_exp=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            out_queue.put(("ok", (i, it, gauss, t1 - t0)))
    except Exception as e:
        out_queue.put(("err", str(e)))
    finally:
        out_queue.put(("eof", None))


def build_warmup_view_from(view: MiniCam, scale=0.25) -> MiniCam:
    W = max(32, int(view.image_width * scale))
    H = max(32, int(view.image_height * scale))
    world_view = view.world_view_transform
    proj = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(W, H, view.FoVy, view.FoVx, view.znear, view.zfar, world_view, full)


def main(argv):
    parser = argparse.ArgumentParser(description="Time Archival 3DGS — Sequencer (Prefetch)")
    parser.add_argument("-m","--models_root", type=str, required=True)
    parser.add_argument("-o","--output_dir", type=str, required=True)
    # Camera
    parser.add_argument("-c","--camera_json", type=str, default=None)
    parser.add_argument("--colmap_path", type=str, default=None)
    parser.add_argument("--image_name", type=str, default=None)
    parser.add_argument("--sparse_id", type=int, default=0)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    # Sequence
    parser.add_argument("--frames", type=str, default="all")
    parser.add_argument("--prefix", type=str, default="model_frame_")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--video_name", type=str, default="time_archival.mp4")
    # 新增：预取/预热/保存格式
    parser.add_argument("--preload_depth", type=int, default=2, help="后台队列最大深度（推荐 1–3）")
    parser.add_argument("--warmup", action="store_true", help="启动后用小分辨率相机预热一次 render kernels")
    parser.add_argument("--save_format", choices=["png","jpeg"], default="png")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    pp = PipelineParams(parser)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models_root = Path(args.models_root)
    frames = parse_frames(args.frames, models_root)
    print(f"[TA-Render] Frames to render: {frames}")

    # Camera
    if args.colmap_path and args.image_name:
        print(f"[TA-Render] Using COLMAP camera from {args.colmap_path}, image '{args.image_name}', sparse_id={args.sparse_id}")
        view = load_cam_from_colmap(Path(args.colmap_path), args.image_name, args.sparse_id, args.znear, args.zfar)
    elif args.camera_json:
        print(f"[TA-Render] Using camera JSON: {args.camera_json}")
        view = load_cam_from_json(Path(args.camera_json))
    else:
        raise ValueError("Please specify either --camera_json or (--colmap_path AND --image_name).")

    bg_color = [1,1,1] if args.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)

    # ---------- 启动预取线程 ----------
    q_out = queue.Queue(maxsize=max(1, args.preload_depth))
    stop_flag = {"stop": False}
    worker = threading.Thread(target=prefetch_worker, args=(frames, models_root, args.prefix, args.iteration, args.sh_degree, q_out, stop_flag), daemon=True)
    worker.start()

    # ---------- 可选：预热 ----------
    # 等第一帧模型到位后，用小分辨率相机预渲染一次，编译 CUDA kernels
    got_first = False

    frame_paths = []
    total_start = time.perf_counter()
    sum_load = sum_render = sum_save = 0.0
    n_done = 0

    while True:
        tag, payload = q_out.get()
        if tag == "eof": break
        if tag == "skip":
            _, i, reason, _ = payload
            print(f"[TA-Render] SKIP frame {i}: {reason}")
            continue
        if tag == "err":
            print(f"[TA-Render][ERROR] prefetch: {payload}")
            continue
        if tag != "ok": continue

        i, it, gauss, load_secs = payload
        if not got_first and args.warmup:
            got_first = True
            try:
                warm_view = build_warmup_view_from(view, scale=0.25)
                with torch.no_grad():
                    _ = render(warm_view, gauss, pipe, background, use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                if torch.cuda.is_available(): torch.cuda.synchronize()
                print("[TA-Render] Warmup done.")
            except Exception as e:
                print(f"[TA-Render] Warmup failed: {e}")

        # ---- 正式渲染 ----
        t1 = time.perf_counter()
        with torch.no_grad():
            out = render(view, gauss, pipe, background, use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.perf_counter()

        # ---- 保存 ----
        png_path = out_dir / f"{i:05d}.png"
        if args.save_format == "png":
            torchvision.utils.save_image(out, str(png_path))
        else:
            # JPEG：快得多
            img = (torch.clamp(out, 0, 1) * 255).byte().permute(1,2,0).contiguous().cpu().numpy()
            Image.fromarray(img).save(str(out_dir / f"{i:05d}.jpg"), format="JPEG", quality=args.jpeg_quality, optimize=True)
        t3 = time.perf_counter()

        # 统计（load 发生在后台，这里展示后台耗时；主线程 load=0）
        load_dt = load_secs
        render_dt = t2 - t1
        save_dt = t3 - t2
        sum_load += load_dt; sum_render += render_dt; sum_save += save_dt; n_done += 1

        frame_paths.append(str(png_path if args.save_format=="png" else out_dir / f"{i:05d}.jpg"))
        print(f"[TA-Render] Saved {out_dir / (f'{i:05d}.png' if args.save_format=='png' else f'{i:05d}.jpg')}")
        print(f"[TA-Render][Timing] frame {i}: load(bg)={load_dt:.3f}s, render={render_dt:.3f}s, save={save_dt:.3f}s, total~={(render_dt+save_dt):.3f}s")

        # 主动释放上一帧；（保留由 CUDA allocator 管理的缓存）
        del gauss
        torch.cuda.empty_cache()

    total_end = time.perf_counter()
    if n_done > 0:
        print(f"[TA-Render][Timing][Summary] frames={n_done} | "
              f"prefetch_load_sum={sum_load:.3f}s, render_sum={sum_render:.3f}s, save_sum={sum_save:.3f}s, "
              f"end2end_no_video={(total_end-total_start):.3f}s | "
              f"avg_render={sum_render/n_done:.3f}s | render_FPS={n_done/sum_render:.2f}")

    # 可选拼视频
    try:
        if frame_paths and args.save_format=="png":
            tvid0 = time.perf_counter()
            import imageio.v3 as iio
            imgs = [iio.imread(p) for p in frame_paths]
            iio.imwrite(str(out_dir / args.video_name), imgs, fps=24)
            tvid1 = time.perf_counter()
            print(f"[TA-Render] Video written to {out_dir/args.video_name}")
            print(f"[TA-Render][Timing] video_build={tvid1 - tvid0:.3f}s")
        elif frame_paths and args.save_format=="jpeg":
            print("[TA-Render] You saved JPEGs; use ffmpeg to build video, e.g.:")
            print("            ffmpeg -y -framerate 24 -i %05d.jpg -pix_fmt yuv420p time_archival.mp4")
    except Exception as e:
        print(f"[TA-Render] Could not build video: {e}")

if __name__ == "__main__":
    main(sys.argv[1:])
