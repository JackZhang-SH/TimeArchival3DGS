# ta_server.py
import io, os, math, argparse, threading, time
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify

from gaussian_renderer import render
from arguments import PipelineParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# ---------- utils ----------
def fov_from_fx(fx, w): return 2.0 * math.atan(w / (2.0 * fx))

def latest_iteration(model_path: Path) -> int:
    pc = model_path / "point_cloud"
    if not pc.exists(): return -1
    iters = []
    for d in pc.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try: iters.append(int(d.name.split("_")[1]))
            except: pass
    return max(iters) if iters else -1

def list_frames(models_root: Path, prefix: str):
    frames = []
    for p in models_root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            try: frames.append(int(p.name.split("_")[-1]))
            except: pass
    return sorted(frames)

class MiniCam:
    def __init__(self, W, H, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = W; self.image_height = H
        self.FoVy = fovy; self.FoVx = fovx
        self.znear = znear; self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]

def build_minicam(cam_dict):
    W = int(cam_dict["width"]); H = int(cam_dict["height"])
    znear = float(cam_dict.get("znear", 0.01)); zfar = float(cam_dict.get("zfar", 100.0))

    if all(k in cam_dict for k in ("fx","fy","cx","cy")):
        fx, fy = float(cam_dict["fx"]), float(cam_dict["fy"])
        fovx = fov_from_fx(fx, W); fovy = fov_from_fx(fy, H)
    else:
        fovx = float(cam_dict["FoVx"]); fovy = float(cam_dict["FoVy"])

    R = np.asarray(cam_dict["R"], dtype=np.float32).reshape(3,3)
    T = np.asarray(cam_dict["T"], dtype=np.float32).reshape(3)

    world_view = torch.tensor(getWorld2View2(R, T, np.array([0,0,0],dtype=np.float32), 1.0)).transpose(0,1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(W,H,fovy,fovx,znear,zfar,world_view,full)

# ---------- LRU cache ----------
class ModelCache:
    def __init__(self, max_items=3):
        self.max = max_items
        self.lock = threading.Lock()
        self._d = OrderedDict()  # key: (frame_idx, iter) -> GaussianModel

    def get(self, key):
        with self.lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key, value):
        with self.lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.max:
                k, v = self._d.popitem(last=False)
                del v
                torch.cuda.empty_cache()

# ---------- app ----------
def create_app(models_root: Path, prefix: str, default_iter: int, sh_degree: int,
               jpeg_quality: int, white_bg_default: bool, cache_size: int,
               preload_first: int, do_warmup: bool, neighbor_prefetch: bool):
    app = Flask(__name__, static_folder="web", static_url_path="/")
    cache = ModelCache(max_items=cache_size)

    # default pipeline
    pp = PipelineParams(argparse.ArgumentParser(add_help=False))
    default_pipe = pp.extract(pp.parser.parse_args([]))

    frames_all = list_frames(models_root, prefix)

    def load_model(frame_idx: int, iteration: int):
        mp = models_root / f"{prefix}{frame_idx}"
        if not mp.exists(): 
            raise FileNotFoundError(f"Model folder not found: {mp}")
        it = iteration if iteration >= 0 else latest_iteration(mp)
        if it < 0:
            raise FileNotFoundError(f"No iteration found under {mp}/point_cloud/")
        key = (frame_idx, it)
        gm = cache.get(key)
        if gm is not None:
            return gm, mp, it
        ply = mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if not ply.exists(): raise FileNotFoundError(f"PLY not found: {ply}")
        gm = GaussianModel(sh_degree)
        gm.load_ply(str(ply), use_train_test_exp=False)
        cache.put(key, gm)
        return gm, mp, it

    def warmup_once():
        if not frames_all: return
        try:
            f0 = frames_all[0]
            gm, _, it = load_model(f0, default_iter)
            # dummy tiny camera just to compile kernels
            W, H = 64, 64
            fovx = math.radians(60.0); fovy = math.radians(60.0)
            R = np.eye(3, dtype=np.float32); T = np.zeros(3, dtype=np.float32)
            world_view = torch.tensor(getWorld2View2(R, T, np.array([0,0,0],dtype=np.float32), 1.0)).transpose(0,1).cuda()
            proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
            full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            cam = MiniCam(W,H,fovy,fovx,0.01,100.0,world_view,full)
            bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                _ = render(cam, gm, default_pipe, bg, use_trained_exp=False)["render"]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            print(f"[TA-Server] Warmup done on frame {f0} (iter {it}).")
        except Exception as e:
            print(f"[TA-Server] Warmup failed: {e}")

    def prefill_first(n:int):
        n = max(0, min(n, len(frames_all)))
        for f in frames_all[:n]:
            try:
                gm, _, it = load_model(f, default_iter)
                print(f"[TA-Server] Prefilled frame {f} (iter {it})")
            except Exception as e:
                print(f"[TA-Server] Prefill failed for {f}: {e}")

    def prefetch_neighbors_async(center_f:int, it:int):
        if not neighbor_prefetch: return
        def _job():
            for nf in (center_f-1, center_f+1):
                if nf in frames_all:
                    try:
                        _ , _, _ = load_model(nf, default_iter)
                    except: pass
        t = threading.Thread(target=_job, daemon=True)
        t.start()

    # Prefill + warmup at startup
    if preload_first > 0:
        prefill_first(preload_first)
    if do_warmup:
        warmup_once()

    @app.get("/health")
    def health():
        return {"ok": True, "frames": frames_all, "cache_size": cache_size}

    @app.post("/render")
    def render_frame():
        """
        JSON:
        {
          "frame": 12, "iteration": -1,
          "camera": {...}, "white_background": false, "format": "jpeg"|"png",
          "pipeline": { ... }
        }
        """
        t0 = time.time()
        payload = request.get_json(force=True)
        frame_idx = int(payload["frame"])
        iteration = int(payload.get("iteration", default_iter))
        cam = build_minicam(payload["camera"])
        fmt = (payload.get("format") or "jpeg").lower()
        white_bg = bool(payload.get("white_background", white_bg_default))

        pipe = default_pipe
        if "pipeline" in payload and isinstance(payload["pipeline"], dict):
            for k, v in payload["pipeline"].items():
                if hasattr(pipe, k): setattr(pipe, k, v)

        bg_color = [1,1,1] if white_bg else [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gm, mp, it = load_model(frame_idx, iteration)  # LRU cached

        with torch.no_grad():
            out = render(cam, gm, pipe, background, use_trained_exp=False)["render"]
            img = (torch.clamp(out, 0, 1) * 255).byte().permute(1,2,0).contiguous().cpu().numpy()
        pil = Image.fromarray(img)

        buf = io.BytesIO()
        if fmt == "png":
            pil.save(buf, format="PNG")
            mime = "image/png"
        else:
            pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            mime = "image/jpeg"
        buf.seek(0)

        # 邻居预取（不会阻塞）
        prefetch_neighbors_async(frame_idx, it)

        resp = send_file(buf, mimetype=mime)
        resp.headers["X-Frame"] = str(frame_idx)
        resp.headers["X-Iter"]  = str(it)
        resp.headers["X-RT-ms"] = f"{(time.time()-t0)*1000:.1f}"
        return resp

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    return app

def main():
    ap = argparse.ArgumentParser("Time Archival 3DGS — Interactive Server (Prefetch+Warmup)")
    ap.add_argument("-m","--models_root", required=True, type=str)
    ap.add_argument("--prefix", default="model_frame_", type=str)
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--sh_degree", default=3, type=int)
    ap.add_argument("--host", default="0.0.0.0", type=str)
    ap.add_argument("--port", default=7860, type=int)
    ap.add_argument("--jpeg_quality", default=85, type=int)
    ap.add_argument("--white_background", action="store_true")
    # 新增
    ap.add_argument("--cache_size", type=int, default=3, help="LRU 缓存模型数（GPU 内存允许可增大）")
    ap.add_argument("--preload_first", type=int, default=0, help="启动时预先加载前 N 个帧")
    ap.add_argument("--warmup", action="store_true", help="启动时做一次渲染预热")
    ap.add_argument("--neighbor_prefetch", action="store_true", help="渲染后后台预取邻居帧")
    args = ap.parse_args()

    models_root = Path(args.models_root).resolve()
    app = create_app(models_root, args.prefix, args.iteration, args.sh_degree,
                     args.jpeg_quality, args.white_background, args.cache_size,
                     args.preload_first, args.warmup, args.neighbor_prefetch)
    print(f"[TA-Server] Serving models in {models_root}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
