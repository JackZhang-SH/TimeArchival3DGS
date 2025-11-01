#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Time Archival 3DGS — Server (CPU preload + GPU slots, native 3DGS semantics)

Usage:
  python ta_server_slots.py -p ../output_seq_packed --prefix model_frame_ \
    --slots 4 --warmup --neighbor_prefetch --camera_json ../camera.json \
    --host 0.0.0.0 --port 7860
"""
import io, math, argparse, threading, time, sys, importlib, re, json
from pathlib import Path
from collections import OrderedDict
import types, copy
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, send_file, jsonify

# -----------------------------------------------------------------------------
# 0) 将项目根目录注入到 sys.path（先做！）
# -----------------------------------------------------------------------------
def _add_repo_root():
    here = Path(__file__).resolve().parent
    cur = here
    for _ in range(5):
        if (cur / "scene").exists() or (cur / "server").exists() or (cur / "pack").exists():
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            return
        cur = cur.parent
_add_repo_root()

# 1) 导入原生组件（与 3DGS 对齐）
from scene.gaussian_model import GaussianModel as _GM


from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

# -----------------------------------------------------------------------------
# 2) 渲染后端（延迟加载，缺扩展给出安装指令）
# -----------------------------------------------------------------------------
def load_render_backend_or_die():
    try:
        mod = importlib.import_module("gaussian_renderer")
        fn = getattr(mod, "render")
        print("[Renderer] Using gaussian_renderer.render")
        return fn
    except Exception as e:
        raise ImportError(
            "\n[Renderer] 未找到 CUDA 扩展（diff_gaussian_rasterization / simple_knn）。\n"
            "请在当前环境安装（Windows 示例）：\n"
            "  pip install --upgrade pip setuptools wheel ninja cmake\n"
            "  pip install -e ./submodules/diff-gaussian-rasterization\n"
            "  pip install -e ./submodules/simple-knn\n"
            "若仓库无 submodules，可直接：\n"
            "  pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization\n"
            "  pip install git+https://github.com/graphdeco-inria/simple-knn\n"
            f"原始异常：{repr(e)}\n"
        ) from e


_PIPELINE_DEFAULTS = {
    "convert_SHs_python":   False,
    "compute_cov3D_python": False,
    "compute_cov2D_python": False,
    "antialiasing":         False,
    "sh_degree":            3,
    "debug":                False,
}

try:
    from arguments import PipelineParams as _GraphDECO_PipelineParams
except Exception:
    _GraphDECO_PipelineParams = None

def make_default_pipeline():
    if _GraphDECO_PipelineParams is not None:
        try:
            pp = _GraphDECO_PipelineParams(argparse.ArgumentParser(add_help=False))
            if hasattr(pp, "parser"):
                args = pp.parser.parse_args([])
                obj = pp.extract(args)
            else:
                obj = pp.extract([])
            for k, v in _PIPELINE_DEFAULTS.items():
                if not hasattr(obj, k): setattr(obj, k, v)
            return obj
        except Exception:
            pass
    return types.SimpleNamespace(**_PIPELINE_DEFAULTS)

def clone_pipeline(p):
    ns = types.SimpleNamespace()
    for k in dir(p):
        if k.startswith("_"): continue
        v = getattr(p, k)
        if callable(v): continue
        try: setattr(ns, k, copy.deepcopy(v))
        except Exception: setattr(ns, k, v)
    for k, v in _PIPELINE_DEFAULTS.items():
        if not hasattr(ns, k): setattr(ns, k, v)
    return ns

def safe_render(RENDER, cam, gauss, pipe, background, max_missing=16):
    """若渲染访问了不存在的 pipe 字段，自动补默认继续。"""
    tried = set()
    for _ in range(max_missing):
        try:
            return RENDER(cam, gauss, pipe, background, use_trained_exp=False)["render"]
        except AttributeError as e:
            m = re.search(r"has no attribute '([^']+)'", str(e))
            if not m: raise
            attr = m.group(1)
            if attr in tried: raise
            tried.add(attr)
            default = _PIPELINE_DEFAULTS.get(attr, False)
            setattr(pipe, attr, default)
            print(f"[Pipeline] missing '{attr}', default -> {default}")
    raise RuntimeError("safe_render exceeded max_missing")

# -----------------------------------------------------------------------------
# 4) 相机（原生 3DGS 数学）
# -----------------------------------------------------------------------------
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
        try:
            fovx = float(focal2fov(fx, W)); fovy = float(focal2fov(fy, H))
        except Exception:
            fovx = 2.0 * math.atan(W / (2.0 * fx)); fovy = 2.0 * math.atan(H / (2.0 * fy))
    else:
        fovx = float(cam_dict["FoVx"]); fovy = float(cam_dict["FoVy"])

    R = np.asarray(cam_dict["R"], dtype=np.float32).reshape(3,3)
    T = np.asarray(cam_dict["T"], dtype=np.float32).reshape(3)

    # --- 关键：统一放到 GPU 上 ---
    dev = torch.device("cuda")
    world_view = torch.tensor(
        getWorld2View2(R, T, np.array([0,0,0], dtype=np.float32), 1.0),
        dtype=torch.float32, device=dev
    ).transpose(0,1)

    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy) \
                .transpose(0,1) \
                .to(device=dev, dtype=torch.float32)

    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    return MiniCam(W, H, fovy, fovx, znear, zfar, world_view, full)


# -----------------------------------------------------------------------------
# 5) 发现打包帧 & CPU 预载
# -----------------------------------------------------------------------------
def list_packed_frames(packed_root: Path, prefix: str):
    frames = []; iters = {}
    for p in packed_root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            try: f = int(p.name.split("_")[-1])
            except: continue
            best_it = -1; best = None
            for q in p.iterdir():
                if q.is_file() and q.name.startswith("iter_") and q.suffix==".pt":
                    try:
                        it = int(q.stem.split("_")[1])
                        if it > best_it: best_it, best = it, q
                    except: pass
            if best is not None:
                frames.append(f); iters[f] = (best_it, best)
    frames.sort(); return frames, iters

class CpuCache:
    def __init__(self, packed_map: dict[int, tuple[int, Path]]):
        self.frames = sorted(packed_map.keys())
        self.data = {}; self.ni = {}; self.L = None; self.N_max = 0
        self._load_all(packed_map)
    def _pin(self, t: torch.Tensor):
        try: return t.pin_memory()
        except: return t
    def _load_all(self, packed_map):
        for f in self.frames:
            it, path = packed_map[f]
            pkg = torch.load(path, map_location="cpu")
            N = int(pkg["n"]); self.ni[f] = N; self.N_max = max(self.N_max, N)
            if self.L is None: self.L = int(pkg["sh_degree"])
            self.data[f] = {
                "iter": it,
                "xyz": self._pin(pkg["xyz"].contiguous()),
                "scaling": self._pin(pkg["scaling"].contiguous()),
                "rotation": self._pin(pkg["rotation"].contiguous()),
                "opacity": self._pin(pkg["opacity"].contiguous()),
                "sh_dc": self._pin(pkg["sh_dc"].contiguous()),
                "sh_rest": self._pin(pkg["sh_rest"].contiguous()),
            }
        print(f"[CPU] loaded {len(self.frames)} frames | N_max={self.N_max} | L={self.L}")

# -----------------------------------------------------------------------------
# 6) GPU 槽位（一次分配，反复覆盖）
# -----------------------------------------------------------------------------
class GpuSlot:
    def __init__(self, device: torch.device, N_max: int, L: int):
        self.dev = device; self.N_max = N_max; self.L = L
        self.xyz      = torch.empty((N_max,3), dtype=torch.float32, device=device)
        self.scaling  = torch.empty((N_max,3), dtype=torch.float16, device=device)  # log-scale
        self.rotation = torch.empty((N_max,4), dtype=torch.float16, device=device)  # quat
        self.opacity  = torch.empty((N_max,1), dtype=torch.float16, device=device)  # pre-sigmoid
        self.K = (L+1)*(L+1) - 1
        self.sh_dc    = torch.empty((N_max,3,1), dtype=torch.float16, device=device)
        self.sh_rest  = torch.empty((N_max,3,self.K), dtype=torch.float16, device=device)
        self.valid = 0; self.frame_id = None
        self.stream = torch.cuda.Stream(device=device)
        self.event_ready = torch.cuda.Event(blocking=False, interprocess=False)
        self.in_use = 0
    def async_upload(self, cpu_pkg: dict, N_i: int):
        assert N_i <= self.N_max
        with torch.cuda.stream(self.stream):
            self.xyz[:N_i].copy_(cpu_pkg["xyz"][:N_i], non_blocking=True)
            self.scaling[:N_i].copy_(cpu_pkg["scaling"][:N_i], non_blocking=True)
            self.rotation[:N_i].copy_(cpu_pkg["rotation"][:N_i], non_blocking=True)
            self.opacity[:N_i].copy_(cpu_pkg["opacity"][:N_i], non_blocking=True)
            self.sh_dc[:N_i].copy_(cpu_pkg["sh_dc"][:N_i], non_blocking=True)
            self.sh_rest[:N_i].copy_(cpu_pkg["sh_rest"][:N_i], non_blocking=True)
            self.event_ready.record(self.stream)
        self.valid = N_i
    def wait_ready(self):
        torch.cuda.current_stream(device=self.dev).wait_event(self.event_ready)

class SlotManager:
    def __init__(self, num_slots: int, N_max: int, L: int, device: str = "cuda"):
        self.dev = torch.device(device)
        self.slots = [GpuSlot(self.dev, N_max, L) for _ in range(num_slots)]
        self.frame2slot: dict[int,int] = {}
        self.lru = OrderedDict()
        self.lock = threading.RLock()  # <<< 新增

    def has(self, frame_id: int):
        with self.lock:
            return frame_id in self.frame2slot

    def get_slot_for(self, frame_id: int):
        with self.lock:
            sid = self.frame2slot.get(frame_id)
            return None if sid is None else self.slots[sid]

    def touch(self, frame_id: int):
        # 仅在持锁语境下被调用
        if frame_id in self.lru: self.lru.move_to_end(frame_id)
        else: self.lru[frame_id] = None

    def _evict_one_locked(self):
        # 调用方已持 self.lock
        used = set(self.frame2slot.values())
        free = list(set(range(len(self.slots))) - used)
        if free:
            return free[0]

        # 找到最老且不在使用中的槽位
        for old in list(self.lru.keys()):
            sid = self.frame2slot[old]
            if self.slots[sid].in_use == 0:
                self.lru.pop(old)
                self.frame2slot.pop(old)
                self.slots[sid].frame_id = None
                return sid

        # 如果全在使用中，短暂等待并重试（避免死锁/自旋）
        while True:
            for old in list(self.lru.keys()):
                sid = self.frame2slot[old]
                if self.slots[sid].in_use == 0:
                    self.lru.pop(old)
                    self.frame2slot.pop(old)
                    self.slots[sid].frame_id = None
                    return sid
            time.sleep(0.001)

    def ensure_on_gpu(self, frame_id: int, cpu_pkg: dict, N_i: int):
        with self.lock:
            if frame_id in self.frame2slot:
                self.touch(frame_id)
                return self.slots[self.frame2slot[frame_id]]
            sid = self._evict_one_locked()
            slot = self.slots[sid]
            slot.async_upload(cpu_pkg, N_i)
            slot.frame_id = frame_id
            self.frame2slot[frame_id] = sid
            self.touch(frame_id)
            return slot

    @contextmanager
    def lease(self, frame_id: int):
        """渲染期间占用该帧所在槽位，禁止回收。"""
        with self.lock:
            sid = self.frame2slot.get(frame_id)
            if sid is None:
                raise KeyError(f"frame {frame_id} not on GPU")
            slot = self.slots[sid]
            slot.in_use += 1
        try:
            yield slot
        finally:
            with self.lock:
                slot.in_use -= 1


# -----------------------------------------------------------------------------
# 7) 槽位 → 原生 GaussianModel（严格等价语义）
# -----------------------------------------------------------------------------
def make_gaussian_from_slot(slot: GpuSlot) -> _GM:
    """
    将槽位中的张量“挂载”为一个只读 GaussianModel：
      - _xyz:               [N,3]
      - _features_dc:       [N,1,3]   （注意：通道在最后维度，和 3DGS 一致）
      - _features_rest:     [N,K,3]
      - _scaling/_rotation/_opacity: 前激活域（交由 GM property 做 exp/sigmoid/normalize）
    """
    M = _GM(slot.L)
    M.max_sh_degree = slot.L
    M.active_sh_degree = slot.L

    N = slot.valid
    M._xyz = slot.xyz[:N].to(torch.float32)

    # from [N,3,1] -> [N,1,3] ; from [N,3,K] -> [N,K,3]
    M._features_dc   = slot.sh_dc[:N].to(torch.float32).permute(0,2,1).contiguous()
    M._features_rest = slot.sh_rest[:N].to(torch.float32).permute(0,2,1).contiguous()

    M._scaling  = slot.scaling[:N].to(torch.float32)   # log-scale
    M._rotation = slot.rotation[:N].to(torch.float32)  # quat（GM 内部会 normalize）
    M._opacity  = slot.opacity[:N].to(torch.float32)   # pre-sigmoid
    return M

# -----------------------------------------------------------------------------
# 8) Flask 应用
# -----------------------------------------------------------------------------
def create_app(packed_root: Path, prefix: str, slots: int, jpeg_quality: int,
               white_bg_default: bool, warmup: bool, neighbor_prefetch: bool,
               camera_json: Path | None):
    app = Flask(__name__, static_folder="web", static_url_path="/")

    RENDER = load_render_backend_or_die()

    # 默认相机
    default_cam = None
    if camera_json is not None and camera_json.exists():
        with open(camera_json, "r", encoding="utf-8") as f:
            default_cam = json.load(f)
        for k in ["width","height","R","T","fx","fy","cx","cy"]:
            if k not in default_cam:
                print(f"[Camera] WARN: default camera missing '{k}'")
        print(f"[Camera] loaded {camera_json} as home camera")
    else:
        print("[Camera] no camera.json (or path not found) -> starting at freecam origin")

    frames, itmap = list_packed_frames(packed_root, prefix)
    if not frames:
        raise RuntimeError("No packed frames found. Run ta_pack.py first.")
    cpu_cache = CpuCache(itmap)
    sm = SlotManager(slots, cpu_cache.N_max, cpu_cache.L, device="cuda")

    default_pipe = make_default_pipeline()
    default_pipe.antialiasing = False          # 关键：关闭 AA，显存峰值立刻大幅下降
    default_pipe.compute_cov2D_python = True   # 让 2D 协方差走 PyTorch 路径，减少扩展工作区
    default_pipe.convert_SHs_python = True     # 同理（通常对速度影响很小）
    def _scale_cam_intrinsics(cam_dict, new_w, new_h):
        sw = new_w / float(cam_dict["width"]); sh = new_h / float(cam_dict["height"])
        cd = dict(cam_dict)
        cd["width"], cd["height"] = int(new_w), int(new_h)
        cd["fx"], cd["fy"] = cd["fx"] * sw, cd["fy"] * sh
        cd["cx"], cd["cy"] = cd["cx"] * sw, cd["cy"] * sh
        return cd

    def do_warmup():
        try:
            f0 = frames[0]
            t0 = time.time()
            pkg = cpu_cache.data[f0]; N0 = cpu_cache.ni[f0]
            slot = sm.ensure_on_gpu(f0, pkg, N0); slot.wait_ready()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            load_ms = (time.time()-t0)*1000.0
            # 渲染 64x64
            warm_cam_dict = _scale_cam_intrinsics(default_cam, 64, 64) if default_cam else {
                "width":64, "height":64, "fx":32, "fy":32, "cx":32, "cy":32,
                "R":[1,0,0, 0,1,0, 0,0,1], "T":[0,0,0], "znear":0.01, "zfar":10.0
            }
            cam = build_minicam(warm_cam_dict)
            bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
            t1 = time.time()
            with torch.no_grad():
                _ = safe_render(RENDER, cam, make_gaussian_from_slot(slot), clone_pipeline(default_pipe), bg)
                if torch.cuda.is_available(): torch.cuda.synchronize()
            render_ms = (time.time()-t1)*1000.0
            print(f"[Warmup] ok on frame {f0} | H2D={load_ms:.1f} ms | render={render_ms:.1f} ms")
        except Exception as e:
            print(f"[Warmup] failed: {e}")


    if warmup: do_warmup()

    def prefetch_neighbors(center_f: int):
        if not neighbor_prefetch: return
        for nf in (center_f-1, center_f+1):
            if nf in cpu_cache.frames and not sm.has(nf):
                pkg = cpu_cache.data[nf]; N = cpu_cache.ni[nf]
                threading.Thread(target=lambda: sm.ensure_on_gpu(nf, pkg, N), daemon=True).start()

    @app.get("/health")
    def health():
        return jsonify({
            "ok": True,
            "frames": frames,
            "slots": slots,
            "N_max": cpu_cache.N_max,
            "L": cpu_cache.L,
            "default_camera": default_cam
        })

    @app.get("/home_cam")
    def home_cam():
        if default_cam is None:
            return jsonify({"ok": False, "reason": "no camera.json"}), 404
        return jsonify({"ok": True, "camera": default_cam})
    @app.post("/load")
    def load_frame():
        payload = request.get_json(force=True)
        frame_idx = int(payload["frame"])
        cache_hit = sm.has(frame_idx)

        t0 = time.time()
        pkg = cpu_cache.data[frame_idx]; N_i = cpu_cache.ni[frame_idx]
        slot = sm.ensure_on_gpu(frame_idx, pkg, N_i)
        slot.wait_ready()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        load_ms = (time.time() - t0) * 1000.0

        print(f"[Load]  frame={frame_idx} | cache={'hit' if cache_hit else 'miss'} | H2D+wait={load_ms:.1f} ms | valid={slot.valid}/{slot.N_max}")
        prefetch_neighbors(frame_idx)

        resp = jsonify({"ok": True, "frame": frame_idx, "valid": slot.valid, "cache": ("hit" if cache_hit else "miss")})
        resp.headers["X-Load-ms"] = f"{load_ms:.1f}"
        resp.headers["X-Cache"] = "hit" if cache_hit else "miss"
        return resp


    @app.post("/render")
    def render_frame():
        t_all0 = time.time()
        payload = request.get_json(force=True)
        frame_idx = int(payload["frame"])
        fmt = (payload.get("format") or "jpeg").lower()
        white_bg = bool(payload.get("white_background", white_bg_default))
        cam = build_minicam(payload["camera"])

        pipe = clone_pipeline(default_pipe)
        if "pipeline" in payload and isinstance(payload["pipeline"], dict):
            for k, v in payload["pipeline"].items():
                setattr(pipe, k, v)

        background = torch.tensor(([1,1,1] if white_bg else [0,0,0]),
                                  dtype=torch.float32, device="cuda")

        cache_hit = sm.has(frame_idx)
        t_h2d0 = time.time()
        pkg = cpu_cache.data[frame_idx]; N_i = cpu_cache.ni[frame_idx]
        sm.ensure_on_gpu(frame_idx, pkg, N_i)  # 先确保在 GPU

        with sm.lease(frame_idx) as slot:
            slot.wait_ready()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            h2d_ms = (time.time() - t_h2d0) * 1000.0

            gauss = make_gaussian_from_slot(slot)
            print(f"[Render] frame={frame_idx} | cache={'hit' if cache_hit else 'miss'} | valid={slot.valid}/{slot.N_max}")

            # --- 渲染计时 ---
            t_r0 = time.time()
            with torch.no_grad():
                out = safe_render(RENDER, cam, gauss, pipe, background)
                if torch.cuda.is_available(): torch.cuda.synchronize()
            render_ms = (time.time() - t_r0) * 1000.0

            # --- 编码计时 ---
            t_e0 = time.time()
            img = (torch.clamp(out, 0, 1) * 255).byte().permute(1,2,0).contiguous().cpu().numpy()
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            if fmt == "png":
                pil.save(buf, format="PNG"); mime = "image/png"
            else:
                pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True); mime = "image/jpeg"
            buf.seek(0)
            encode_ms = (time.time() - t_e0) * 1000.0

        prefetch_neighbors(frame_idx)

        total_ms = (time.time() - t_all0) * 1000.0
        print(f"[Render] done  frame={frame_idx} | H2D={h2d_ms:.1f} ms | render={render_ms:.1f} ms | encode={encode_ms:.1f} ms | total={total_ms:.1f} ms")

        resp = send_file(buf, mimetype=mime)
        resp.headers["X-Frame"] = str(frame_idx)
        resp.headers["X-Cache"] = "hit" if cache_hit else "miss"
        resp.headers["X-H2D-ms"] = f"{h2d_ms:.1f}"
        resp.headers["X-Render-ms"] = f"{render_ms:.1f}"
        resp.headers["X-Encode-ms"] = f"{encode_ms:.1f}"
        resp.headers["X-RT-ms"] = f"{total_ms:.1f}"
        return resp



    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    return app

def main():
    ap = argparse.ArgumentParser("Time Archival 3DGS — Server (CPU preload + GPU slots)")
    ap.add_argument("-p","--packed_root", required=True, type=str, help="folder of packed .pt (or dataset subfolder)")
    ap.add_argument("--prefix", default="model_frame_", type=str)
    ap.add_argument("--slots", type=int, default=3, help="GPU slot window size")
    ap.add_argument("--host", default="0.0.0.0", type=str)
    ap.add_argument("--port", default=7860, type=int)
    ap.add_argument("--jpeg_quality", default=85, type=int)
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--neighbor_prefetch", action="store_true")
    ap.add_argument("--camera_json", default="camera.json", type=str,
                    help="path to default camera json (optional)")
    args = ap.parse_args()

    app = create_app(
        Path(args.packed_root).resolve(),
        args.prefix, args.slots, args.jpeg_quality,
        args.white_background, args.warmup, args.neighbor_prefetch,
        Path(args.camera_json) if args.camera_json else None
    )
    print(f"[TA-ServerSlots] Serving packed from {args.packed_root} | slots={args.slots}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
