#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Time Archival 3DGS — Server (CPU preload + GPU slots, native 3DGS semantics)

Features
--------
- Preload packed 3DGS frames (.pt from ta_pack.py) into CPU RAM.
- Maintain a small number of GPU "slots" that cache recent frames.
- Optional static packed scene (--static_pt) that is added to every frame
  (A + filtered-B, implemented as concatenation in 3DGS space, not image-space).
- Expose HTTP endpoints (/health, /home_cam, /load, /render) for a web viewer.

Typical usage
-------------
Packed merged or per-frame 3DGS:

    python ta_server_slots.py \
        -p output_seq_packed/soccer_merged \
        --prefix model_frame_ \
        --slots 4 --warmup --neighbor_prefetch \
        --camera_json camera.json \
        --host 0.0.0.0 --port 7860

Static A + dynamic filtered-B (recommended for memory efficiency):

    # A.pt is the packed static scene (single .pt)
    # B_root is packed per-frame filtered-B models (model_frame_n/*.pt)
    python ta_server_slots.py \
        -p output_seq_packed/soccer_B_filtered \
        --prefix model_frame_ \
        --static_pt output_seq_packed/static_A.pt \
        --slots 4 --warmup --neighbor_prefetch \
        --camera_json camera.json \
        --host 0.0.0.0 --port 7860
"""

import io
import math
import argparse
import threading
import time
import sys
import importlib
import re
import json
from pathlib import Path
from collections import OrderedDict
import types
import copy
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, send_file, jsonify

# -----------------------------------------------------------------------------
# 0) Add project root to sys.path so "scene", "server", "pack" imports work
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

# 1) Import native 3DGS components
from scene.gaussian_model import GaussianModel as _GM
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

# -----------------------------------------------------------------------------
# 2) Render backend (lazy import, with installation hints if missing)
# -----------------------------------------------------------------------------
def load_render_backend_or_die():
    try:
        mod = importlib.import_module("gaussian_renderer")
        fn = getattr(mod, "render")
        print("[Renderer] Using gaussian_renderer.render")
        return fn
    except Exception as e:
        raise ImportError(
            "\n[Renderer] Failed to import CUDA renderer 'gaussian_renderer'.\n"
            "You need the diff-gaussian-rasterization and simple-knn extensions.\n"
            "Example installation (Windows / general):\n"
            "  pip install --upgrade pip setuptools wheel ninja cmake\n"
            "  pip install -e ./submodules/diff-gaussian-rasterization\n"
            "  pip install -e ./submodules/simple-knn\n"
            "If you do not have the submodules, you can install from GitHub:\n"
            "  pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization\n"
            "  pip install git+https://github.com/graphdeco-inria/simple-knn\n"
            f"\nOriginal exception:\n  {repr(e)}\n"
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
    """Create a default PipelineParams object, falling back to a simple namespace."""
    if _GraphDECO_PipelineParams is not None:
        try:
            pp = _GraphDECO_PipelineParams(argparse.ArgumentParser(add_help=False))
            if hasattr(pp, "parser"):
                args = pp.parser.parse_args([])
                obj = pp.extract(args)
            else:
                obj = pp.extract([])
            for k, v in _PIPELINE_DEFAULTS.items():
                if not hasattr(obj, k):
                    setattr(obj, k, v)
            return obj
        except Exception:
            pass
    return types.SimpleNamespace(**_PIPELINE_DEFAULTS)


def clone_pipeline(p):
    """Deep-copy a pipeline config while keeping non-callable public attributes."""
    ns = types.SimpleNamespace()
    for k in dir(p):
        if k.startswith("_"):
            continue
        v = getattr(p, k)
        if callable(v):
            continue
        try:
            setattr(ns, k, copy.deepcopy(v))
        except Exception:
            setattr(ns, k, v)
    for k, v in _PIPELINE_DEFAULTS.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def safe_render(RENDER, cam, gauss, pipe, background, max_missing: int = 16):
    """
    Call gaussian_renderer.render with a pipeline object, and if the renderer
    accesses a missing attribute on `pipe`, automatically add that attribute
    with a default value from _PIPELINE_DEFAULTS and retry.
    """
    tried = set()
    for _ in range(max_missing):
        try:
            return RENDER(cam, gauss, pipe, background, use_trained_exp=False)["render"]
        except AttributeError as e:
            m = re.search(r"has no attribute '([^']+)'", str(e))
            if not m:
                raise
            attr = m.group(1)
            if attr in tried:
                raise
            tried.add(attr)
            default = _PIPELINE_DEFAULTS.get(attr, False)
            setattr(pipe, attr, default)
            print(f"[Pipeline] Missing '{attr}', setting default -> {default}")
    raise RuntimeError("safe_render exceeded max_missing attempts")


# -----------------------------------------------------------------------------
# 4) Cameras (native 3DGS math)
# -----------------------------------------------------------------------------
class MiniCam:
    def __init__(self, W, H, fovy, fovx, znear, zfar,
                 world_view_transform, full_proj_transform):
        self.image_width = W
        self.image_height = H
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        # Camera center in world space
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


def build_minicam(cam_dict: dict) -> MiniCam:
    W = int(cam_dict["width"])
    H = int(cam_dict["height"])
    znear = float(cam_dict.get("znear", 0.01))
    zfar = float(cam_dict.get("zfar", 100.0))

    if all(k in cam_dict for k in ("fx", "fy", "cx", "cy")):
        fx, fy = float(cam_dict["fx"]), float(cam_dict["fy"])
        try:
            fovx = float(focal2fov(fx, W))
            fovy = float(focal2fov(fy, H))
        except Exception:
            fovx = 2.0 * math.atan(W / (2.0 * fx))
            fovy = 2.0 * math.atan(H / (2.0 * fy))
    else:
        fovx = float(cam_dict["FoVx"])
        fovy = float(cam_dict["FoVy"])

    R = np.asarray(cam_dict["R"], dtype=np.float32).reshape(3, 3)
    T = np.asarray(cam_dict["T"], dtype=np.float32).reshape(3)

    dev = torch.device("cuda")

    world_view = torch.tensor(
        getWorld2View2(R, T, np.array([0, 0, 0], dtype=np.float32), 1.0),
        dtype=torch.float32,
        device=dev,
    ).transpose(0, 1)

    proj = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        .transpose(0, 1)
        .to(device=dev, dtype=torch.float32)
    )

    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    return MiniCam(W, H, fovy, fovx, znear, zfar, world_view, full)


# -----------------------------------------------------------------------------
# 5) Packed frames discovery + CPU cache
# -----------------------------------------------------------------------------
def list_packed_frames(packed_root: Path, prefix: str):
    """
    Discover packed frames under packed_root.

    1) "Merged root" layout:
       <packed_root>/<prefix><n>/<prefix><n>.pt
       e.g. output_seq_packed/soccer_merged/model_frame_3/model_frame_3.pt

    2) Legacy layout:
       <packed_root>/<prefix><n>/iter_XXXX.pt
       We pick the largest XXXX (last iteration).
    """
    frames = []
    iters: dict[int, tuple[int, Path]] = {}

    for p in packed_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(prefix):
            continue
        try:
            f = int(name.split("_")[-1])
        except Exception:
            continue

        # Preferred: <root>/<prefix><n>/<prefix><n>.pt
        samefile = p / f"{name}.pt"
        if samefile.exists() and samefile.is_file():
            frames.append(f)
            iters[f] = (0, samefile)
            continue

        # Fallback: iter_*.pt, choose largest iter
        best_it = -1
        best = None
        for q in p.iterdir():
            if not q.is_file() or q.suffix != ".pt":
                continue
            stem = q.stem
            if stem.startswith("iter_"):
                try:
                    it = int(stem.split("_", 1)[1])
                except Exception:
                    continue
                if it > best_it:
                    best_it = it
                    best = q
        if best is not None:
            frames.append(f)
            iters[f] = (best_it, best)

    frames.sort()
    return frames, iters


class CpuCache:
    """
    Load all packed frames into pinned CPU memory.

    packed_map: dict[frame_id] -> (iteration, path_to_pt)
    """

    def __init__(self, packed_map: dict[int, tuple[int, Path]]):
        self.frames = sorted(packed_map.keys())
        self.data: dict[int, dict] = {}
        self.ni: dict[int, int] = {}
        self.L: int | None = None
        self.N_max: int = 0
        self._load_all(packed_map)

    def _pin(self, t: torch.Tensor):
        try:
            return t.pin_memory()
        except Exception:
            return t

    def _load_all(self, packed_map: dict[int, tuple[int, Path]]):
        for f in self.frames:
            it, path = packed_map[f]
            pkg = torch.load(path, map_location="cpu")
            N = int(pkg["n"])
            self.ni[f] = N
            self.N_max = max(self.N_max, N)
            if self.L is None:
                self.L = int(pkg["sh_degree"])
            self.data[f] = {
                "iter": it,
                "xyz": self._pin(pkg["xyz"].contiguous()),
                "scaling": self._pin(pkg["scaling"].contiguous()),
                "rotation": self._pin(pkg["rotation"].contiguous()),
                "opacity": self._pin(pkg["opacity"].contiguous()),
                "sh_dc": self._pin(pkg["sh_dc"].contiguous()),
                "sh_rest": self._pin(pkg["sh_rest"].contiguous()),
            }
        print(
            f"[CPU] Loaded {len(self.frames)} frames | "
            f"N_max={self.N_max} | L={self.L}"
        )


# -----------------------------------------------------------------------------
# 6) GPU slots (allocated once, overwritten for each frame)
# -----------------------------------------------------------------------------
class GpuSlot:
    """
    One GPU "slot" that can hold:
      - Optional static A Gaussians (prefix of the arrays, constant across frames).
      - One dynamic frame (e.g. filtered-B) appended after A.

    The `valid` count is the total number of active Gaussians (A + B).
    """

    def __init__(
        self,
        device: torch.device,
        N_max: int,
        L: int,
        static_pkg: dict | None = None,
    ):
        self.dev = device
        self.N_max = N_max
        self.L = L

        self.xyz = torch.empty((N_max, 3), dtype=torch.float32, device=device)
        self.scaling = torch.empty(
            (N_max, 3), dtype=torch.float16, device=device
        )  # log-scales
        self.rotation = torch.empty(
            (N_max, 4), dtype=torch.float16, device=device
        )  # quaternions
        self.opacity = torch.empty(
            (N_max, 1), dtype=torch.float16, device=device
        )  # pre-sigmoid

        self.K = (L + 1) * (L + 1) - 1
        self.sh_dc = torch.empty(
            (N_max, 3, 1), dtype=torch.float16, device=device
        )  # [N,3,1]
        self.sh_rest = torch.empty(
            (N_max, 3, self.K), dtype=torch.float16, device=device
        )  # [N,3,K]

        self.N_static: int = 0  # how many leading entries belong to static A
        self.valid: int = 0     # total active entries (A + B)
        self.frame_id: int | None = None

        self.stream = torch.cuda.Stream(device=device)
        self.event_ready = torch.cuda.Event(
            blocking=False, interprocess=False
        )
        self.in_use: int = 0

        # Upload static A (if provided) into the beginning of the slot
        if static_pkg is not None:
            Ns = int(static_pkg["xyz"].shape[0])
            if Ns > N_max:
                raise ValueError(
                    f"static_pkg.n = {Ns} exceeds slot capacity N_max = {N_max}"
                )
            self.N_static = Ns
            with torch.cuda.stream(self.stream):
                self.xyz[:Ns].copy_(static_pkg["xyz"], non_blocking=True)
                self.scaling[:Ns].copy_(static_pkg["scaling"], non_blocking=True)
                self.rotation[:Ns].copy_(static_pkg["rotation"], non_blocking=True)
                self.opacity[:Ns].copy_(static_pkg["opacity"], non_blocking=True)
                self.sh_dc[:Ns].copy_(static_pkg["sh_dc"], non_blocking=True)
                self.sh_rest[:Ns].copy_(static_pkg["sh_rest"], non_blocking=True)
                self.event_ready.record(self.stream)
            self.valid = Ns

    def async_upload(self, cpu_pkg: dict, N_i: int):
        """
        Asynchronously upload a dynamic frame (B) into this slot, *after*
        the static prefix (if any). N_i is the number of dynamic Gaussians.
        """
        start = self.N_static
        assert start + N_i <= self.N_max, (
            f"Too many Gaussians for this slot: start={start}, N_i={N_i}, "
            f"N_max={self.N_max}"
        )
        with torch.cuda.stream(self.stream):
            self.xyz[start : start + N_i].copy_(
                cpu_pkg["xyz"][:N_i], non_blocking=True
            )
            self.scaling[start : start + N_i].copy_(
                cpu_pkg["scaling"][:N_i], non_blocking=True
            )
            self.rotation[start : start + N_i].copy_(
                cpu_pkg["rotation"][:N_i], non_blocking=True
            )
            self.opacity[start : start + N_i].copy_(
                cpu_pkg["opacity"][:N_i], non_blocking=True
            )
            self.sh_dc[start : start + N_i].copy_(
                cpu_pkg["sh_dc"][:N_i], non_blocking=True
            )
            self.sh_rest[start : start + N_i].copy_(
                cpu_pkg["sh_rest"][:N_i], non_blocking=True
            )
            self.event_ready.record(self.stream)
        # total = static + dynamic
        self.valid = start + N_i

    def wait_ready(self):
        torch.cuda.current_stream(device=self.dev).wait_event(self.event_ready)


class SlotManager:
    """
    LRU-managed collection of GPU slots.

    Each slot can hold:
      - optional static A Gaussians (fixed prefix)
      - one dynamic frame (B) appended after A

    We keep a mapping frame_id -> slot_index and an LRU list to evict
    the least-recently-used (and currently idle) slot when needed.
    """

    def __init__(
        self,
        num_slots: int,
        N_max: int,
        L: int,
        device: str = "cuda",
        static_pkg: dict | None = None,
    ):
        self.dev = torch.device(device)
        self.slots = [
            GpuSlot(self.dev, N_max, L, static_pkg=static_pkg)
            for _ in range(num_slots)
        ]
        self.frame2slot: dict[int, int] = {}
        self.lru: OrderedDict[int, None] = OrderedDict()
        self.lock = threading.RLock()

    def has(self, frame_id: int) -> bool:
        with self.lock:
            return frame_id in self.frame2slot

    def get_slot_for(self, frame_id: int) -> GpuSlot | None:
        with self.lock:
            sid = self.frame2slot.get(frame_id)
            return None if sid is None else self.slots[sid]

    def touch(self, frame_id: int):
        """Mark a frame as recently used (LRU bookkeeping)."""
        if frame_id in self.lru:
            self.lru.move_to_end(frame_id)
        else:
            self.lru[frame_id] = None

    def _evict_one_locked(self) -> int:
        """
        Choose a slot to use (either a free slot index or the LRU idle one).
        Caller must hold self.lock.
        """
        used = set(self.frame2slot.values())
        free = list(set(range(len(self.slots))) - used)
        if free:
            return free[0]

        # Otherwise, find the least recently used slot whose in_use == 0.
        for old in list(self.lru.keys()):
            sid = self.frame2slot[old]
            if self.slots[sid].in_use == 0:
                self.lru.pop(old)
                self.frame2slot.pop(old)
                self.slots[sid].frame_id = None
                return sid

        # All slots are currently in use; wait briefly and retry.
        while True:
            for old in list(self.lru.keys()):
                sid = self.frame2slot[old]
                if self.slots[sid].in_use == 0:
                    self.lru.pop(old)
                    self.frame2slot.pop(old)
                    self.slots[sid].frame_id = None
                    return sid
            time.sleep(0.001)

    def ensure_on_gpu(self, frame_id: int, cpu_pkg: dict, N_i: int) -> GpuSlot:
        """
        Ensure that the given frame is resident in some slot on the GPU.
        If it is already loaded, return that slot (no new upload).
        Otherwise, evict an old slot and upload the frame's dynamic part.
        """
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
        """
        Context manager for using a frame's slot.
        While leased, the slot's in_use counter > 0, so it will not be evicted.
        """
        with self.lock:
            sid = self.frame2slot.get(frame_id)
            if sid is None:
                raise KeyError(f"Frame {frame_id} not on GPU")
            slot = self.slots[sid]
            slot.in_use += 1
        try:
            yield slot
        finally:
            with self.lock:
                slot.in_use -= 1


# -----------------------------------------------------------------------------
# 7) Convert a GpuSlot into a read-only GaussianModel
# -----------------------------------------------------------------------------
def make_gaussian_from_slot(slot: GpuSlot) -> _GM:
    """
    Wrap the slot's GPU tensors in a GaussianModel instance.

    The slot contains:
      - xyz:        [N,3]
      - scaling:    [N,3] (log-scale)
      - rotation:   [N,4] (quaternion, will be normalized by GM)
      - opacity:    [N,1] (pre-sigmoid)
      - sh_dc:      [N,3,1]   (we permute to [N,1,3])
      - sh_rest:    [N,3,K]   (we permute to [N,K,3])
    """
    M = _GM(slot.L)
    M.max_sh_degree = slot.L
    M.active_sh_degree = slot.L

    N = slot.valid
    M._xyz = slot.xyz[:N].to(torch.float32)

    # [N,3,1] -> [N,1,3], [N,3,K] -> [N,K,3]
    M._features_dc = slot.sh_dc[:N].to(torch.float32).permute(0, 2, 1).contiguous()
    M._features_rest = (
        slot.sh_rest[:N].to(torch.float32).permute(0, 2, 1).contiguous()
    )

    M._scaling = slot.scaling[:N].to(torch.float32)   # log-scale
    M._rotation = slot.rotation[:N].to(torch.float32)  # quaternion
    M._opacity = slot.opacity[:N].to(torch.float32)   # pre-sigmoid
    return M


# -----------------------------------------------------------------------------
# 8) Flask application
# -----------------------------------------------------------------------------
def create_app(
    packed_root: Path,
    prefix: str,
    slots: int,
    jpeg_quality: int,
    white_bg_default: bool,
    warmup: bool,
    neighbor_prefetch: bool,
    camera_json: Path | None,
    static_pt: Path | None = None,
):
    app = Flask(__name__, static_folder="web", static_url_path="/")

    RENDER = load_render_backend_or_die()

    # -------------------------------------------------------------------------
    # Default camera (optional, used as "home" / viewer reset camera)
    # -------------------------------------------------------------------------
    default_cam = None
    if camera_json is not None and camera_json.exists():
        with open(camera_json, "r", encoding="utf-8") as f:
            default_cam = json.load(f)
        for k in ["width", "height", "R", "T", "fx", "fy", "cx", "cy"]:
            if k not in default_cam:
                print(f"[Camera] WARN: default camera missing '{k}'")
        print(f"[Camera] Loaded {camera_json} as home camera")
    else:
        print("[Camera] No camera.json found -> starting at freecam origin")

    frames, itmap = list_packed_frames(packed_root, prefix)
    if not frames:
        raise RuntimeError("No packed frames found. Run ta_pack.py first.")
    cpu_cache = CpuCache(itmap)

    # -------------------------------------------------------------------------
    # Optional static A scene (added to every frame)
    # -------------------------------------------------------------------------
    static_pkg: dict | None = None
    static_n = 0
    if static_pt is not None and static_pt.exists():
        raw = torch.load(static_pt, map_location="cpu")
        static_n = int(raw["n"])
        L_static = int(raw["sh_degree"])

        if cpu_cache.L is not None and L_static != cpu_cache.L:
            raise RuntimeError(
                f"sh_degree mismatch: static_pt={L_static} vs dynamic={cpu_cache.L}"
            )
        if cpu_cache.L is None:
            cpu_cache.L = L_static

        def _pin_static(t: torch.Tensor):
            try:
                return t.contiguous().pin_memory()
            except Exception:
                return t.contiguous()

        static_pkg = {
            "xyz": _pin_static(raw["xyz"]),
            "scaling": _pin_static(raw["scaling"]),
            "rotation": _pin_static(raw["rotation"]),
            "opacity": _pin_static(raw["opacity"]),
            "sh_dc": _pin_static(raw["sh_dc"]),
            "sh_rest": _pin_static(raw["sh_rest"]),
        }
        print(f"[Static] Loaded {static_pt} | N={static_n} | L={L_static}")
    else:
        print("[Static] No static_pt provided -> rendering dynamic frames only")

    total_N_max = cpu_cache.N_max + static_n
    sm = SlotManager(
        slots,
        total_N_max,
        cpu_cache.L,
        device="cuda",
        static_pkg=static_pkg,
    )

    # -------------------------------------------------------------------------
    # Default pipeline tuning (memory-friendly)
    # -------------------------------------------------------------------------
    default_pipe = make_default_pipeline()
    # Important memory tweaks:
    default_pipe.antialiasing = False          # Disable AA to reduce peak VRAM
    default_pipe.compute_cov2D_python = True   # Use PyTorch path for 2D covariance
    default_pipe.convert_SHs_python = True     # Convert SHs in Python (small overhead)

    def _scale_cam_intrinsics(cam_dict, new_w, new_h):
        sw = new_w / float(cam_dict["width"])
        sh = new_h / float(cam_dict["height"])
        cd = dict(cam_dict)
        cd["width"], cd["height"] = int(new_w), int(new_h)
        cd["fx"], cd["fy"] = cd["fx"] * sw, cd["fy"] * sh
        cd["cx"], cd["cy"] = cd["cx"] * sw, cd["cy"] * sh
        return cd

    # -------------------------------------------------------------------------
    # Optional warmup on the first frame (helps to measure baseline timings)
    # -------------------------------------------------------------------------
    def do_warmup():
        try:
            f0 = frames[0]
            t0 = time.time()
            pkg = cpu_cache.data[f0]
            N0 = cpu_cache.ni[f0]
            slot = sm.ensure_on_gpu(f0, pkg, N0)
            slot.wait_ready()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_ms = (time.time() - t0) * 1000.0

            # Render a tiny 64x64 view just to prime kernels
            if default_cam is not None:
                warm_cam_dict = _scale_cam_intrinsics(default_cam, 64, 64)
            else:
                warm_cam_dict = {
                    "width": 64,
                    "height": 64,
                    "fx": 32,
                    "fy": 32,
                    "cx": 32,
                    "cy": 32,
                    "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "T": [0, 0, 0],
                    "znear": 0.01,
                    "zfar": 10.0,
                }
            cam = build_minicam(warm_cam_dict)
            bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

            t1 = time.time()
            with torch.no_grad():
                _ = safe_render(
                    RENDER,
                    cam,
                    make_gaussian_from_slot(slot),
                    clone_pipeline(default_pipe),
                    bg,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            render_ms = (time.time() - t1) * 1000.0
            print(
                f"[Warmup] ok on frame {f0} | "
                f"H2D={load_ms:.1f} ms | render={render_ms:.1f} ms"
            )
        except Exception as e:
            print(f"[Warmup] failed: {e}")

    if warmup:
        do_warmup()

    # -------------------------------------------------------------------------
    # Background prefetch of neighbor frames (for smoother scrubbing)
    # -------------------------------------------------------------------------
    def prefetch_neighbors(center_f: int):
        if not neighbor_prefetch:
            return
        for nf in (center_f - 1, center_f + 1):
            if nf in cpu_cache.frames and not sm.has(nf):
                pkg = cpu_cache.data[nf]
                N = cpu_cache.ni[nf]
                threading.Thread(
                    target=lambda: sm.ensure_on_gpu(nf, pkg, N),
                    daemon=True,
                ).start()

    # -------------------------------------------------------------------------
    # HTTP endpoints
    # -------------------------------------------------------------------------
    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "frames": frames,
                "slots": slots,
                "N_dynamic_max": cpu_cache.N_max,
                "N_static": static_n,
                "N_max": total_N_max,
                "L": cpu_cache.L,
                "default_camera": default_cam,
            }
        )

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
        pkg = cpu_cache.data[frame_idx]
        N_i = cpu_cache.ni[frame_idx]
        slot = sm.ensure_on_gpu(frame_idx, pkg, N_i)
        slot.wait_ready()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_ms = (time.time() - t0) * 1000.0

        print(
            f"[Load]  frame={frame_idx} | cache={'hit' if cache_hit else 'miss'} | "
            f"H2D+wait={load_ms:.1f} ms | valid={slot.valid}/{slot.N_max}"
        )
        prefetch_neighbors(frame_idx)

        resp = jsonify(
            {
                "ok": True,
                "frame": frame_idx,
                "valid": slot.valid,
                "cache": ("hit" if cache_hit else "miss"),
            }
        )
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

        background = torch.tensor(
            ([1, 1, 1] if white_bg else [0, 0, 0]),
            dtype=torch.float32,
            device="cuda",
        )

        cache_hit = sm.has(frame_idx)

        # Ensure frame is on GPU (dynamic part)
        t_h2d0 = time.time()
        pkg = cpu_cache.data[frame_idx]
        N_i = cpu_cache.ni[frame_idx]
        sm.ensure_on_gpu(frame_idx, pkg, N_i)

        with sm.lease(frame_idx) as slot:
            slot.wait_ready()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            h2d_ms = (time.time() - t_h2d0) * 1000.0

            gauss = make_gaussian_from_slot(slot)
            print(
                f"[Render] frame={frame_idx} | cache={'hit' if cache_hit else 'miss'} | "
                f"valid={slot.valid}/{slot.N_max}"
            )

            # Render
            t_r0 = time.time()
            with torch.no_grad():
                out = safe_render(RENDER, cam, gauss, pipe, background)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            render_ms = (time.time() - t_r0) * 1000.0

            # Encode to JPEG/PNG
            t_e0 = time.time()
            img = (
                torch.clamp(out, 0, 1)
                * 255
            ).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            if fmt == "png":
                pil.save(buf, format="PNG")
                mime = "image/png"
            else:
                pil.save(
                    buf,
                    format="JPEG",
                    quality=jpeg_quality,
                    optimize=True,
                )
                mime = "image/jpeg"
            buf.seek(0)
            encode_ms = (time.time() - t_e0) * 1000.0

        prefetch_neighbors(frame_idx)

        total_ms = (time.time() - t_all0) * 1000.0
        print(
            f"[Render] done  frame={frame_idx} | "
            f"H2D={h2d_ms:.1f} ms | render={render_ms:.1f} ms | "
            f"encode={encode_ms:.1f} ms | total={total_ms:.1f} ms"
        )

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
    ap = argparse.ArgumentParser(
        "Time Archival 3DGS — Server (CPU preload + GPU slots)"
    )
    ap.add_argument(
        "-p",
        "--packed_root",
        required=True,
        type=str,
        help="Folder containing packed .pt frames (output of ta_pack.py)",
    )
    ap.add_argument(
        "--prefix",
        default="model_frame_",
        type=str,
        help="Frame prefix (e.g. model_frame_).",
    )
    ap.add_argument(
        "--slots",
        type=int,
        default=3,
        help="Number of GPU slots (GPU cache size).",
    )
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="Host to bind the HTTP server.",
    )
    ap.add_argument(
        "--port",
        default=7860,
        type=int,
        help="Port to bind the HTTP server.",
    )
    ap.add_argument(
        "--jpeg_quality",
        default=85,
        type=int,
        help="JPEG quality for /render when format == 'jpeg'.",
    )
    ap.add_argument(
        "--white_background",
        action="store_true",
        help="Use white background by default (otherwise black).",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="Do a warmup load+render on the first frame at startup.",
    )
    ap.add_argument(
        "--neighbor_prefetch",
        action="store_true",
        help="Background-preload neighbor frames (frame-1, frame+1).",
    )
    ap.add_argument(
        "--camera_json",
        default="camera.json",
        type=str,
        help="Path to default camera json (optional).",
    )
    ap.add_argument(
        "--static_pt",
        default=None,
        type=str,
        help=(
            "Optional packed .pt file for a static scene (A). "
            "If provided, those Gaussians are added to every frame."
        ),
    )

    args = ap.parse_args()

    app = create_app(
        Path(args.packed_root).resolve(),
        args.prefix,
        args.slots,
        args.jpeg_quality,
        args.white_background,
        args.warmup,
        args.neighbor_prefetch,
        Path(args.camera_json) if args.camera_json else None,
        Path(args.static_pt).resolve() if args.static_pt else None,
    )
    print(
        f"[TA-ServerSlots] Serving packed from {args.packed_root} | "
        f"slots={args.slots}"
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
