#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Archival 3DGS — Camera Sequencer (Prefetch + Warmup + Camera Track)

Features
--------
- Prefetch next 3DGS frame to GPU while rendering/saving the current frame
- Optional warmup render to compile CUDA kernels before the main loop
- Optional JPEG output for faster saving
- Supports:
    * Static camera from a single JSON file
    * Moving camera track from a JSON list (one entry per frame)
    * Static camera from COLMAP intrinsics/extrinsics
- Optional on-the-fly merge of:
    * Static A point cloud (stadium, background, etc.)
    * Per-frame B-only (players) point clouds
  with configurable feature alignment (--feature_align)

Example usages
--------------
1) Static camera from JSON, render all model_frame_* directories:
   python ta_render.py -m ./output_seq -o ./renders -c ./camera.json --frames all \
       --preload_depth 2 --warmup --save_format jpeg --jpeg_quality 90

2) Moving camera track from JSON list (exported from Blender, etc.):
   python ta_render.py -m ./output_seq -o ./renders -c ./camera_track.json --frames 1-300 \
       --preload_depth 2 --warmup --save_format jpeg --jpeg_quality 92

3) Static camera from COLMAP:
   python ta_render.py -m ./output_seq -o ./renders \
       --colmap_path ./dataset --image_name 0001.png --frames 1-50

4) On-the-fly A+B merge (static A + filtered B-only), with feature alignment:
   python ta_render.py \
       -m ./output_seq/soccer_B_60cams_FILTERED \
       -o ./renders_AB \
       -c ./camera_track.json \
       --frames 1-60 \
       --static_a_ply Static_Point_Cloud/70cams_A_point_cloud.ply \
       --feature_align pad \
       --preload_depth 2 --warmup --save_format jpeg --jpeg_quality 92
"""
import argparse, json, math, os, sys, time, threading, queue, tempfile
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
from ta_common import parse_frames, find_latest_iteration, load_cam_from_colmap, align_features_for_merge

try:
    # Reuse PLY helpers from merge_A_B_batch so that layout is consistent
    from merge_A_B_batch import read_ply_xyzcso, write_ply_xyzcso
    HAS_MERGE_HELPERS = True
except Exception:
    HAS_MERGE_HELPERS = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


# --------------------------------------------------------------------------
# Camera utilities
# --------------------------------------------------------------------------


def fov_from_fx(fx, w):
    """Convert focal length fx (in pixels) to horizontal FoV (radians)."""
    return 2.0 * math.atan(w / (2.0 * fx))


def build_view_from_cam_dict(C: dict) -> MiniCam:
    """
    Build a MiniCam from a dictionary of camera parameters.

    Supported keys (typical):
        width, height : image size in pixels
        znear, zfar   : near/far planes (optional, defaults used if missing)
        Either:
          - fx, fy, cx, cy (pinhole intrinsics in pixels)
        Or:
          - FoVx, FoVy (field of view in radians)
        R : 3x3 rotation matrix
        T : translation vector (3 elements)

    IMPORTANT CONVENTION (MATCHING COLMAP & camera_track.json):
        We assume the JSON stores a **world->camera** extrinsic:
            X_cam = R_wc * X_world + T_wc

        The original 3DGS utility getWorld2View2() expects:
            R_cw : camera->world rotation
            T_wc : world->camera translation (COLMAP's tvec)

        COLMAP path does:
            R_cw = qvec2rotmat(...).T
            T_wc = tvec
            world_view = getWorld2View2(R_cw, T_wc, ...)

        So here we:
            1) read R_wc, T_wc from JSON
            2) convert to R_cw = R_wc^T
            3) pass (R_cw, T_wc) into getWorld2View2

        This makes ta_render consistent with:
            - COLMAP cameras
            - ply_visualizer.py (which visualizes camera centers with
              C = -R_wc^T * T_wc)
    """
    W, H = int(C["width"]), int(C["height"])
    znear = float(C.get("znear", 0.01))
    zfar = float(C.get("zfar", 100.0))

    # Intrinsics
    if all(k in C for k in ["fx", "fy", "cx", "cy"]):
        fx, fy = float(C["fx"]), float(C["fy"])
        fovx = fov_from_fx(fx, W)
        fovy = fov_from_fx(fy, H)
    else:
        # Fall back to FoV-based specification
        fovx = float(C["FoVx"])
        fovy = float(C["FoVy"])

    # ---- Extrinsics: JSON is world->cam (R_wc, T_wc) ----
    R_wc = np.asarray(C["R"], dtype=np.float32)
    T_wc = np.asarray(C["T"], dtype=np.float32)

    # Convert to camera->world rotation (R_cw) to match COLMAP path.
    R_cw = R_wc.transpose()  # (3,3)

    # For debugging: you can uncomment to print camera center
    # C_world = -R_cw @ T_wc  # same as -R_wc^T @ T_wc
    # print("[DEBUG] cam center:", C_world)

    world_view = torch.tensor(
        getWorld2View2(R_cw, T_wc, np.array([0, 0, 0], dtype=np.float32), 1.0),
        dtype=torch.float32,
    ).transpose(0, 1).cuda()

    proj = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).cuda()

    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    view = MiniCam(W, H, fovy, fovx, znear, zfar, world_view, full)
    setattr(view, "image_name", C.get("image_name", "custom"))
    return view


def load_cam_from_json(cam_json_path: Path) -> MiniCam:
    """
    Load a static camera from a JSON file that contains a single dict.
    This is kept for backward compatibility (non-track usage).
    """
    with open(cam_json_path, "r") as f:
        C = json.load(f)
    if not isinstance(C, dict):
        raise ValueError(
            f"Expected a single camera dict in {cam_json_path}, "
            f"but got type {type(C)}. For a track, pass a list and let "
            f"ta_render.py handle it as a camera track."
        )
    return build_view_from_cam_dict(C)


def load_camera_track(cam_json_path: Path):
    """
    Load a camera track from JSON.

    The JSON file is expected to be:
      - Either a list of dicts, each containing at least:
            "frame": int
            ... (other camera parameters compatible with build_view_from_cam_dict)
      - Or a single dict (then it is treated as a static camera, not a track)

    Returns:
        base_view : MiniCam for warmup / fallback
        cam_track : dict[int, dict] mapping frame index -> camera dict,
                    or None if this is a single static camera.
    """
    with open(cam_json_path, "r") as f:
        data = json.load(f)

    # Case 1: single static camera
    if isinstance(data, dict):
        base_view = build_view_from_cam_dict(data)
        cam_track = None
        print(f"[TA-Render] Loaded single static camera from JSON: {cam_json_path}")
        return base_view, cam_track

    # Case 2: list of per-frame cameras (track)
    if not isinstance(data, list):
        raise ValueError(
            f"Unsupported camera JSON format in {cam_json_path}. "
            f"Expected dict or list, got {type(data)}."
        )

    cam_track = {}
    for entry in data:
        if "frame" not in entry:
            raise KeyError(
                "Each camera track entry must contain a 'frame' field "
                "(integer frame index)."
            )
        frame_idx = int(entry["frame"])
        if frame_idx in cam_track:
            raise ValueError(
                f"Duplicate camera entry for frame {frame_idx} in {cam_json_path}."
            )
        cam_track[frame_idx] = entry

    if not cam_track:
        raise ValueError(f"Camera track JSON {cam_json_path} is empty.")

    first_frame = sorted(cam_track.keys())[0]
    base_view = build_view_from_cam_dict(cam_track[first_frame])
    print(
        f"[TA-Render] Loaded camera track from JSON: {cam_json_path} "
        f"(entries={len(cam_track)}, first_frame={first_frame})"
    )
    return base_view, cam_track


# --------------------------------------------------------------------------
# Model and frame utilities
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Feature alignment helper (for A+B merge)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Prefetch worker
# --------------------------------------------------------------------------


def prefetch_worker(
    frames,
    models_root,
    prefix,
    iteration,
    sh_degree,
    out_queue: queue.Queue,
    stop_flag,
    static_A=None,
    feature_align: str = "none",
):
    """
    Background worker that preloads Gaussian models to GPU.

    For each frame index i in `frames`, it pushes one of:

        ("ok", (i, it, gaussian_model, load_secs))
        ("skip", i, reason_str, 0.0)
        ("err", error_message_str)
        ("eof", None)   # signals the end of the stream

    If `static_A` is not None, models_root is assumed to contain B-only
    point clouds, and this worker will:
        1) Read the per-frame B PLY as numpy arrays
        2) Optionally align features between static_A and B
        3) Concatenate static_A + B along the point dimension
        4) Write a temporary merged PLY
        5) Load that merged PLY into GaussianModel
        6) Delete the temporary file
    """
    try:
        for i in frames:
            if stop_flag.get("stop"):
                break

            mp = models_root / f"{prefix}{i}"
            it = iteration if iteration >= 0 else find_latest_iteration(mp)
            if it < 0:
                out_queue.put(("skip", i, "no_iter", 0.0))
                continue

            ply = mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
            if not ply.exists():
                out_queue.put(("skip", i, "no_ply", 0.0))
                continue

            ply_to_load = ply
            tmp_path = None

            # Optional: on-the-fly merge static A + per-frame B-only
            if static_A is not None:
                if not HAS_MERGE_HELPERS:
                    out_queue.put(
                        ("err", "static_A provided but merge helpers not available")
                    )
                    break

                B = read_ply_xyzcso(str(ply))
                A_local = {k: v for k, v in static_A.items()}

                # Align feature dimensions if requested
                if feature_align is not None and feature_align != "none":
                    A_local, B = align_features_for_merge(A_local, B, feature_align)

                OUT = {
                    k: np.concatenate([A_local[k], B[k]], axis=0)
                    for k in ["xyz", "opacity", "scale", "rot", "f_dc", "f_rest"]
                }

                fd, tmp_path = tempfile.mkstemp(
                    prefix=f"ta_render_merge_{i:05d}_", suffix=".ply"
                )
                os.close(fd)
                write_ply_xyzcso(tmp_path, OUT)
                ply_to_load = Path(tmp_path)

            t0 = time.perf_counter()
            gauss = GaussianModel(sh_degree)
            gauss.load_ply(str(ply_to_load), use_train_test_exp=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Remove temp file if we created one
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            out_queue.put(("ok", (i, it, gauss, t1 - t0)))

    except Exception as e:
        out_queue.put(("err", str(e)))
    finally:
        out_queue.put(("eof", None))


# --------------------------------------------------------------------------
# Warmup helper
# --------------------------------------------------------------------------


def build_warmup_view_from(view: MiniCam, scale=0.25) -> MiniCam:
    """
    Build a low-resolution MiniCam for warmup purposes, preserving the same
    world_view transform and FoV.
    """
    W = max(32, int(view.image_width * scale))
    H = max(32, int(view.image_height * scale))
    world_view = view.world_view_transform
    proj = getProjectionMatrix(
        znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy
    ).transpose(0, 1).cuda()
    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(W, H, view.FoVy, view.FoVx, view.znear, view.zfar, world_view, full)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv):
    parser = argparse.ArgumentParser(
        description="Time Archival 3DGS — Sequencer (Prefetch, Camera Track Support)"
    )
    parser.add_argument("-m", "--models_root", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    # Camera
    parser.add_argument(
        "-c",
        "--camera_json",
        type=str,
        default=None,
        help=(
            "Camera JSON path. "
            "Can be either a single-camera dict or a list for a camera track."
        ),
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        default=None,
        help="Dataset root containing COLMAP 'sparse/{sparse_id}' folder.",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Image name for COLMAP camera lookup.",
    )
    parser.add_argument(
        "--sparse_id",
        type=int,
        default=0,
        help="COLMAP sparse reconstruction id (usually 0).",
    )
    parser.add_argument(
        "--znear",
        type=float,
        default=0.01,
        help="Near plane override for COLMAP camera.",
    )
    parser.add_argument(
        "--zfar",
        type=float,
        default=100.0,
        help="Far plane override for COLMAP camera.",
    )

    # Sequence / model
    parser.add_argument("--frames", type=str, default="all")
    parser.add_argument("--prefix", type=str, default="model_frame_")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--video_name", type=str, default="time_archival.mp4")
    parser.add_argument(
        "--static_a_ply",
        type=str,
        default=None,
        help=(
            "Optional static A point cloud PLY. If provided, ta_render will "
            "add this static A model to every per-frame model (assumed to be "
            "filtered B-only) by merging the point clouds on-the-fly before "
            "loading them into GaussianModel."
        ),
    )
    parser.add_argument(
        "--feature_align",
        type=str,
        choices=["none", "pad", "trim"],
        default="none",
        help=(
            "Feature alignment mode when merging static A and per-frame B-only "
            "point clouds:\n"
            "  none : assume f_rest dimensions already match\n"
            "  pad  : pad the smaller f_rest to match the larger one\n"
            "  trim : trim both to the smaller f_rest dimension"
        ),
    )

    # Prefetch / warmup / output format
    parser.add_argument(
        "--preload_depth",
        type=int,
        default=2,
        help="Max depth of prefetch queue (recommend 1–3).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up render kernels with a low-resolution camera once.",
    )
    parser.add_argument(
        "--save_format", choices=["png", "jpeg"], default="png", help="Output format."
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality (only used when save_format=jpeg).",
    )

    pp = PipelineParams(parser)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_root = Path(args.models_root)
    frames = parse_frames(args.frames, models_root)
    print(f"[TA-Render] Frames to render: {frames}")

    # Optional static A: will be added to every per-frame model
    static_A_data = None    # noqa: N806
    if args.static_a_ply is not None:
        if not HAS_MERGE_HELPERS:
            raise RuntimeError(
                "--static_a_ply was provided, but merge_A_B_batch helpers "
                "(read_ply_xyzcso/write_ply_xyzcso) could not be imported."
            )
        static_A_data = read_ply_xyzcso(args.static_a_ply)
        print(
            f"[TA-Render] Loaded static A PLY: {args.static_a_ply} | "
            f"N={static_A_data['xyz'].shape[0]}, "
            f"f_rest={static_A_data['f_rest'].shape[1]}"
        )

    # ------------------------------------------------------------------
    # Camera setup: either COLMAP, static JSON, or camera track JSON
    # ------------------------------------------------------------------
    cam_track = None  # dict[int, dict] if track mode is used
    base_view = None  # MiniCam used for static camera or warmup / fallback

    if args.colmap_path and args.image_name:
        print(
            f"[TA-Render] Using COLMAP camera from {args.colmap_path}, "
            f"image '{args.image_name}', sparse_id={args.sparse_id}"
        )
        base_view = load_cam_from_colmap(
            Path(args.colmap_path),
            args.image_name,
            args.sparse_id,
            args.znear,
            args.zfar,
        )
    elif args.camera_json:
        print(f"[TA-Render] Using camera JSON: {args.camera_json}")
        base_view, cam_track = load_camera_track(Path(args.camera_json))
        if cam_track is None:
            print("[TA-Render] Camera JSON is a single static camera (no track).")
        else:
            print(
                "[TA-Render] Camera JSON is a per-frame camera track. "
                "Per-frame cameras will be used if available."
            )
    else:
        raise ValueError(
            "Please specify either --camera_json or "
            "(--colmap_path AND --image_name)."
        )

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)

    # ------------------------------------------------------------------
    # Start prefetch worker
    # ------------------------------------------------------------------
    q_out = queue.Queue(maxsize=max(1, args.preload_depth))
    stop_flag = {"stop": False}
    worker = threading.Thread(
        target=prefetch_worker,
        args=(
            frames,
            models_root,
            args.prefix,
            args.iteration,
            args.sh_degree,
            q_out,
            stop_flag,
            static_A_data,
            args.feature_align,
        ),
        daemon=True,
    )
    worker.start()

    # ------------------------------------------------------------------
    # Optional warmup (will use the *first* camera view encountered)
    # ------------------------------------------------------------------
    got_first = False

    frame_paths = []
    total_start = time.perf_counter()
    sum_load = 0.0
    sum_render = 0.0
    sum_save = 0.0
    n_done = 0

    while True:
        tag, payload = q_out.get()
        if tag == "eof":
            break
        if tag == "skip":
            _, i, reason, _ = payload
            print(f"[TA-Render] SKIP frame {i}: {reason}")
            continue
        if tag == "err":
            print(f"[TA-Render][ERROR] prefetch: {payload}")
            continue
        if tag != "ok":
            continue

        i, it, gauss, load_secs = payload

        # Select camera for this frame:
        # - If cam_track is not None, use the per-frame camera (required)
        # - Otherwise, fall back to base_view (static camera)
        if cam_track is not None:
            if i not in cam_track:
                raise ValueError(
                    f"No camera entry found in track for frame {i}. "
                    f"Either add it to the track JSON or restrict --frames "
                    f"to the frames that exist in the camera track."
                )
            view = build_view_from_cam_dict(cam_track[i])
        else:
            view = base_view

        # Optional warmup: run once on the first loaded Gaussian model
        if not got_first and args.warmup:
            got_first = True
            try:
                warm_view = build_warmup_view_from(view, scale=0.25)
                with torch.no_grad():
                    _ = render(
                        warm_view,
                        gauss,
                        pipe,
                        background,
                        use_trained_exp=False,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("[TA-Render] Warmup done.")
            except Exception as e:
                print(f"[TA-Render] Warmup failed: {e}")

        # ------------------------------------------------------------------
        # Render
        # ------------------------------------------------------------------
        t1 = time.perf_counter()
        with torch.no_grad():
            out = render(
                view,
                gauss,
                pipe,
                background,
                use_trained_exp=False,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # ------------------------------------------------------------------
        # Save image
        # ------------------------------------------------------------------
        img_stem = f"{i:05d}"
        if args.save_format == "png":
            img_path = out_dir / f"{img_stem}.png"
            torchvision.utils.save_image(out, str(img_path))
        else:
            # JPEG: usually much faster to write
            img = (
                torch.clamp(out, 0, 1) * 255
            ).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            img_path = out_dir / f"{img_stem}.jpg"
            Image.fromarray(img).save(
                str(img_path),
                format="JPEG",
                quality=args.jpeg_quality,
                optimize=True,
            )
        t3 = time.perf_counter()

        # Timing statistics (load happens in the background worker)
        load_dt = load_secs
        render_dt = t2 - t1
        save_dt = t3 - t2
        sum_load += load_dt
        sum_render += render_dt
        sum_save += save_dt
        n_done += 1

        frame_paths.append(str(img_path))
        print(f"[TA-Render] Saved {img_path}")
        print(
            f"[TA-Render][Timing] frame {i}: "
            f"load(bg)={load_dt:.3f}s, render={render_dt:.3f}s, "
            f"save={save_dt:.3f}s, total~={(render_dt + save_dt):.3f}s"
        )

        # Explicitly release the Gaussian model for this frame
        del gauss
        torch.cuda.empty_cache()

    total_end = time.perf_counter()
    if n_done > 0:
        print(
            f"[TA-Render][Timing][Summary] frames={n_done} | "
            f"prefetch_load_sum={sum_load:.3f}s, render_sum={sum_render:.3f}s, "
            f"save_sum={sum_save:.3f}s, "
            f"end2end_no_video={(total_end - total_start):.3f}s | "
            f"avg_render={sum_render / n_done:.3f}s | "
            f"render_FPS={n_done / sum_render:.2f}"
        )

    # ----------------------------------------------------------------------
    # Optional video assembly (PNG only; for JPEG use ffmpeg separately)
    # ----------------------------------------------------------------------
    try:
        if frame_paths and args.save_format == "png":
            tvid0 = time.perf_counter()
            import imageio.v3 as iio

            imgs = [iio.imread(p) for p in frame_paths]
            video_out = out_dir / args.video_name
            iio.imwrite(str(video_out), imgs, fps=24)
            tvid1 = time.perf_counter()
            print(f"[TA-Render] Video written to {video_out}")
            print(
                f"[TA-Render][Timing] video_build={tvid1 - tvid0:.3f}s"
            )
        elif frame_paths and args.save_format == "jpeg":
            print("[TA-Render] You saved JPEGs; use ffmpeg to build video, e.g.:")
            print(
                "            ffmpeg -y -framerate 24 -i %05d.jpg "
                "-pix_fmt yuv420p time_archival.mp4"
            )
    except Exception as e:
        print(f"[TA-Render] Could not build video: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
