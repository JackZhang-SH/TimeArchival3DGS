#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ta_test.py — Render a model on its test images and compute metrics (PSNR/SSIM/LPIPS),
with detailed timing breakdown.

Now supports:
  - Evaluating either:
      * A standalone 3DGS model (default, as before), OR
      * On-the-fly merge of:
          static A PLY  +  per-frame B-only (filtered) PLY
    via --static_a_ply and --feature_align.

Usage examples:
  # 1) Classic mode: merged model_frame_* (no A/B splitting)
  python ta_test.py \
      -s ./dataset/soccer_dynamic_player_B \
      -m ./output_seq/soccer_merged \
      --frames 1-5 --prefix model_frame_ \
      --read_test_from_model_cfg --sparse_id 0 --iteration -1

  # 2) New mode: filtered B-only + static A (on-the-fly A+B merge for metrics)
  python ta_test.py \
      -s ./dataset/soccer_B_60cams \
      -m ./output_seq/soccer_B_60cams_FILTERED \
      --frames 1-5 --prefix model_frame_ \
      --static_a_ply Static_Point_Cloud/70cams_A_point_cloud.ply \
      --feature_align pad \
      --prefer_model_test_list \
      --sparse_id 0 --iteration -1
"""

import argparse, json, math, os, re, sys, time, tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from tqdm import tqdm

# 3DGS imports
from arguments import PipelineParams  # noqa: F401
from scene.gaussian_model import GaussianModel  # noqa: F401
from gaussian_renderer import render  # noqa: F401
from scene.cameras import MiniCam  # noqa: F401
from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                 read_intrinsics_binary, read_intrinsics_text,
                                 qvec2rotmat)  # noqa: F401
from utils.image_utils import psnr  # noqa: F401
from utils.loss_utils import ssim  # noqa: F401
from lpipsPyTorch import lpips  # noqa: F401
from ta_common import parse_frames, find_latest_iteration, load_cam_from_colmap, align_features_for_merge

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

# Optional PLY helpers (shared with merge_A_B_batch)
try:
    from merge_A_B_batch import read_ply_xyzcso, write_ply_xyzcso
    HAS_MERGE_HELPERS = True
except Exception:
    HAS_MERGE_HELPERS = False


# ----------------------------- Helpers (COLMAP I/O) -----------------------------
# ----------------------------- FS / parsing helpers -----------------------------
def _read_model_cfg_tests(model_dir: Path) -> Tuple[Optional[List[str]], Optional[str], bool]:
    """Parse train.py's cfg_args for test split hints. Returns (names, regex, clear_default)."""
    cfg = model_dir / "cfg_args"
    if not cfg.exists():
        return None, None, False
    s = cfg.read_text(encoding="utf-8")

    # Extract with regex and literal_eval safely
    import ast
    def _lit(m):
        try:
            return ast.literal_eval(m)
        except Exception:
            return None

    names = None
    m1 = re.search(r"test_images\s*=\s*(\[[^\]]*\])", s)
    if m1:
        names = _lit(m1.group(1))

    regex = None
    m2 = re.search(r"test_regex\s*=\s*([\"'].*?[\"'])", s, flags=re.DOTALL)
    if m2:
        regex = _lit(m2.group(1))

    clear_default = False
    m3 = re.search(r"test_clear_default\s*=\s*(True|False)", s)
    if m3:
        clear_default = (m3.group(1) == "True")

    return names, regex, clear_default


def _read_test_list_file(model_dir: Path) -> Optional[list]:
    """Read <model_dir>/test_images.txt if present; return list of names (strings) or None."""
    txt = model_dir / "test_images.txt"
    if txt.exists():
        names = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t:
                names.append(t)
        return names
    return None


def _find_merged_ply(model_dir: Path) -> Optional[Path]:
    """
    Try common merged-PLY locations if iteration_* is absent:
      - <model_dir>/point_cloud_merged/point_cloud.ply
      - <model_dir>/point_cloud_merged/point_cloud_merged.ply
      - <model_dir>/point_cloud/merged/point_cloud.ply
      - <model_dir>/point_cloud_merged.ply (flat copy)
    """
    cands = [
        model_dir / "point_cloud_merged" / "point_cloud.ply",
        model_dir / "point_cloud_merged" / "point_cloud_merged.ply",
        model_dir / "point_cloud" / "merged" / "point_cloud.ply",
        model_dir / "point_cloud_merged.ply",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def _resolve_ply(model_dir: Path, iteration: int) -> Path:
    """
    Resolve a model PLY for the "classic" case:
      - Prefer point_cloud/iteration_X/point_cloud.ply
      - Fallback to merged PLY locations
    Used when we are NOT doing on-the-fly A+B merging.
    """
    it = iteration if iteration >= 0 else find_latest_iteration(model_dir)
    if it >= 0:
        ply = model_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            return ply
    m = _find_merged_ply(model_dir)
    if m is not None:
        return m
    raise FileNotFoundError(f"No PLY found under {model_dir} (iteration or merged fallback)")


def _resolve_b_ply(model_dir: Path, iteration: int) -> Path:
    """
    Resolve B-only PLY in filtered-B mode:
      - Require point_cloud/iteration_X/point_cloud.ply
      - Do NOT fall back to any merged PLY.
    """
    it = iteration if iteration >= 0 else find_latest_iteration(model_dir)
    if it < 0:
        raise FileNotFoundError(f"No iteration_* under {model_dir}/point_cloud")
    ply = model_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"B-only PLY not found at {ply}")
    return ply


def _choose_test_names(gt_images_dir: Path,
                       prefer_model_test_list: bool,
                       model_dir_for_list: Path,
                       cli_names: Optional[Sequence[str]],
                       cli_regex: Optional[str],
                       cli_list_txt: Optional[Path],
                       read_from_cfg: bool,
                       model_dir: Path) -> List[str]:
    # base pool
    pool = sorted([p.name for p in gt_images_dir.glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    stems = {Path(n).stem: n for n in pool}

    names: List[str] = []

    # 0) Prefer model_dir/test_images.txt if asked and present
    if prefer_model_test_list and model_dir_for_list.exists():
        lst = _read_test_list_file(model_dir_for_list)
        if lst:
            for t in lst:
                k = stems.get(Path(t).stem, None)
                if k and (k not in names):
                    names.append(k)

    # 1) CLI list file
    if cli_list_txt and Path(cli_list_txt).exists():
        for line in Path(cli_list_txt).read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if not t:
                continue
            k = stems.get(Path(t).stem, None)
            if k:
                names.append(k)

    # 2) CLI explicit names
    if cli_names:
        for t in cli_names:
            k = stems.get(Path(t).stem, None)
            if k and (k not in names):
                names.append(k)

    # 3) CLI regex
    if cli_regex:
        try:
            r = re.compile(cli_regex)
            for n in pool:
                if r.search(n) and (n not in names):
                    names.append(n)
        except re.error:
            print(f"[warn] invalid --test_regex ignored: {cli_regex}")

    # 4) From model cfg (if asked)
    if read_from_cfg and (not names) and model_dir.exists():
        cfg_names, cfg_regex, cfg_clear = _read_model_cfg_tests(model_dir)
        if cfg_names:
            for t in cfg_names:
                k = stems.get(Path(t).stem, None)
                if k and (k not in names):
                    names.append(k)
        if cfg_regex:
            try:
                r = re.compile(cfg_regex)
                for n in pool:
                    if r.search(n) and (n not in names):
                        names.append(n)
            except re.error:
                print(f"[warn] invalid test_regex in cfg ignored: {cfg_regex}")
        if cfg_clear and not names:
            # User explicitly cleared default set but didn't give anything:
            # keep empty -> we'll warn below.
            pass

    # 5) Fallback: if still empty, use all images
    if not names:
        print("[info] No test list provided/found; defaulting to ALL images in this frame.")
        names = pool

    return names


# ------------------------- Feature alignment helper -------------------------
# ------------------------- Visualization helpers -------------------------
def _tensor_to_pil_chw01(t: torch.Tensor) -> Image.Image:
    """
    t: torch.float32, shape (C,H,W), range [0,1]
    return PIL.Image (RGB)
    """
    t = t.detach().cpu().clamp(0, 1)
    return TF.to_pil_image(t)


def _save_side_by_side(pil_left: Image.Image, pil_right: Image.Image, out_path: Path):
    w = pil_left.width + pil_right.width
    h = max(pil_left.height, pil_right.height)
    canvas = Image.new("RGB", (w, h))
    canvas.paste(pil_left, (0, 0))
    canvas.paste(pil_right, (pil_left.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


# --------------------------------- Core testing --------------------------------
def test_one_frame(frame_idx: int,
                   gt_frame_dir: Path,
                   model_dir: Path,
                   iteration: int,
                   sparse_id: int,
                   pipe, sh_degree: int,
                   white_background: bool,
                   test_names: Sequence[str],
                   lpips_net: str = "vgg",
                   znear: float = 0.01, zfar: float = 100.0,
                   limit: Optional[int] = None,
                   vis_dir: Optional[Path] = None,
                   static_A: Optional[dict] = None,
                   feature_align: str = "none") -> Dict[str, float]:
    """
    Run evaluation on a single frame and return metrics + timing info.

    Two modes:
      - If static_A is None   : load a single 3DGS model from model_dir (classic)
      - If static_A is not None:
            treat model_dir as holding B-only PLY; on-the-fly merge static_A+B
            into a temporary PLY, load that merged model, and evaluate.

    Timing breakdown (per frame):
      - model_load_time: PLY resolve + GaussianModel + load_ply
      - colmap_time_total: sum of load_cam_from_colmap over all views
      - render_time_total: sum of pure render() time (with CUDA sync)
      - metrics_time_total: GT I/O + resize + PSNR/SSIM/LPIPS (with CUDA sync)
    """
    # ------------------------ Model loading timing ------------------------
    t_model0 = time.perf_counter()

    if static_A is None:
        # Classic mode: use iteration / merged PLY resolution
        ply = _resolve_ply(model_dir, iteration)
        gauss = GaussianModel(sh_degree)
        gauss.load_ply(str(ply), use_train_test_exp=False)
        print(f"[frame {frame_idx}] Model loaded from {ply}")
    else:
        # Filtered B-only + static A : on-the-fly A+B merge
        if not HAS_MERGE_HELPERS:
            raise RuntimeError(
                "static_A was provided, but merge_A_B_batch helpers "
                "(read_ply_xyzcso/write_ply_xyzcso) are not available."
            )

        ply_b = _resolve_b_ply(model_dir, iteration)
        print(f"[frame {frame_idx}] B-only model loaded from {ply_b}")
        B = read_ply_xyzcso(str(ply_b))
        A_local = {k: v for k, v in static_A.items()}

        if feature_align is not None and feature_align != "none":
            A_local, B = align_features_for_merge(A_local, B, feature_align)

        OUT = {
            k: np.concatenate([A_local[k], B[k]], axis=0)
            for k in ["xyz", "opacity", "scale", "rot", "f_dc", "f_rest"]
        }

        fd, tmp_path = tempfile.mkstemp(
            prefix=f"ta_test_merge_{frame_idx:05d}_", suffix=".ply"
        )
        os.close(fd)
        write_ply_xyzcso(tmp_path, OUT)

        gauss = GaussianModel(sh_degree)
        gauss.load_ply(tmp_path, use_train_test_exp=False)

        # Remove temporary PLY
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        print(
            f"[frame {frame_idx}] On-the-fly merged A+B "
            f"(A={A_local['xyz'].shape[0]}, B={B['xyz'].shape[0]}, "
            f"total={OUT['xyz'].shape[0]})"
        )

    t_model1 = time.perf_counter()
    model_load_time = t_model1 - t_model0
    print(f"[frame {frame_idx}] Model load time: {model_load_time:.4f} s")

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Metric accumulators
    ssims: List[float] = []
    psnrs: List[float] = []
    lpipss: List[float] = []

    # Timing accumulators
    colmap_time_total = 0.0
    render_time_total = 0.0
    metrics_time_total = 0.0

    images_dir = gt_frame_dir / "images"

    # Optionally limit number of views
    names_iter = list(test_names) if limit is None else list(test_names)[:int(limit)]

    with torch.no_grad():
        for name in tqdm(names_iter, desc=f"[frame {frame_idx}] Metric eval"):
            # -------------------- COLMAP camera timing --------------------
            t_cam0 = time.perf_counter()
            view = load_cam_from_colmap(gt_frame_dir, name, sparse_id, znear, zfar)
            t_cam1 = time.perf_counter()
            dt_cam = t_cam1 - t_cam0
            colmap_time_total += dt_cam

            # ------------------------ Render timing -----------------------
            torch.cuda.synchronize()
            t_render0 = time.perf_counter()
            out = render(
                view, gauss, pipe, background,
                use_trained_exp=False,
                separate_sh=SPARSE_ADAM_AVAILABLE
            )["render"]
            torch.cuda.synchronize()
            t_render1 = time.perf_counter()
            dt_render = t_render1 - t_render0
            render_time_total += dt_render

            # ------------------------ Metrics timing ----------------------
            t_metric0 = time.perf_counter()

            # Load GT
            gt_path = images_dir / name
            gt_img = Image.open(str(gt_path))

            # out is CHW CUDA tensor in [0,1]
            t_out = out.detach().clamp(0, 1).cpu().to(torch.float32)   # (C,H,W)
            # GT as RGB float tensor in [0,1]
            t_gt = TF.to_tensor(gt_img).to(torch.float32)[:3, :, :]

            # Ensure same size
            if t_out.shape[-2:] != t_gt.shape[-2:]:
                t_gt = TF.resize(t_gt, t_out.shape[-2:], antialias=True)

            # Metrics expect batched tensors on CUDA
            b_out = t_out.unsqueeze(0).cuda()
            b_gt = t_gt.unsqueeze(0).cuda()
            ssims.append(float(ssim(b_out, b_gt)))
            psnrs.append(float(psnr(b_out, b_gt)))
            lpipss.append(float(lpips(b_out, b_gt, net_type=lpips_net)))

            torch.cuda.synchronize()
            t_metric1 = time.perf_counter()
            dt_metric = t_metric1 - t_metric0
            metrics_time_total += dt_metric

            # ---------------------- Per-view debug print ------------------
            print(
                f"[frame {frame_idx}] view={name} | "
                f"COLMAP={dt_cam:.4f}s, render={dt_render:.4f}s, metrics={dt_metric:.4f}s"
            )

            # Visualization (optional)
            if vis_dir is not None:
                vis_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(name).stem
                pil_out = _tensor_to_pil_chw01(t_out)
                pil_gt = _tensor_to_pil_chw01(t_gt)
                # Single images
                pil_out.save(str(vis_dir / f"{stem}_render.png"))
                pil_gt.save(str(vis_dir / f"{stem}_gt.png"))
                # Side-by-side (left = GT, right = render)
                _save_side_by_side(pil_gt, pil_out, vis_dir / f"{stem}_side.png")

    # Free model memory
    del gauss
    torch.cuda.empty_cache()

    num_views = len(names_iter)
    avg_colmap = colmap_time_total / num_views if num_views > 0 else 0.0
    avg_render = render_time_total / num_views if num_views > 0 else 0.0
    avg_metrics = metrics_time_total / num_views if num_views > 0 else 0.0

    print(f"\n[frame {frame_idx}] Timing summary (views={num_views}):")
    print(f"  Model load      : {model_load_time:.4f} s")
    print(f"  COLMAP cameras  : {colmap_time_total:.4f} s "
          f"(avg {avg_colmap:.4f} s/view)")
    print(f"  Rendering       : {render_time_total:.4f} s "
          f"(avg {avg_render:.4f} s/view)")
    print(f"  Metrics + GT I/O: {metrics_time_total:.4f} s "
          f"(avg {avg_metrics:.4f} s/view)\n")

    # Aggregate metrics
    res = {
        "count": num_views,
        "SSIM": float(np.mean(ssims)) if ssims else 0.0,
        "PSNR": float(np.mean(psnrs)) if psnrs else 0.0,
        "LPIPS": float(np.mean(lpipss)) if lpipss else 0.0,
        "timing": {
            "model_load_sec": model_load_time,
            "colmap_sec": colmap_time_total,
            "render_sec": render_time_total,
            "metrics_sec": metrics_time_total,
        },
    }
    return res


def main(argv=None):
    parser = argparse.ArgumentParser("ta_test.py — Evaluate frames' test images against GT (with timing)")
    parser.add_argument("-s", "--gt_root", required=True, type=str,
                        help="GT dataset root (with frame_* subfolders)")
    parser.add_argument("-m", "--models_root", required=True, type=str,
                        help="Models root (with model_frame_* subfolders)")
    parser.add_argument("--frames", type=str, default="all",
                        help="e.g., all | 1-5 | 1,3,7")
    parser.add_argument("--prefix", type=str, default="model_frame_")
    parser.add_argument("--per_frame_subdir", type=str, default=None,
                        help="If your models are under per-frame subdir (e.g., point_cloud_merged)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="-1 = auto-detect latest iteration_*")
    parser.add_argument("--sparse_id", type=int, default=0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--lpips_net", choices=["vgg", "alex", "squeeze"], default="vgg")
    parser.add_argument("--prefer_model_test_list", action="store_true",
                        help="Use <model_dir>/test_images.txt when present.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally limit views per frame for a quick pass")

    # Test split controls (CLI)
    parser.add_argument("--test_images", nargs="+", default=None,
                        help="Explicit test images (names or stems)")
    parser.add_argument("--test_regex", type=str, default=None,
                        help="Regex on file name to select test images")
    parser.add_argument("--test_list_txt", type=str, default=None,
                        help="A text file listing test images (one per line)")
    parser.add_argument("--read_test_from_model_cfg", action="store_true",
                        help="Try to read test split from each model's cfg_args")

    # A+B merge controls (new)
    parser.add_argument(
        "--static_a_ply",
        type=str,
        default=None,
        help=(
            "Optional static A point cloud PLY. If provided, ta_test will "
            "assume models_root contains per-frame B-only (filtered) models, "
            "and will on-the-fly merge static A + B for evaluation."
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

    # Output
    parser.add_argument("--save_json", type=str, default=None,
                        help="Write full results to this JSON path")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save GT/render/side-by-side images to a test folder for visual inspection")
    parser.add_argument("--vis_root", type=str, default=None,
                        help="Root dir to write visual outputs; defaults to <models_root>/_test_vis")

    # Allow pipeline params to be passed through
    pp = PipelineParams(parser)
    args = parser.parse_args(argv)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    gt_root = Path(args.gt_root).resolve()
    models_root = Path(args.models_root).resolve()

    frames = parse_frames(args.frames, models_root, args.prefix)
    if not frames:
        print("[ERROR] No frames matched under models_root with given --frames / --prefix")
        sys.exit(1)

    # Extract pipeline
    pipe = pp.extract(args)

    # Visualization root
    vis_root = Path(args.vis_root).resolve() if (args.save_vis and args.vis_root) else (
        (models_root / "_test_vis") if args.save_vis else None
    )

    # Optional static A (for filtered-B + A configuration)
    static_A_data = None
    if args.static_a_ply is not None:
        if not HAS_MERGE_HELPERS:
            raise RuntimeError(
                "--static_a_ply was provided, but merge_A_B_batch helpers "
                "(read_ply_xyzcso/write_ply_xyzcso) could not be imported."
            )
        static_A_data = read_ply_xyzcso(args.static_a_ply)
        print(
            f"[ta_test] Loaded static A PLY: {args.static_a_ply} | "
            f"N={static_A_data['xyz'].shape[0]}, "
            f"f_rest={static_A_data['f_rest'].shape[1]}"
        )

    t0 = time.perf_counter()
    all_results: Dict[int, Dict[str, float]] = {}
    global_ssim_sum = global_psnr_sum = global_lpips_sum = 0.0
    global_count = 0

    # Global timing accumulators
    total_model_load = 0.0
    total_colmap = 0.0
    total_render = 0.0
    total_metrics = 0.0

    for i in frames:
        gt_frame_dir = gt_root / f"frame_{i}"
        if not gt_frame_dir.exists():
            print(f"[skip] GT folder missing for frame {i}: {gt_frame_dir}")
            continue

        model_dir = models_root / f"{args.prefix}{i}"
        if args.per_frame_subdir:
            model_dir = model_dir / args.per_frame_subdir
        if not model_dir.exists():
            print(f"[skip] Model folder missing for frame {i}: {model_dir}")
            continue

        # Decide test names for this frame
        names = _choose_test_names(
            gt_frame_dir / "images",
            args.prefer_model_test_list,
            model_dir,
            args.test_images,
            args.test_regex,
            Path(args.test_list_txt) if args.test_list_txt else None,
            args.read_test_from_model_cfg,
            model_dir,
        )
        if not names:
            print(f"[warn] No test images for frame {i}; skipping.")
            continue

        # Per-frame visualization dir: <vis_root>/<prefix><i>
        vis_dir = None
        if vis_root is not None:
            vis_dir = vis_root / f"{args.prefix}{i}"

        try:
            res = test_one_frame(
                i, gt_frame_dir, model_dir, args.iteration, args.sparse_id,
                pipe, args.sh_degree, args.white_background, names,
                lpips_net=args.lpips_net, limit=args.limit, vis_dir=vis_dir,
                static_A=static_A_data, feature_align=args.feature_align
            )
        except Exception as e:
            print(f"[frame {i}][ERROR] {e}")
            continue

        all_results[i] = res
        global_ssim_sum += res["SSIM"] * res["count"]
        global_psnr_sum += res["PSNR"] * res["count"]
        global_lpips_sum += res["LPIPS"] * res["count"]
        global_count += res["count"]

        timing = res.get("timing", {})
        total_model_load += float(timing.get("model_load_sec", 0.0))
        total_colmap += float(timing.get("colmap_sec", 0.0))
        total_render += float(timing.get("render_sec", 0.0))
        total_metrics += float(timing.get("metrics_sec", 0.0))

        print(
            f"\n[frame {i}] views={res['count']}  "
            f"SSIM={res['SSIM']:.6f}  PSNR={res['PSNR']:.6f}  LPIPS={res['LPIPS']:.6f}"
        )

    dt = time.perf_counter() - t0

    if global_count > 0:
        g_ssim = global_ssim_sum / global_count
        g_psnr = global_psnr_sum / global_count
        g_lpips = global_lpips_sum / global_count
        print("\n================= [TOTAL METRICS] =================")
        print(f"Total views = {global_count}")
        print(f"SSIM  = {g_ssim:.7f}")
        print(f"PSNR  = {g_psnr:.7f}")
        print(f"LPIPS = {g_lpips:.7f}")

        print("\n================= [TOTAL TIMING] ==================")
        print(f"Total wall time          : {dt:.3f} s")
        print(f"  Model loading (sum)    : {total_model_load:.3f} s")
        print(f"  COLMAP cameras (sum)   : {total_colmap:.3f} s")
        print(f"  Rendering (sum)        : {total_render:.3f} s")
        print(f"  Metrics + GT I/O (sum) : {total_metrics:.3f} s")
        accounted = total_model_load + total_colmap + total_render + total_metrics
        print(f"  Other overhead (approx): {max(0.0, dt - accounted):.3f} s")
        print(f"\nPer-view averages (over all frames/views):")
        print(f"  Render only            : {total_render / global_count:.5f} s/view")
        print(f"  COLMAP + render        : {(total_colmap + total_render) / global_count:.5f} s/view")
        print(f"  Full pipeline          : {dt / global_count:.5f} s/view")

        global_result = {
            "count": global_count,
            "SSIM": g_ssim,
            "PSNR": g_psnr,
            "LPIPS": g_lpips,
            "time_sec": dt,
            "timing_breakdown": {
                "model_load_sec": total_model_load,
                "colmap_sec": total_colmap,
                "render_sec": total_render,
                "metrics_sec": total_metrics,
                "other_overhead_sec": max(0.0, dt - accounted),
            },
        }
    else:
        print("\n[WARN] No frames evaluated.")
        global_result = {
            "count": 0,
            "SSIM": 0.0,
            "PSNR": 0.0,
            "LPIPS": 0.0,
            "time_sec": dt,
            "timing_breakdown": {},
        }

    if args.save_json:
        out = {"frames": all_results, "global": global_result}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[write] JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
