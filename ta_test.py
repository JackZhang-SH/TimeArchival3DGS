
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ta_test.py — Render a model on its test images and compute metrics (PSNR/SSIM/LPIPS).

Usage example:
  python ta_test.py -s ./dataset/soccer_dynamic_player_B --models_root ./output_seq \
      --frames 1-5 --prefix model_frame_ --read_test_from_model_cfg \
      --sparse_id 0 --iteration -1

You can also explicitly set a test split:
  # by names (exact file names or stems without extension)
  python ta_test.py -s ./dataset/... -m ./output_seq --frames 3 \
      --test_images 0001.png 0005.png 0012.png

  # by regex (Python re applied to filename)
  python ta_test.py -s ./dataset/... -m ./output_seq --frames all \
      --test_regex "^(000[5-9]|001[0-2])\.png$"

  # by list file (one name per line)
  python ta_test.py -s ./dataset/... -m ./output_seq --frames 1 \
      --test_list_txt ./test_images.txt

Notes:
- Ground-truth datasets are expected in COLMAP format per frame:
    <gt_root>/frame_<N>/{images/, sparse/0/{cameras,images}.(txt|bin)}
- Model folders are expected as:
    <models_root>/<prefix><N>/point_cloud/iteration_*/point_cloud.ply
- If --read_test_from_model_cfg is set, the script will try to parse
  each model's 'cfg_args' written by train.py to recover --test_images / --test_regex.
- Metrics are printed per frame and for the whole set (weighted by the number of views).
"""
import argparse, json, math, os, re, sys, time
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

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


# ----------------------------- Helpers (COLMAP I/O) -----------------------------
def _try_read_colmap_extrinsics(path_bin: Path, path_txt: Path):
    try:
        return read_extrinsics_binary(str(path_bin)), True
    except Exception:
        return read_extrinsics_text(str(path_txt)), False


def _try_read_colmap_intrinsics(path_bin: Path, path_txt: Path):
    try:
        return read_intrinsics_binary(str(path_bin)), True
    except Exception:
        return read_intrinsics_text(str(path_txt)), False


def load_cam_from_colmap(dataset_root: Path, image_name: str, sparse_id: int = 0,
                         znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
    sp = dataset_root / "sparse" / str(sparse_id)
    if not sp.exists():
        raise FileNotFoundError(f"COLMAP sparse folder not found: {sp}")

    images_bin = sp / "images.bin"
    images_txt = sp / "images.txt"
    cameras_bin = sp / "cameras.bin"
    cameras_txt = sp / "cameras.txt"

    extr_map, _ = _try_read_colmap_extrinsics(images_bin, images_txt)
    intr_map, _ = _try_read_colmap_intrinsics(cameras_bin, cameras_txt)

    # Match by full path or basename
    target_key = None
    for k, extr in extr_map.items():
        if extr.name == image_name:
            target_key = k
            break
    if target_key is None:
        bn = os.path.basename(image_name)
        for k, extr in extr_map.items():
            if os.path.basename(extr.name) == bn:
                target_key = k
                break
    if target_key is None:
        raise KeyError(f"Image named '{image_name}' not found in COLMAP images.")

    extr = extr_map[target_key]
    intr = intr_map[extr.camera_id]

    width, height = int(intr.width), int(intr.height)
    model = intr.model
    if model == "SIMPLE_PINHOLE":
        fx = float(intr.params[0]); fy = fx
    elif model == "PINHOLE":
        fx = float(intr.params[0]); fy = float(intr.params[1])
    else:
        raise AssertionError(f"Unsupported COLMAP camera model: {model}. Use PINHOLE/SIMPLE_PINHOLE.")

    # Convert to FoV
    def focal2fov(f, w): return 2.0 * math.atan(w / (2.0 * f))
    fovx = focal2fov(fx, width); fovy = focal2fov(fy, height)

    R = np.transpose(qvec2rotmat(extr.qvec)).astype(np.float32)
    T = np.asarray(extr.tvec, dtype=np.float32)

    # Assemble MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # lazy import
    world_view = torch.tensor(getWorld2View2(R, T, np.array([0, 0, 0], dtype=np.float32), 1.0)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    view = MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full)
    setattr(view, "image_name", extr.name)
    return view


# ----------------------------- FS / parsing helpers -----------------------------
def parse_frames(frames_arg: str, models_root: Path, prefix: str) -> List[int]:
    if frames_arg == "all":
        frames: List[int] = []
        for d in sorted(models_root.glob(f"{prefix}*")):
            if not d.is_dir():
                continue
            try:
                frames.append(int(d.name.split("_")[-1]))
            except Exception:
                pass
        return frames
    out: List[int] = []
    for part in frames_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    # dedup & sort
    return sorted(list(dict.fromkeys(out)))


def find_latest_iteration(model_path: Path) -> int:
    p = model_path / "point_cloud"
    if not p.exists():
        return -1
    iters = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                iters.append(int(d.name.split("_")[1]))
            except Exception:
                pass
    return max(iters) if iters else -1


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
    """
    cands = [
        model_dir / "point_cloud_merged" / "point_cloud.ply",
        model_dir / "point_cloud_merged" / "point_cloud_merged.ply",
        model_dir / "point_cloud" / "merged" / "point_cloud.ply",
    ]
    for c in cands:
        if c.exists():
            return c
    return None

def _choose_test_names(gt_images_dir: Path,
                       prefer_model_test_list: bool,
                       model_dir_for_list: Path,
                       
                       cli_names: Optional[Sequence[str]],
                       cli_regex: Optional[str],
                       cli_list_txt: Optional[Path],
                       read_from_cfg: bool,
                       model_dir: Path) -> List[str]:
    # base pool
    pool = sorted([p.name for p in gt_images_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
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
            # User explicitly cleared default set but didn't give anything: keep empty -> we'll warn below.
            pass

    # 5) Fallback: if still empty, use all images
    if not names:
        print("[info] No test list provided/found; defaulting to ALL images in this frame.")
        names = pool

    return names


# --------------------------------- Core testing --------------------------------
def _resolve_ply(model_dir: Path, iteration: int) -> Path:
    it = iteration if iteration >= 0 else find_latest_iteration(model_dir)
    if it >= 0:
        ply = model_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            return ply
    m = _find_merged_ply(model_dir)
    if m is not None:
        return m
    raise FileNotFoundError(f"No PLY found under {model_dir} (iteration or merged fallback)")


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
                   limit: Optional[int] = None) -> Dict[str, float]:
    # Prepare model
    ply = _resolve_ply(model_dir, iteration)

    gauss = GaussianModel(sh_degree)
    gauss.load_ply(str(ply), use_train_test_exp=False)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Metric accumulators
    ssims: List[float] = []
    psnrs: List[float] = []
    lpipss: List[float] = []

    images_dir = gt_frame_dir / "images"

    # Optionally limit number of views
    names_iter = list(test_names) if limit is None else list(test_names)[:int(limit)]

    with torch.no_grad():
        for name in tqdm(names_iter, desc=f"[frame {frame_idx}] Metric eval"):
            # Camera
            view = load_cam_from_colmap(gt_frame_dir, name, sparse_id, znear, zfar)

            # Render
            out = render(view, gauss, pipe, background, use_trained_exp=False,
                         separate_sh=SPARSE_ADAM_AVAILABLE)["render"]

            # Load GT
            gt_path = images_dir / name
            gt_img = Image.open(str(gt_path))
            t_out = TF.to_tensor(out.detach().clamp(0, 1).cpu())  # (C,H,W)
            t_gt  = TF.to_tensor(gt_img)[:3, :, :]                # strip alpha

            # Size sanity (should already match via COLMAP intrinsics)
            if t_out.shape[-2:] != t_gt.shape[-2:]:
                # Resize GT to render (rare)
                t_gt = TF.resize(t_gt, t_out.shape[-2:], antialias=True)

            # Metrics expect batched tensors on CUDA
            b_out = t_out.unsqueeze(0).cuda()
            b_gt  = t_gt.unsqueeze(0).cuda()
            ssims.append(float(ssim(b_out, b_gt)))
            psnrs.append(float(psnr(b_out, b_gt)))
            lpipss.append(float(lpips(b_out, b_gt, net_type=lpips_net)))

    # Free model memory
    del gauss
    torch.cuda.empty_cache()

    # Aggregate
    res = {
        "count": len(names_iter),
        "SSIM": float(np.mean(ssims)) if ssims else 0.0,
        "PSNR": float(np.mean(psnrs)) if psnrs else 0.0,
        "LPIPS": float(np.mean(lpipss)) if lpipss else 0.0,
    }
    return res


def main(argv=None):
    parser = argparse.ArgumentParser("ta_test.py — Evaluate frames' test images against GT")
    parser.add_argument("-s", "--gt_root", required=True, type=str, help="GT dataset root (with frame_* subfolders)")
    parser.add_argument("-m", "--models_root", required=True, type=str, help="Models root (with model_frame_* subfolders)")
    parser.add_argument("--frames", type=str, default="all", help="e.g., all | 1-5 | 1,3,7")
    parser.add_argument("--prefix", type=str, default="model_frame_")
    parser.add_argument("--per_frame_subdir", type=str, default=None, help="If your models are under per-frame subdir (e.g., point_cloud_merged)")
    parser.add_argument("--iteration", type=int, default=-1, help="-1 = auto-detect latest iteration_*")
    parser.add_argument("--sparse_id", type=int, default=0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--lpips_net", choices=["vgg", "alex", "squeeze"], default="vgg")
    parser.add_argument("--prefer_model_test_list", action="store_true", help="Use <model_dir>/test_images.txt when present.")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit views per frame for a quick pass")

    # Test split controls (CLI)
    parser.add_argument("--test_images", nargs="+", default=None, help="Explicit test images (names or stems)")
    parser.add_argument("--test_regex", type=str, default=None, help="Regex on file name to select test images")
    parser.add_argument("--test_list_txt", type=str, default=None, help="A text file listing test images (one per line)")
    parser.add_argument("--read_test_from_model_cfg", action="store_true", help="Try to read test split from each model's cfg_args")

    # Output
    parser.add_argument("--save_json", type=str, default=None, help="Write full results to this JSON path")

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

    t0 = time.perf_counter()
    all_results: Dict[int, Dict[str, float]] = {}
    global_ssim_sum = global_psnr_sum = global_lpips_sum = 0.0
    global_count = 0

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
            model_dir
        )
        if not names:
            print(f"[warn] No test images for frame {i}; skipping.")
            continue

        try:
            res = test_one_frame(
                i, gt_frame_dir, model_dir, args.iteration, args.sparse_id,
                pipe, args.sh_degree, args.white_background, names,
                lpips_net=args.lpips_net, limit=args.limit
            )
        except Exception as e:
            print(f"[frame {i}][ERROR] {e}")
            continue

        all_results[i] = res
        global_ssim_sum += res["SSIM"] * res["count"]
        global_psnr_sum += res["PSNR"] * res["count"]
        global_lpips_sum += res["LPIPS"] * res["count"]
        global_count += res["count"]

        print(f"\n[frame {i}] views={res['count']}  SSIM={res['SSIM']:.6f}  PSNR={res['PSNR']:.6f}  LPIPS={res['LPIPS']:.6f}")

    dt = time.perf_counter() - t0

    if global_count > 0:
        g_ssim = global_ssim_sum / global_count
        g_psnr = global_psnr_sum / global_count
        g_lpips = global_lpips_sum / global_count
        print("\n================= [TOTAL] =================")
        print(f"Total views = {global_count}")
        print(f"SSIM  = {g_ssim:.7f}")
        print(f"PSNR  = {g_psnr:.7f}")
        print(f"LPIPS = {g_lpips:.7f}")
        print(f"Time  = {dt:.3f} s")
        global_result = {"count": global_count, "SSIM": g_ssim, "PSNR": g_psnr, "LPIPS": g_lpips, "time_sec": dt}
    else:
        print("\n[WARN] No frames evaluated.")
        global_result = {"count": 0, "SSIM": 0.0, "PSNR": 0.0, "LPIPS": 0.0, "time_sec": dt}

    if args.save_json:
        out = {"frames": all_results, "global": global_result}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[write] JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
