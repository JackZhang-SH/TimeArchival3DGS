#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from random import randint
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False
print("[fused-ssim]", "ON" if FUSED_SSIM_AVAILABLE else "OFF")

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def _load_mask_and_roi(source_path: str, image_name: str, H: int, W: int, pad_px: int = 2):
    """
    Load a binary 0/1 mask from work_dataset/masks_raw/<stem>.npy.
    Returns (mask_hw: torch.float32 CUDA [H,W], (y0,y1,x0,x1) or None).
    If the file is missing or empty, returns (None, None).
    """
    npy_path = Path(source_path) / "masks_raw" / (Path(image_name).stem + ".npy")
    if not npy_path.exists():
        return None, None
    try:
        mask_np = np.load(str(npy_path))
        if mask_np.dtype not in (np.uint8, np.bool_):
            mask_np = (mask_np > 0).astype(np.uint8)
        ys, xs = np.nonzero(mask_np)
        if ys.size == 0:
            return None, None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if pad_px > 0:
            y0 = max(0, y0 - pad_px)
            x0 = max(0, x0 - pad_px)
            y1 = min(H, y1 + pad_px)
            x1 = min(W, x1 + pad_px)
        mask_t = torch.from_numpy((mask_np > 0).astype(np.float32))  # HxW, 0/1 float
        mask_t = mask_t.to("cuda", non_blocking=True)
        return mask_t, (y0, y1, x0, x1)
    except Exception:
        return None, None


def _masked_l1_and_ssim(image, gt_image, mask_hw, roi, lambda_dssim, fused_ssim_available):
    """
    Compute masked L1 inside ROI; for SSIM, replace prediction with GT outside mask
    so outside pixels don't contribute to the difference.

    Returns (Ll1, ssim_value) as scalar tensors.
    """
    if roi is not None:
        y0, y1, x0, x1 = roi
        img_roi = image[:, y0:y1, x0:x1]
        gt_roi = gt_image[:, y0:y1, x0:x1]
        m_roi = mask_hw[y0:y1, x0:x1]
    else:
        img_roi = image
        gt_roi = gt_image
        m_roi = mask_hw

    # L1 over mask only
    m3 = m_roi.unsqueeze(0)  # 1xHxW
    diff = (img_roi - gt_roi).abs() * m3
    denom = (m_roi.sum() * img_roi.shape[0]).clamp_min(1.0)
    Ll1 = diff.sum() / denom

    # SSIM with prediction replaced by GT outside mask
    img4ssim = img_roi * m3 + gt_roi * (1.0 - m3)
    if lambda_dssim == 0:
        ssim_value = torch.tensor(0.0, device=image.device)
    else:
        if fused_ssim_available:
            ssim_value = fused_ssim(img4ssim.unsqueeze(0), gt_roi.unsqueeze(0))
        else:
            ssim_value = ssim(img4ssim, gt_roi)
    return Ll1, ssim_value


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, reset_checkpoint_iter: bool = False,):
    # Cache for (mask_hw, roi) per image_name to avoid per-iter np.load/roi compute
    mask_cache = {}

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel]."
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # ---- TA: allow specifying test images at train time (by file name or stem, or regex) ----
    import re

    def _name_key(cam):
        n = getattr(cam, "image_name", None)
        return n if n is not None else ""

    try:
        want_names = set()
        if getattr(dataset, "test_images", None):
            for t in dataset.test_images:
                t = str(t)
                want_names.add(t)              # full name, e.g., 0007.png
                want_names.add(Path(t).stem)   # stem, e.g., 0007

        regex_obj = None
        if getattr(dataset, "test_regex", None):
            try:
                regex_obj = re.compile(dataset.test_regex)
            except Exception as e:
                print(f"[warn] invalid --test_regex: {e}")

        if want_names or regex_obj is not None or getattr(dataset, "test_clear_default", False):
            train_list = scene.getTrainCameras().copy()
            test_list = scene.getTestCameras().copy()
            picked, remain = [], []

            for cam in train_list:
                nm = _name_key(cam)
                stem = Path(nm).stem
                hit = False
                if nm in want_names or stem in want_names:
                    hit = True
                if (not hit) and (regex_obj is not None) and regex_obj.search(nm):
                    hit = True
                (picked if hit else remain).append(cam)

            if getattr(dataset, "test_clear_default", False):
                test_list = []

            test_list = list(test_list) + picked
            train_list = remain

            # Monkey-patch accessors so downstream code uses the overridden split
            scene.getTrainCameras = (lambda tl=train_list: (lambda: tl))()
            scene.getTestCameras = (lambda vl=test_list: (lambda: vl))()
            print(
                f"[test-split] train={len(train_list)}  test={len(test_list)} "
                f"(picked={len(picked)}, clear_default={getattr(dataset, 'test_clear_default', False)})"
            )
            # [TA] Persist the final test list to <model_dir>/test_images.txt for reproducible evaluation
            try:
                out_txt = Path(scene.model_path) / "test_images.txt"
                out_txt.parent.mkdir(parents=True, exist_ok=True)
                with open(out_txt, "w", encoding="utf-8") as f:
                    for cam in test_list:
                        name = getattr(cam, "image_name", "")
                        f.write(f"{Path(name).name}\n")   # 写入 basename，如 0007.png
                print(f"[test-split] wrote test list → {out_txt} ({len(test_list)} names)")
            except Exception as e:
                print(f"[test-split][warn] failed to write test_images.txt: {e}")

    except Exception as e:
        print(f"[test-split][warn] failed to apply test override: {e}")
    # ---- end TA block ----

    if checkpoint:
        model_params, ckpt_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # reset_checkpoint_iter=True 表示：忽略保存的迭代号，把 checkpoint 当作“初始化”
        first_iter = 0 if reset_checkpoint_iter else ckpt_iter

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    # Fine-grained timers for A/B speed comparison
    tR0 = torch.cuda.Event(enable_timing=True)
    tR1 = torch.cuda.Event(enable_timing=True)  # render
    tL0 = torch.cuda.Event(enable_timing=True)
    tL1 = torch.cuda.Event(enable_timing=True)  # loss
    tB0 = torch.cuda.Event(enable_timing=True)
    tB1 = torch.cuda.Event(enable_timing=True)  # backward
    tS0 = torch.cuda.Event(enable_timing=True)
    tS1 = torch.cuda.Event(enable_timing=True)  # optimizer step

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = (
                    network_gui.receive()
                )
                if custom_cam is not None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 iterations increase SH degree up to a maximum
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        tR0.record()
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )
        tR1.record()
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        tL0.record()
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(opt, "masked", False):
            H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
            key = viewpoint_cam.image_name
            if key not in mask_cache:
                mask_cache[key] = _load_mask_and_roi(dataset.source_path, key, H, W, pad_px=2)
            mask_hw, roi = mask_cache[key]

            if mask_hw is not None:
                Ll1, ssim_value = _masked_l1_and_ssim(
                    image, gt_image, mask_hw, roi, opt.lambda_dssim, FUSED_SSIM_AVAILABLE
                )
            else:
                # Fallback to full-frame if mask missing
                Ll1 = l1_loss(image, gt_image)
                if FUSED_SSIM_AVAILABLE:
                    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    ssim_value = ssim(image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        tL1.record()

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        tB0.record()
        loss.backward()
        tB1.record()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"}
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Track max radii (image-space) for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii
                    )

                if (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                tS0.record()
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                tS1.record()
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Print timing breakdown (render/loss/backward/step/iter)
            if iteration % 500 == 0:
                torch.cuda.synchronize()
                ms_render = tR0.elapsed_time(tR1)
                ms_loss = tL0.elapsed_time(tL1)
                ms_back = tB0.elapsed_time(tB1)
                ms_step = tS0.elapsed_time(tS1) if iteration < opt.iterations else 0.0
                ms_iter = iter_start.elapsed_time(iter_end)
                print(
                    f"[TimeBreakdown] render={ms_render:.3f} ms  loss={ms_loss:.3f} ms  "
                    f"backward={ms_back:.3f} ms  step={ms_step:.3f} ms  iter={ms_iter:.3f} ms"
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + f"_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--masked", action="store_true", help="Use mask-aware ROI cropping and masked loss if masks_raw/<stem>.npy exists")
    parser.add_argument(
        "--reset_start_iter",
        action="store_true",
        help="When resuming from --start_checkpoint, ignore stored iteration and restart schedule from 0.",
    )
    # ---- TA: CLI for test split override (passed through from ta_train_masked.py after '--') ----
    parser.add_argument(
        "--test_images",
        nargs="+",
        default=None,
        help="Image names or stems to move from train to test (match against final work/images names).",
    )
    parser.add_argument(
        "--test_regex",
        type=str,
        default=None,
        help="Regex applied to image file name; matched images are moved from train to test.",
    )
    parser.add_argument(
        "--test_clear_default",
        action="store_true",
        help="If set, clear the default test set and only keep those specified by --test_images/--test_regex.",
    )
    # We store these on the dataset object so the training() block can read them after ModelParams extraction.
    # ModelParams/PipelineParams/OptimizationParams .extract(...) will not consume these custom flags.
    # ----------------------------------------------------------------------------

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize RNG / environment
    safe_state(args.quiet)

    # Start GUI server if not disabled
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Extract structured args
    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    # Attach test override flags onto dataset so training() can access them
    dataset_args.test_images = args.test_images
    dataset_args.test_regex = args.test_regex
    dataset_args.test_clear_default = args.test_clear_default

    training(
        dataset_args,
        opt_args,
        pipe_args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.reset_start_iter,
    )

    print("\nTraining complete.")