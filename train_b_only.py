# train_b_only.py
# 目的：硬性固定A（训练时完全不参与），只在B内训练；损失只在B投影区域计算。
import os, sys, torch
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
from random import randint
from tqdm import tqdm

from roi_utils import AABB, compute_b_mask, project_aabb_to_mask

# --------------------
# 点云 / 深度 / 掩码 IO
# --------------------
def load_points_world(path: str) -> torch.Tensor:
    """
    支持：
      - .npy   : shape (N,3 or >=3)
      - .ply   : Graphdeco常用格式（x,y,z在vertex属性）
      - 其它   : fallback 到 Open3D
    返回 CUDA float32 Tensor, shape (N,3)
    """
    import numpy as np
    if path is None:
        return None
    if path.endswith(".npy"):
        arr = np.load(path)
        xyz = arr[:, :3].astype("float32")
        return torch.from_numpy(xyz).cuda()
    try:
        import plyfile
        ply = plyfile.PlyData.read(path)
        v = ply["vertex"].data
        import numpy as np
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("float32")
        return torch.from_numpy(xyz).cuda()
    except Exception:
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            import numpy as np
            xyz = np.asarray(pcd.points).astype("float32")
            return torch.from_numpy(xyz).cuda()
        except Exception as e:
            raise RuntimeError(f"Failed to read point cloud: {path}\n{e}")

def load_depth_map(path: str) -> torch.Tensor:
    """
    读取 .pt（torch.save保存的 [H,W] float32）
    """
    d = torch.load(path, map_location="cuda")
    if d.dim() == 3 and d.shape[0] == 1:
        d = d[0]
    return d.float().cuda()

def maybe_precompute_masks(mask_mode, train_cams, H_list, W_list,
                           aabb_B, points_world, depthA_dir,
                           dilate_ks, intersect_aabb, eps,
                           save_masks_dir=None):
    """
    可选预计算所有相机的掩码，返回 list[Tensor(H,W)]
    """
    masks = []
    os.makedirs(save_masks_dir, exist_ok=True) if save_masks_dir else None
    for i, cam in enumerate(train_cams):
        H, W = H_list[i], W_list[i]
        depth_A = None
        if mask_mode == "pcd_depth":
            if depthA_dir is None:
                raise ValueError("pcd_depth 模式需要 --depthA_dir")
            dp = os.path.join(depthA_dir, f"depth_{i:04d}.pt")
            if not os.path.exists(dp):
                raise FileNotFoundError(f"缺少 A 深度: {dp}")
            depth_A = load_depth_map(dp)

        m = compute_b_mask(mask_mode, cam, H, W,
                           aabb=aabb_B,
                           points_world=points_world,
                           depth_A=depth_A,
                           dilate_ks=dilate_ks,
                           intersect_aabb=intersect_aabb,
                           eps=eps)
        masks.append(m.half())  # 省显存
        if save_masks_dir:
            torch.save(m.half().cpu(), os.path.join(save_masks_dir, f"mask_{i:04d}.pt"))
    return masks

def maybe_load_masks(load_masks_dir, n_cams):
    if not load_masks_dir:
        return None
    masks = []
    for i in range(n_cams):
        p = os.path.join(load_masks_dir, f"mask_{i:04d}.pt")
        if not os.path.exists(p):
            return None
        m = torch.load(p, map_location="cuda").float()
        masks.append(m)
    return masks

# --------------------
# 仅保留 B 区域高斯（开局剔除A）
# --------------------
def filter_points_in_roi(gaussians: GaussianModel, aabb: AABB):
    # 直接修改模型中的点：只保留B中点
    xyz = gaussians.get_xyz  # (N,3)
    mask = aabb.contains(xyz)
    keep = mask.nonzero(as_tuple=False).squeeze(1)
    if keep.numel() == 0:
        print("[warn] No initial Gaussians remain in B after ROI filtering.")
    if keep.numel() == xyz.shape[0]:
        return  # 全在B，跳过
    def _index_rows(p):
        return torch.nn.Parameter(p.data[keep].contiguous(), requires_grad=p.requires_grad)
    gaussians._xyz      = _index_rows(gaussians._xyz)
    gaussians._scaling  = _index_rows(gaussians._scaling)
    gaussians._rotation = _index_rows(gaussians._rotation)
    gaussians._opacity  = _index_rows(gaussians._opacity)
    gaussians._features_dc   = _index_rows(gaussians._features_dc)
    gaussians._features_rest = _index_rows(gaussians._features_rest)
    gaussians.training_setup(gaussians.opts)  # 参数数变了需要重设优化器

# --------------------
# 训练主过程
# --------------------
def training_b_only(dataset, opt, pipe,
                    testing_iterations, saving_iterations, checkpoint_iterations,
                    checkpoint, debug_from,
                    aabb: AABB,
                    mask_mode: str,
                    points_world: torch.Tensor,
                    depthA_dir: str,
                    dilate_ks: int,
                    intersect_aabb: bool,
                    eps: float,
                    precompute_masks: bool,
                    save_masks_dir: str,
                    load_masks_dir: str):
    first_iter = 0
    # === 初始化 ===
    tb = None
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.opts = opt
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 仅保留B内初始点（开局就剔除A）
    filter_points_in_roi(gaussians, aabb)

    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 相机列表（固定索引，便于掩码/深度对齐）
    train_cams = scene.getTrainCameras()
    n_cams = len(train_cams)
    H_list = [c.original_image.shape[1] for c in train_cams]
    W_list = [c.original_image.shape[2] for c in train_cams]

    # 掩码：优先加载，其次预计算
    masks = maybe_load_masks(load_masks_dir, n_cams)
    if masks is None and precompute_masks:
        masks = maybe_precompute_masks(mask_mode, train_cams, H_list, W_list,
                                       aabb, points_world, depthA_dir,
                                       dilate_ks, intersect_aabb, eps,
                                       save_masks_dir=save_masks_dir)

    progress = tqdm(range(first_iter, opt.iterations), desc=f"Training (B only, {mask_mode})")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                custom_cam, do_train, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling = network_gui.receive()
                if custom_cam is not None:
                    img = render(custom_cam, gaussians, pipe, background)["render"]
                    network_gui.send(memoryview((torch.clamp(img,0,1)*255).byte().permute(1,2,0).contiguous().cpu().numpy()), dataset.source_path)
                else:
                    network_gui.send(None, dataset.source_path)
                if do_train and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        # 采样相机（固定索引，不pop，方便取对应掩码/深度）
        i = randint(0, n_cams-1)
        cam = train_cams[i]

        # 渲染
        pkg = render(cam, gaussians, pipe, background)
        pred = pkg["render"]
        gt   = cam.original_image.cuda()
        H, W = gt.shape[1], gt.shape[2]

        # ROI 掩码
        if masks is not None:
            roi_mask = masks[i].float().cuda()
        else:
            depth_A = None
            if mask_mode == "pcd_depth":
                if depthA_dir is None:
                    raise ValueError("pcd_depth 模式需要 --depthA_dir")
                dp = os.path.join(depthA_dir, f"depth_{i:04d}.pt")
                if not os.path.exists(dp):
                    raise FileNotFoundError(f"缺少 A 深度: {dp}")
                depth_A = load_depth_map(dp)
            roi_mask = compute_b_mask(mask_mode, cam, H, W,
                                      aabb=aabb,
                                      points_world=points_world,
                                      depth_A=depth_A,
                                      dilate_ks=dilate_ks,
                                      intersect_aabb=intersect_aabb,
                                      eps=eps)

        roi_mask = roi_mask[None, ...]  # [1,H,W]
        pred = pred * roi_mask
        gt   = gt   * roi_mask

        # Loss
        Ll1 = l1_loss(pred, gt)
        try:
            from fused_ssim import fused_ssim
            ssim_val = fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        except Exception:
            ssim_val = ssim(pred, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
        loss.backward()

        # densify/prune（仅B有点）
        if iteration < opt.densify_until_iter:
            vis = pkg["visibility_filter"]
            gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], pkg["radii"][vis])
            gaussians.add_densification_stats(pkg["viewspace_points"], vis)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_th = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                            scene.cameras_extent, size_th, pkg["radii"])
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

        # 优化器步进
        gaussians.exposure_optimizer.step(); gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        if hasattr(gaussians, "optimizer") and gaussians.optimizer is not None:
            gaussians.optimizer.step(); gaussians.optimizer.zero_grad(set_to_none=True)

        # 存档/评估
        if iteration in saving_iterations:
            scene.save(iteration)
        if iteration in checkpoint_iterations:
            torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        if iteration % 10 == 0:
            progress.set_postfix({"loss": float(loss.item())})
            progress.update(10)
        if iteration == opt.iterations:
            progress.close()

    print("\n[B-only training complete]\n")

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    parser = ArgumentParser("Train only B (hard-freeze A) with flexible ROI masks")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # ROI & 掩码相关
    parser.add_argument("--roi_b", type=str, default=None, help='B的AABB，格式"xmin,xmax,ymin,ymax,zmin,zmax"')
    parser.add_argument("--roi_b_json", type=str, default=None, help="B的AABB json文件（含 xmin/xmax/...）")
    parser.add_argument("--mask_mode", type=str, default="aabb", choices=["aabb","pcd","pcd_depth"],
                        help="ROI 掩码模式：aabb | pcd | pcd_depth")
    parser.add_argument("--b_pcd", type=str, default=None,
                        help="B-only 点云路径（.npy/.ply）。mask_mode=pcd/pcd_depth 时必需")
    parser.add_argument("--depthA_dir", type=str, default=None,
                        help="第0帧A的深度目录，包含 depth_0000.pt 等；mask_mode=pcd_depth 时必需")
    parser.add_argument("--dilate_ks", type=int, default=5, help="掩码膨胀核大小（奇数）")
    parser.add_argument("--no_intersect_aabb", action="store_true", help="不与AABB投影求交（默认会求交）")
    parser.add_argument("--precompute_masks", action="store_true", help="开局为所有相机预计算掩码")
    parser.add_argument("--save_masks_dir", type=str, default=None, help="保存预计算掩码的目录（可选）")
    parser.add_argument("--load_masks_dir", type=str, default=None, help="从该目录加载已算好的掩码（可选）")
    parser.add_argument("--eps", type=float, default=1e-3, help="pcd_depth 模式下的深度松弛量")

    # 其他
    parser.add_argument('--ip', type=str, default="127.0.0.1"); parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--disable_viewer', action='store_true', default=True)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    safe_state(False)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)

    aabb = AABB.from_json(args.roi_b_json) if args.roi_b_json else AABB.from_str(args.roi_b)
    points_world = None
    if args.mask_mode in ("pcd", "pcd_depth"):
        if not args.b_pcd:
            raise ValueError("mask_mode=pcd/pcd_depth 需要提供 --b_pcd")
        points_world = load_points_world(args.b_pcd)  # (N,3) CUDA

    training_b_only(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, -1, aabb,
        mask_mode=args.mask_mode,
        points_world=points_world,
        depthA_dir=args.depthA_dir,
        dilate_ks=args.dilate_ks,
        intersect_aabb=(not args.no_intersect_aabb),
        eps=args.eps,
        precompute_masks=args.precompute_masks,
        save_masks_dir=args.save_masks_dir,
        load_masks_dir=args.load_masks_dir
    )
