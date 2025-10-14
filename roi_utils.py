# roi_utils.py
# 工具：1) AABB 定义/解析；2) 判断点是否在 AABB；
#      3) B 的 AABB 投影掩码（已有）；
#      4) [新增] 点云支持掩码（②）
#      5) [新增] 带遮挡判断（与 A 深度比较）的点云支持掩码（③）
#      6) [新增] 通用 compute_b_mask() 统一入口

from dataclasses import dataclass
from typing import Tuple, Optional
import json
import torch
import numpy as np
import os

# ========== AABB & AABB投影 ==========
@dataclass
class AABB:
    xmin: float; xmax: float
    ymin: float; ymax: float
    zmin: float; zmax: float

    @staticmethod
    def from_str(s: str) -> "AABB":
        # 格式： "xmin,xmax,ymin,ymax,zmin,zmax"
        vals = [float(x) for x in s.split(",")]
        assert len(vals) == 6
        return AABB(*vals)

    @staticmethod
    def from_json(path: str) -> "AABB":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return AABB(j["xmin"], j["xmax"], j["ymin"], j["ymax"], j["zmin"], j["zmax"])

    def contains(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (N,3)
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        return (x >= self.xmin) & (x <= self.xmax) & \
               (y >= self.ymin) & (y <= self.ymax) & \
               (z >= self.zmin) & (z <= self.zmax)

def project_aabb_to_mask(aabb: AABB, cam, H: int, W: int) -> torch.Tensor:
    """
    给定一个相机，把世界坐标AABB的8个角点投影到像平面，生成二值mask（B 投影=1，其他=0）。
    要求 cam 提供 full_proj_transform (4x4).
    """
    corners = torch.tensor([
        [aabb.xmin, aabb.ymin, aabb.zmin, 1.0],
        [aabb.xmin, aabb.ymin, aabb.zmax, 1.0],
        [aabb.xmin, aabb.ymax, aabb.zmin, 1.0],
        [aabb.xmin, aabb.ymax, aabb.zmax, 1.0],
        [aabb.xmax, aabb.ymin, aabb.zmin, 1.0],
        [aabb.xmax, aabb.ymin, aabb.zmax, 1.0],
        [aabb.xmax, aabb.ymax, aabb.zmin, 1.0],
        [aabb.xmax, aabb.ymax, aabb.zmax, 1.0],
    ], dtype=torch.float32, device="cuda")

    M = cam.full_proj_transform.cuda()  # (4,4)
    clip = (M @ corners.T).T  # (8,4)
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-6)
    xs = (ndc[:,0] * 0.5 + 0.5) * (W - 1)
    ys = (1.0 - (ndc[:,1] * 0.5 + 0.5)) * (H - 1)
    poly = torch.stack([xs, ys], dim=1).detach().cpu().numpy().astype(np.float32)

    try:
        import cv2
        hull = cv2.convexHull(poly)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        return torch.from_numpy(mask.astype(np.float32) / 255.0).cuda()
    except Exception:
        x0, y0 = np.clip(poly.min(axis=0), [0,0], [W-1,H-1]).astype(int)
        x1, y1 = np.clip(poly.max(axis=0), [0,0], [W-1,H-1]).astype(int)
        mask = torch.zeros((H, W), device="cuda", dtype=torch.float32)
        mask[y0:y1+1, x0:x1+1] = 1.0
        return mask

# ========== 公共小工具：点投影、形态学“膨胀” ==========
def _dilate_mask(mask_hw: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    用 max_pool2d 近似膨胀。mask_hw: [H,W] (0/1)
    """
    if k <= 1: 
        return mask_hw
    import torch.nn.functional as F
    x = mask_hw[None,None]  # [1,1,H,W]
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=k//2)
    return y[0,0]

def _project_points_world_to_pixels(points_world: torch.Tensor, cam, H: int, W: int):
    """
    将世界坐标点投影到像素，并返回：
      - uv_ij: (N,2) 像素整型坐标 (x=j, y=i)
      - z_cam: (N,) 相机坐标系深度（正向为前）
      - valid: (N,) 是否投影到可见像素且 z_cam>0
    要求 cam 提供：
      - full_proj_transform (4x4) 用于像素坐标
      - world_view_transform (4x4) 用于取相机空间 z
    """
    device = torch.device("cuda")
    if not torch.is_tensor(points_world):
        points_world = torch.tensor(points_world, dtype=torch.float32, device=device)
    else:
        points_world = points_world.to(device).float()

    one = torch.ones((points_world.shape[0],1), device=device, dtype=torch.float32)
    Pw = torch.cat([points_world, one], dim=1)  # (N,4)

    # 像素坐标
    M = cam.full_proj_transform.cuda()  # [4,4]
    clip = (M @ Pw.T).T  # [N,4]
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-6)
    xs = (ndc[:,0] * 0.5 + 0.5) * (W - 1)
    ys = (1.0 - (ndc[:,1] * 0.5 + 0.5)) * (H - 1)

    # 相机空间 z
    # world -> camera
    W2C = cam.world_view_transform.cuda()
    Pc = (W2C @ Pw.T).T  # [N,4]
    z_cam = Pc[:,2]  # 右手系下，通常 z>0 表示在前方（如实现不同请据工程改符号）

    # 有效性
    valid = (z_cam > 0) & \
            (xs >= 0) & (xs <= (W-1)) & \
            (ys >= 0) & (ys <= (H-1))
    uv_ij = torch.stack([xs.round().long(), ys.round().long()], dim=1)  # (N,2), x=j, y=i
    return uv_ij, z_cam, valid

# ========== ② 点云支持掩码 ==========
def pcd_support_mask(points_world, cam, H: int, W: int,
                     aabb: Optional[AABB] = None,
                     dilate_ks: int = 5,
                     intersect_aabb: bool = True) -> torch.Tensor:
    """
    基于 B-only 点云生成“支持掩码”：只有被点云投影覆盖的像素为 1，其余 0。
    - 可选 aabb 与 AABB 投影求交，进一步限制 ROI。
    - 可选 dilate_ks 做形态学膨胀（覆盖边缘）。
    返回: [H,W] float32 0/1
    """
    if aabb is not None:
        # 若传入全场景点云，这里先过滤到 B 内
        if not torch.is_tensor(points_world):
            pw = torch.tensor(points_world, dtype=torch.float32, device="cuda")
        else:
            pw = points_world.to("cuda").float()
        mask_b = aabb.contains(pw[:, :3])
        points_world = pw[mask_b].contiguous()

    if (not torch.is_tensor(points_world)) or points_world.numel() == 0:
        return torch.zeros((H,W), device="cuda", dtype=torch.float32)

    uv, _, valid = _project_points_world_to_pixels(points_world[:, :3], cam, H, W)
    uv = uv[valid]
    mask = torch.zeros((H, W), device="cuda", dtype=torch.float32)
    if uv.numel() > 0:
        mask[uv[:,1].clamp(0,H-1), uv[:,0].clamp(0,W-1)] = 1.0
    mask = _dilate_mask(mask, k=dilate_ks)

    if intersect_aabb and aabb is not None:
        aabb_mask = project_aabb_to_mask(aabb, cam, H, W)
        mask = mask * aabb_mask
    return mask

# ========== ③ 带遮挡判断（与 A 深度比较）的点云支持掩码 ==========
def pcd_support_mask_with_depth(points_world, cam, H: int, W: int,
                                depth_A: torch.Tensor,
                                aabb: Optional[AABB] = None,
                                dilate_ks: int = 5,
                                eps: float = 1e-3,
                                intersect_aabb: bool = True) -> torch.Tensor:
    """
    在 ② 的基础上增加遮挡判断：仅当 B 支持点比 A 的深度更近时，才点亮像素。
    - depth_A: [H,W] torch.float32，单位需与 z_cam 一致（通常是相机坐标 z 或视线深度）。
    - eps: 深度比较的松弛量（防止数值抖动）。
    返回: [H,W] float32 0/1
    """
    if aabb is not None:
        if not torch.is_tensor(points_world):
            pw = torch.tensor(points_world, dtype=torch.float32, device="cuda")
        else:
            pw = points_world.to("cuda").float()
        mask_b = aabb.contains(pw[:, :3])
        points_world = pw[mask_b].contiguous()

    if (not torch.is_tensor(points_world)) or points_world.numel() == 0:
        return torch.zeros((H,W), device="cuda", dtype=torch.float32)

    depth_A = depth_A.to("cuda").float()
    uv, z_cam, valid = _project_points_world_to_pixels(points_world[:, :3], cam, H, W)

    uv = uv[valid]; z_cam = z_cam[valid]
    mask = torch.zeros((H, W), device="cuda", dtype=torch.float32)
    if uv.numel() > 0:
        # 与 A 深度比较：B 更近 → 留下
        da = depth_A[uv[:,1].clamp(0,H-1), uv[:,0].clamp(0,W-1)]
        keep = (da <= 0) | (z_cam < da - eps)  # 若 A 深度无效(<=0)则默认通过
        uv = uv[keep]
        if uv.numel() > 0:
            mask[uv[:,1].clamp(0,H-1), uv[:,0].clamp(0,W-1)] = 1.0

    mask = _dilate_mask(mask, k=dilate_ks)
    if intersect_aabb and aabb is not None:
        aabb_mask = project_aabb_to_mask(aabb, cam, H, W)
        mask = mask * aabb_mask
    return mask

# ========== 统一入口（便于在训练脚本里切换模式） ==========
def compute_b_mask(mode: str,
                   cam,
                   H: int, W: int,
                   aabb: Optional[AABB] = None,
                   points_world: Optional[torch.Tensor] = None,
                   depth_A: Optional[torch.Tensor] = None,
                   dilate_ks: int = 5,
                   intersect_aabb: bool = True,
                   eps: float = 1e-3) -> torch.Tensor:
    """
    mode:
      - "aabb"       : 仅用 AABB 投影（①）
      - "pcd"        : 点云支持掩码（②）
      - "pcd_depth"  : 点云支持掩码 + 与 A 深度比较（③）
    其它参数见各函数说明。
    """
    if mode == "aabb":
        assert aabb is not None, "aabb mode requires AABB"
        return project_aabb_to_mask(aabb, cam, H, W)
    elif mode == "pcd":
        assert points_world is not None, "pcd mode requires points_world"
        return pcd_support_mask(points_world, cam, H, W, aabb=aabb,
                                dilate_ks=dilate_ks, intersect_aabb=intersect_aabb)
    elif mode == "pcd_depth":
        assert (points_world is not None) and (depth_A is not None), "pcd_depth requires points_world & depth_A"
        return pcd_support_mask_with_depth(points_world, cam, H, W, depth_A=depth_A,
                                           aabb=aabb, dilate_ks=dilate_ks, eps=eps,
                                           intersect_aabb=intersect_aabb)
    else:
        raise ValueError(f"Unknown mask mode: {mode}")

# ==========================
# 用法示例（在 train_b_only.py 中）
# --------------------------
# from roi_utils import compute_b_mask, AABB
#
# # ① AABB 掩码：
# roi_mask = compute_b_mask("aabb", cam, H, W, aabb=aabb_B)                 # [H,W]
#
# # ② 点云支持掩码（points_world = 当前帧 B-only 点云 (N,3) or (N,>=3)）：
# roi_mask = compute_b_mask("pcd", cam, H, W, aabb=aabb_B, points_world=P_B) # [H,W]
#
# # ③ 带遮挡判断（depth_A 为 “第0帧A”渲的深度图 [H,W]）：
# roi_mask = compute_b_mask("pcd_depth", cam, H, W, aabb=aabb_B,
#                           points_world=P_B, depth_A=depth_A, eps=1e-3)
#
# # 然后在 loss 前做：
# pred = pred * roi_mask[None]
# gt   = gt   * roi_mask[None]
# ==========================
