# English-only script
import os, numpy as np, random
from pathlib import Path
from PIL import Image
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from plyfile import PlyData

def load_scene(frame_dir):
    sparse = Path(frame_dir)/"sparse/0"
    cam_extr = read_extrinsics_text(os.fspath(sparse/"images.txt"))
    cam_intr = read_intrinsics_text(os.fspath(sparse/"cameras.txt"))
    ply = PlyData.read(os.fspath(sparse/"points3D.ply"))
    xyz = np.vstack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']]).T
    return cam_extr, cam_intr, xyz

def world2cam_uv(R, T, K, X):
    # R here is already transpose(q2R) just like 3DGS
    Xc = (R.T @ (X.T) + T.reshape(3,1)).T  # world->cam
    Zc = Xc[:,2]
    uv = (K @ (Xc[:, :3] / np.clip(Zc[:,None], 1e-9, None)).T).T[:, :2]
    return uv, Zc

def quick_check(frame_dir, n=10000, idx=0):
    cam_extr, cam_intr, xyz = load_scene(frame_dir)
    key = list(cam_extr.keys())[idx]
    extr = cam_extr[key]; intr = cam_intr[extr.camera_id]
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)
    if intr.model == "PINHOLE":
        fx, fy, cx, cy = intr.params[0], intr.params[1], intr.params[2], intr.params[3]
        w, h = intr.width, intr.height
    else:
        raise RuntimeError("Expect PINHOLE")
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

    if xyz.shape[0] > n:
        sel = np.random.choice(xyz.shape[0], n, replace=False)
        xyz = xyz[sel]

    uv, Zc = world2cam_uv(R, T, K, xyz)
    inside = (Zc > 0) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
    print(f"Front-facing %: {(Zc>0).mean()*100:.1f} , inside image %: {inside.mean()*100:.1f} (expect both high)")
    return inside.mean()


quick_check("dataset/frame_1")
