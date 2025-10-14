# make_roi_masks.py
import os, torch, json
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from roi_utils import AABB, project_aabb_to_mask

parser = ArgumentParser("Precompute ROI masks for all train cameras")
lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--roi_b", type=str, default=None)
parser.add_argument("--roi_b_json", type=str, default=None)
args = parser.parse_args()

aabb = AABB.from_json(args.roi_b_json) if args.roi_b_json else AABB.from_str(args.roi_b)
scene = Scene(lp.extract(args), GaussianModel(lp.extract(args).sh_degree, "adam"))
cams = scene.getTrainCameras()

os.makedirs(args.out_dir, exist_ok=True)
for i, cam in enumerate(cams):
    H, W = cam.original_image.shape[1], cam.original_image.shape[2]
    m = project_aabb_to_mask(aabb, cam, H, W).cpu()
    torch.save(m, os.path.join(args.out_dir, f"mask_{i:04d}.pt"))
print(f"Saved {len(cams)} masks to {args.out_dir}")
