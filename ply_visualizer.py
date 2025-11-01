# view_ply_o3d.py
# Interactive PLY point cloud viewer for checking fused_points.ply from RS Studio.
# Usage:
#   python view_ply_o3d.py --ply /path/to/frame_1/fused_points.ply
# Optional:
#   --voxel 0.01         # voxel downsample size in meters (speeds up viewing)
#   --every 1            # keep every N-th point (uniform subsample; applied after voxel)
#   --point-size 2.0     # viewer point size
#   --bg white           # viewer background: black | white
#   --fit-plane          # RANSAC-fit a dominant plane (paint it red) to sanity-check orientation
#   --normals            # estimate normals for better shading (slower for very large clouds)

import argparse
import os
import sys
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    print("Open3D is required. Install with: pip install open3d", file=sys.stderr)
    raise

def pretty_bbox(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    mn   = np.asarray(aabb.get_min_bound())
    mx   = np.asarray(aabb.get_max_bound())
    ext  = mx - mn
    center = (mn + mx) * 0.5
    diag = np.linalg.norm(ext)
    return {
        "min": mn, "max": mx, "extent": ext,
        "center": center, "diag": float(diag)
    }

def ensure_color_unit_range(pcd):
    """Open3D expects colors in [0,1]. Convert from [0,255] if needed."""
    if not pcd.has_colors():
        return
    col = np.asarray(pcd.colors)
    if col.size == 0:
        return
    if col.max() > 1.0:
        col = np.clip(col / 255.0, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(col)

def main():
    ap = argparse.ArgumentParser(description="Interactive PLY viewer for fused_points.ply")
    ap.add_argument("--ply", required=True, help="Path to fused_points.ply (xyz + rgb)")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel size in meters for downsample")
    ap.add_argument("--every", type=int, default=1, help="Keep every N-th point after voxel")
    ap.add_argument("--point-size", type=float, default=2.0, help="Viewer point size")
    ap.add_argument("--bg", choices=["black", "white"], default="white", help="Viewer background color")
    ap.add_argument("--fit-plane", action="store_true", help="RANSAC-fit & highlight a dominant plane (red)")
    ap.add_argument("--normals", action="store_true", help="Estimate normals for shading")
    args = ap.parse_args()

    ply_path = os.path.abspath(args.ply)
    if not os.path.isfile(ply_path):
        print(f"File not found: {ply_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Info] Loading: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        print("[Error] Loaded point cloud is empty.", file=sys.stderr)
        sys.exit(2)

    ensure_color_unit_range(pcd)

    # Optional voxel downsample
    if args.voxel and args.voxel > 0.0:
        print(f"[Info] Voxel downsample @ {args.voxel} m ...")
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    # Optional uniform subsample
    if args.every > 1:
        pts = np.asarray(pcd.points)
        keep = np.arange(0, pts.shape[0], args.every)
        pcd = pcd.select_by_index(keep, invert=False)

    # Optional normals
    if args.normals:
        # Radius ≈ 2 × mean NN spacing heuristic using bbox diagonal
        bbox = pretty_bbox(pcd)
        radius = max(bbox["diag"] * 0.01, 1e-3)
        print(f"[Info] Estimating normals (radius={radius:.4f}) ...")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.normalize_normals()

    # Print quick stats
    bbox = pretty_bbox(pcd)
    N = len(pcd.points)
    print(f"[OK] Points: {N:,}")
    print(f"     BBox min: {bbox['min']}")
    print(f"     BBox max: {bbox['max']}")
    print(f"     Extent  : {bbox['extent']}")
    print(f"     Center  : {bbox['center']}")
    print(f"     Diagonal: {bbox['diag']:.3f} m")

    geoms = [pcd]

    # Optional plane fit to quickly verify an obvious ground/wall
    if args.fit_plane:
        print("[Info] Fitting dominant plane with RANSAC ...")
        # Distance threshold ~ 0.5% of bbox diagonal (clamped)
        thresh = float(max(0.001, min(0.02 * bbox["diag"], 0.05)))
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=thresh, ransac_n=3, num_iterations=1000
            )
            print(f"     Plane inliers: {len(inliers):,} @ thresh={thresh:.4f}")
            inlier = pcd.select_by_index(inliers)
            outlier = pcd.select_by_index(inliers, invert=True)
            inlier.paint_uniform_color([1.0, 0.0, 0.0])   # red plane
            geoms = [outlier, inlier]
        except Exception as e:
            print(f"[Warn] Plane fitting failed: {e}")

    # Add a coordinate frame scaled to bbox size
    cf_size = max(0.05 * bbox["diag"], 0.1)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=cf_size))

    # Launch interactive viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Preview – Open3D", width=1280, height=800, visible=True)
    for g in geoms:
        vis.add_geometry(g)

    # Render options
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([1, 1, 1]) if args.bg == "white" else np.array([0, 0, 0])
    opt.show_coordinate_frame = False

    print("[Info] Controls: Mouse to rotate/pan/zoom, 'H' for help, 'R' to reset view.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()

#python ply_visualizer.py --ply dataset\soccer_dynamic_player_B\frame_1\fused_points.ply --voxel 0.01 --every 1 --point-size 2.0 --bg white --fit-plane --normals