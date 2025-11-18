# ply_visualizer.py
# Optional:
#   --voxel 0.01         # voxel downsample size in meters (speeds up viewing)
#   --every 1            # keep every N-th point (uniform subsample; applied after voxel)
#   --point-size 2.0     # viewer point size
#   --bg white           # viewer background: black | white
#   --fit-plane          # RANSAC-fit a dominant plane (paint it red) to sanity-check orientation
#   --normals            # estimate normals for better shading (slower for very large clouds)
#
# Extra (camera track):
#   --cam-track camera_track.json
#   --cam-every 5
#   --cam-frame 60
#   --cam-scale 1.0

import argparse
import os
import sys
import json
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    print("Open3D is required. Install with: pip install open3d", file=sys.stderr)
    raise


def pretty_bbox(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    mn = np.asarray(aabb.get_min_bound())
    mx = np.asarray(aabb.get_max_bound())
    ext = mx - mn
    center = (mn + mx) * 0.5
    diag = np.linalg.norm(ext)
    return {
        "min": mn,
        "max": mx,
        "extent": ext,
        "center": center,
        "diag": float(diag),
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


def make_camera_geometries(track_json_path, bbox_diag, scale=1.0, every=1, frame_id=None):
    """Load camera_track.json and create Open3D geometries.

    - Draw a small coordinate frame for each camera (or a single frame if frame_id is set).
    - Connect camera centers with a polyline to visualize the trajectory.
    - We assume R, T describe a world->camera transform: X_cam = R * X_world + T
      so camera center C_world = -R^T * T, and camera orientation in world is R^T.
    """
    if not os.path.isfile(track_json_path):
        print(f"[Warn] camera track not found: {track_json_path}", file=sys.stderr)
        return []

    with open(track_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"[Warn] camera track JSON has unexpected structure or is empty: {track_json_path}", file=sys.stderr)
        return []

    # Optional: filter to a specific frame by 'frame' field
    if frame_id is not None:
        cams = [c for c in data if int(c.get("frame", -1)) == int(frame_id)]
        if len(cams) == 0:
            print(f"[Warn] No camera with frame == {frame_id} found in camera_track.json", file=sys.stderr)
            return []
    else:
        cams = data

    if every < 1:
        every = 1

    geoms = []
    centers = []

    # Small default size for camera coordinate frames, relative to scene scale
    cam_cf_size = float(scale) * max(0.02 * bbox_diag, 0.1)

    # iterate with subsampling if we're not locked to a single frame
    iterable = cams if frame_id is not None else cams[::every]

    for cam in iterable:
        R = np.asarray(cam["R"], dtype=float).reshape(3, 3)
        T = np.asarray(cam["T"], dtype=float).reshape(3, 1)

        # world->cam: X_cam = R X_world + T
        # => camera center in world coordinates:
        C = -R.T @ T  # (3,1)
        C = C.reshape(3)

        # camera orientation in world: R_world = R^T
        R_world = R.T

        # Build 4x4 transform for Open3D
        T_world = np.eye(4, dtype=float)
        T_world[:3, :3] = R_world
        T_world[:3, 3] = C

        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cam_cf_size)
        cf.transform(T_world)
        geoms.append(cf)
        centers.append(C)

    centers = np.asarray(centers, dtype=float)
    if centers.shape[0] >= 2:
        # Create a polyline connecting camera centers
        points = o3d.utility.Vector3dVector(centers)
        lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(centers.shape[0] - 1)]
        )
        # black line
        colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.0, 0.0]], dtype=float), (centers.shape[0] - 1, 1))
        )
        line_set = o3d.geometry.LineSet(points=points, lines=lines)
        line_set.colors = colors
        geoms.append(line_set)

    # Some logs
    print(f"[CameraTrack] Loaded {len(cams)} entries from {track_json_path}")
    if centers.shape[0] > 0:
        print(f"[CameraTrack] Visualizing {centers.shape[0]} cameras (every={every}, frame_id={frame_id})")
        print(f"[CameraTrack] First center: {centers[0]}")
        print(f"[CameraTrack] Last  center: {centers[-1]}")
    return geoms


def main():
    ap = argparse.ArgumentParser(description="Interactive PLY viewer for fused_points.ply")
    ap.add_argument("--ply", required=True, help="Path to fused_points.ply (xyz + rgb)")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel size in meters for downsample")
    ap.add_argument("--every", type=int, default=1,
                    help="Keep every N-th point after voxel (for point cloud)")
    ap.add_argument("--point-size", type=float, default=2.0, help="Viewer point size")
    ap.add_argument("--bg", choices=["black", "white"], default="white",
                    help="Viewer background color")
    ap.add_argument("--fit-plane", action="store_true",
                    help="RANSAC-fit & highlight a dominant plane (red)")
    ap.add_argument("--normals", action="store_true", help="Estimate normals for shading")

    # Camera track options
    ap.add_argument(
        "--cam-track",
        dest="cam_track",
        default=None,
        help="Optional path to camera_track.json for visualizing camera centers/orientations",
    )
    ap.add_argument(
        "--cam-every",
        dest="cam_every",
        type=int,
        default=1,
        help="Visualize every N-th camera from track (ignored if --cam-frame is set)",
    )
    ap.add_argument(
        "--cam-frame",
        dest="cam_frame",
        type=int,
        default=None,
        help="If set, only visualize this camera frame id (matches 'frame' field in JSON)",
    )
    ap.add_argument(
        "--cam-scale",
        dest="cam_scale",
        type=float,
        default=1.0,
        help="Relative scale factor for camera coordinate frames (multiplies auto bbox-based size)",
    )

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

    # Optional uniform subsample (point cloud)
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

    # Optional camera track visualization
    if args.cam_track is not None:
        cam_geoms = make_camera_geometries(
            track_json_path=os.path.abspath(args.cam_track),
            bbox_diag=bbox["diag"],
            scale=args.cam_scale,
            every=args.cam_every,
            frame_id=args.cam_frame,
        )
        geoms.extend(cam_geoms)

    # Add a global/world coordinate frame scaled to bbox size
    cf_size = max(0.05 * bbox["diag"], 0.1)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=cf_size))

    # Launch interactive viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="PLY + CameraTrack Preview – Open3D",
        width=1280,
        height=800,
        visible=True,
    )
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

# Example:

