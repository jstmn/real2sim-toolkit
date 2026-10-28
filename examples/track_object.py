import dataclasses
import pathlib
import time
from typing import Literal

import cv2
import h5py
import numpy as np
import trimesh
import tyro
from tqdm import tqdm

from r2st.constants import (
    get_color_intrinsics,
    get_stereo_baseline_m,
    get_workspace_bounds,
)
from r2st.core import SAM3Predictor
from r2st.geometry import (
    depth_mm_to_meters,
    masked_depth_to_points,
    transform_pose_cam_to_world,
)
from r2st.mesh_scaling import floor_points_and_voxelize, scale_glb_to_pointcloud
from r2st.pose_grpc.client import FoundationPoseClient
from r2st.stereo_depth import cached_s2m2_depth_m
from r2st.utils import (
    H264VideoWriter,
    ImageUtils,
    MeshUtils,
    TrackingVisualizer,
    VisualizerCamera,
    erode_mask,
    rgb_depth_dataset_paths,
    validate_stereo_depth_camera_group,
)

"""
# Example usage (FoundationPose gRPC server must already be running in the container):
    uv run python examples/track_object.py \
        --h5-path data/demonstrations/grail_xarm_july_2/0802_mustard/demonstration_0/merged_sensor_data.h5 \
        --cameras cam_1 \
        --camera-model-ids d435 \
        --object-description "mustard bottle" \
        --extrinsics-paths data/demonstrations/grail_xarm_july_2/extrinsics_cam_1.yaml \
        --visualize
"""


# Scene points for the table-plane fit leave out the object mask dilated by this kernel, so the object's edges (where
# the mask and depth disagree) are not mistaken for table
SCENE_EXCLUDE_DILATE_PX = 25


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see data/demonstrations/grail_xarm_july_2/merge_camera_streams.py)."""

    cameras: list[str]
    """Camera names in the h5, e.g. `zed zed2i`. The first is tracked and its frame-0 crop is sent to Meshy. Every
    camera's frame-0 object points go into the merged point cloud that the Meshy mesh is scaled to."""

    camera_model_ids: list[str]
    """Camera model id for each of --cameras, in the same order, e.g. `zed_vga zed2i_vga`. See the README camera-model
    table."""

    object_description: str
    """Language description of the object to segment and mesh, e.g. 'mustard bottle'."""

    extrinsics_paths: list[pathlib.Path] | None = None
    """Extrinsics YAML from examples/estimate_camera_extrinsics*.py (robot-base T camera) for each of --cameras, in the
    same order. Required with more than one camera, to merge their point clouds in the robot-base frame. If set, poses
    are also written in the robot-base frame."""

    output_dir: pathlib.Path = pathlib.Path("data/meshyai")
    """Directory to save generated mesh assets and tracking outputs."""

    scaled_mesh_path: pathlib.Path | None = None
    """Known mesh of the object (e.g. measured ground truth), in meters. Used as-is: no Meshy and no PCA scaling.
    Outputs go to <output_dir>/<mesh stem>__<camera>/. --object-description is still the SAM 3 prompt. Takes a single
    camera, since extra cameras are only used for scaling."""

    unscaled_mesh_path: pathlib.Path | None = None
    """Mesh of the object at an unknown scale (e.g. from an earlier Meshy run). Used instead of calling Meshy, then
    PCA-scaled to the cameras' object points like a Meshy mesh. Outputs go to <output_dir>/<mesh stem>__<camera>/."""

    server_address: str = "localhost:50051"
    """gRPC address of the FoundationPose pose-tracking server."""

    est_refine_iter: int = 5
    """FoundationPose register refine iterations."""

    track_refine_iter: int = 5
    """FoundationPose track refine iterations."""

    max_frames: int | None = None
    """If set, only process the first N frames (useful for smoke tests)."""

    save_video: bool = True
    """If set, write an RGB overlay video of the tracked poses."""

    visualize: bool = False
    """If set, start a viser server with the mesh and a timestep slider over predicted poses."""

    gif: bool = True
    """If set, render a 360-degree orbit GIF of the generated GLB."""

    use_2d_tracker: bool = True
    """Re-anchor FoundationPose (x, y) each frame with Cutie (FoundationPose++)."""

    use_kalman_filter: bool = True
    """Fuse Cutie (x, y) with FoundationPose via a 6-DoF Kalman filter. Requires use_2d_tracker."""

    crop_pointcloud_to_workspace: bool = False
    """If set, the viser point clouds only show points inside the --workspace-id robot-base box. Viewer only: tracking
    and mesh scaling are unchanged. Requires --workspace-id and --extrinsics-paths."""

    workspace_id: str | None = None
    """Workspace id from `r2st.constants.WORKSPACE_BOUNDS` (e.g. 'vention-franka-table'), used by
    --crop-pointcloud-to-workspace."""

    object_mask_erode_px: int = 5
    """Erode each camera's frame-0 SAM mask with this kernel size (0, or odd: 5 removes about 2 px per side) before
    unprojecting it to the object points used for mesh scaling and shown in viser. FoundationPose still registers with
    the full mask."""

    s2m2_model_type: str = "XL"
    """S2M2 model size (S, M, L, XL), used when the camera has a left/right stereo pair."""

    s2m2_refine_iter: int = 3
    """S2M2 local refinement iterations, used when the camera has a left/right stereo pair."""

    object_pointcloud_processing: Literal["none", "floor-points+5mm-voxel"] = "floor-points+5mm-voxel"
    """How the merged frame-0 object points are processed before the Meshy mesh is scaled to them (see
    r2st.mesh_scaling.floor_points_and_voxelize). 'floor-points+5mm-voxel': fit the table plane to the scene points
    around the object; if the object rests on it, add every object point projected onto the plane (the bottom face no
    camera sees); then keep one point per 5 mm voxel. On the 4 and 6 cm red cubes this cut the size error from about 13%
    to about 1.5%. 'none': scale to the object points as-is. Not used with --scaled-mesh-path."""


@dataclasses.dataclass
class CameraObservation:
    """One camera's RGB-D video and its frame-0 segmentation of the object."""

    camera: str
    K: np.ndarray
    rgb: np.ndarray
    """`(T, H, W, 3)` uint8"""
    depth_m: np.ndarray
    """`(T, H, W)` float32 on the rgb pixel grid, 0 where there is no depth"""
    mask0: np.ndarray
    """`(H, W)` bool object mask on frame 0"""
    object_pts0: np.ndarray
    """`(N, 3)` frame-0 object points in this camera's optical frame"""
    scene_pts0: np.ndarray
    """`(M, 3)` frame-0 points away from the object (outside the mask dilated by `SCENE_EXCLUDE_DILATE_PX`), in this
    camera's optical frame. Used to fit the table plane for --object-pointcloud-processing."""


def _load_rgb_depth_m(args: Args, camera: str, camera_model_id: str, fx: float) -> tuple[np.ndarray, np.ndarray]:
    """RGB `(T, H, W, 3)` and depth in meters `(T, H, W)` for `camera`, limited to `args.max_frames`. If the camera has
    a left/right stereo pair, the depth is S2M2 stereo depth; otherwise it is the depth stored in the h5."""
    frames = slice(None) if args.max_frames is None else slice(0, args.max_frames)
    with h5py.File(args.h5_path, "r") as f:
        group_path = f"obs/sensor_data/{camera}"
        if f"{group_path}/right/rgb" not in f:
            rgb_path, depth_path = rgb_depth_dataset_paths(f, args.h5_path, camera)
            print(f"[info] Depth source: {depth_path} (no right/rgb, so no stereo depth)")
            return f[rgb_path][frames], depth_mm_to_meters(f[depth_path][frames])
        validate_stereo_depth_camera_group(f, args.h5_path, camera)
        left_rgb = f[f"{group_path}/left/rgb"][frames]
        right_rgb = f[f"{group_path}/right/rgb"][frames]

    baseline_m = get_stereo_baseline_m(camera_model_id)
    print(
        f"[info] Depth source: S2M2 {args.s2m2_model_type} on {group_path}/left/rgb + right/rgb "
        f"(fx={fx:.3f}, baseline={baseline_m * 1000:.2f} mm)"
    )
    depth_m = cached_s2m2_depth_m(
        args.h5_path, camera, left_rgb, right_rgb, fx, baseline_m, args.s2m2_model_type, args.s2m2_refine_iter
    )
    return left_rgb, depth_m


def _observe_camera(args: Args, predictor: SAM3Predictor, camera: str, camera_model_id: str) -> CameraObservation:
    """Load `camera`'s RGB-D, segment `args.object_description` on frame 0 and unproject the masked depth."""
    color_intrinsics = get_color_intrinsics(camera_model_id)
    print(f"[info] Loading RGB-D from {args.h5_path} ({camera}) ...")
    t0 = time.perf_counter()
    rgb, depth_m = _load_rgb_depth_m(args, camera, camera_model_id, color_intrinsics.fx)
    num_frames, H, W = rgb.shape[:3]
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"Expected NxHxWx3 rgb, got {rgb.shape}"
    _log_elapsed(f"{camera}: loaded {num_frames} frames, resolution {(H, W)}", t0)
    assert (H, W) == (color_intrinsics.height, color_intrinsics.width), (
        f"{camera} rgb is {H}x{W} but {camera_model_id} color intrinsics are "
        f"{color_intrinsics.height}x{color_intrinsics.width}"
    )
    # Both depth sources are on the (left) color pixel grid. Unproject with color K.
    assert depth_m.shape == (num_frames, H, W), f"Bad depth shape: {depth_m.shape}"
    K = color_intrinsics.intrinsic_matrix.astype(np.float64)
    print(f"[info] Color K ({camera_model_id}):\n{K}")

    print(f"[info] Segmenting '{args.object_description}' on {camera} frame 0 ...")
    t0 = time.perf_counter()
    mask0 = ImageUtils.get_sam_mask(predictor, cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR), args.object_description)
    assert mask0.dtype == bool, f"Mask dtype {mask0.dtype} is not bool"
    assert mask0.sum() > 0, f"Segmentation mask is empty on {camera} for description '{args.object_description}'"
    _log_elapsed(f"{camera}: mask pixels {int(mask0.sum())} / {mask0.size}", t0)
    points_mask = erode_mask(mask0, args.object_mask_erode_px)
    if args.object_mask_erode_px > 0:
        print(
            f"[info] {camera}: object points from the mask eroded with a {args.object_mask_erode_px} px kernel "
            f"({int(points_mask.sum())} / {int(mask0.sum())} pixels)"
        )
    object_pts0 = masked_depth_to_points(depth_m[0], points_mask, K)
    near_object = cv2.dilate(mask0.astype(np.uint8), np.ones((SCENE_EXCLUDE_DILATE_PX,) * 2, np.uint8)).astype(bool)
    scene_pts0 = masked_depth_to_points(depth_m[0], ~near_object & (depth_m[0] > 0), K)
    return CameraObservation(
        camera=camera, K=K, rgb=rgb, depth_m=depth_m, mask0=mask0, object_pts0=object_pts0, scene_pts0=scene_pts0
    )


def _log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


def _load_camera_extrinsics_yaml(path: pathlib.Path, camera: str, camera_model_id: str) -> np.ndarray:
    """Load robot-base T camera (4x4) from an Example 4 extrinsics YAML."""
    import yaml

    assert path.is_file(), f"Extrinsics YAML not found: {path}"
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"Extrinsics YAML must be a mapping, got {type(payload)}"
    assert "camera" in payload, f"Extrinsics YAML missing 'camera': {path}"
    assert payload["camera"] == camera, (
        f"Extrinsics YAML camera {payload['camera']!r} != camera {camera!r} "
        "(--cameras and --extrinsics-paths must be in the same order)"
    )
    assert "camera_model_id" in payload, f"Extrinsics YAML missing 'camera_model_id': {path}"
    assert (
        payload["camera_model_id"] == camera_model_id
    ), f"Extrinsics YAML camera_model_id {payload['camera_model_id']!r} != camera_model_id {camera_model_id!r}"
    assert "matrix" in payload, f"Extrinsics YAML missing 'matrix': {path}"
    T = np.asarray(payload["matrix"], dtype=np.float64)
    assert T.shape == (4, 4), f"Extrinsics matrix must be 4x4, got {T.shape}"
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]), f"Bad homogeneous row: {T[3]}"
    return T


def _draw_pose_axes(color_rgb: np.ndarray, pose_cam: np.ndarray, K: np.ndarray, axis_len: float = 0.08) -> np.ndarray:
    """Lightweight RGB overlay of XYZ axes at `pose_cam` (no FoundationPose deps on the host)."""
    from r2st.geometry import project_axes_to_image

    pts = project_axes_to_image(pose_cam, K, axis_len=axis_len)
    img = color_rgb.copy()
    if pts is None:
        return img
    origin, x_pt, y_pt, z_pt = pts
    cv2.line(img, origin, x_pt, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, origin, y_pt, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, origin, z_pt, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main(args: Args) -> None:
    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert len(args.object_description) > 0, "object_description must not be empty"
    assert args.est_refine_iter >= 1, f"est_refine_iter must be >= 1, got {args.est_refine_iter}"
    assert args.track_refine_iter >= 1, f"track_refine_iter must be >= 1, got {args.track_refine_iter}"
    assert args.use_2d_tracker or not args.use_kalman_filter, (
        "use_kalman_filter requires use_2d_tracker: the filter fuses the 2D tracker's "
        "image-plane measurement with FoundationPose's pose estimate each frame."
    )
    assert args.max_frames is None or args.max_frames >= 1, f"max_frames must be >= 1, got {args.max_frames}"
    n_cameras = len(args.cameras)
    assert n_cameras >= 1, "--cameras must not be empty"
    assert len(set(args.cameras)) == n_cameras, f"--cameras has duplicates: {args.cameras}"
    assert (
        len(args.camera_model_ids) == n_cameras
    ), f"--camera-model-ids {args.camera_model_ids} must have one entry per --cameras {args.cameras}"
    if n_cameras > 1:
        assert args.extrinsics_paths is not None, (
            "--extrinsics-paths is required with more than one camera: their point clouds are merged in the "
            "robot-base frame"
        )
        assert args.scaled_mesh_path is None, (
            "--scaled-mesh-path is used as-is (no scaling), and extra cameras are only used for scaling. Pass one "
            "camera."
        )
    assert (
        args.scaled_mesh_path is None or args.unscaled_mesh_path is None
    ), "Pass at most one of --scaled-mesh-path and --unscaled-mesh-path"
    for flag, path in (
        ("--scaled-mesh-path", args.scaled_mesh_path),
        ("--unscaled-mesh-path", args.unscaled_mesh_path),
    ):
        if path is not None:
            assert path.is_file(), f"{flag} not found: {path}"
            assert path.suffix == ".glb", f"{flag} must be a .glb, got {path}"
    given_mesh_path = args.scaled_mesh_path if args.scaled_mesh_path is not None else args.unscaled_mesh_path

    workspace_bounds = None
    assert (
        args.crop_pointcloud_to_workspace or args.workspace_id is None
    ), "--workspace-id is only used by --crop-pointcloud-to-workspace, which is off"
    if args.crop_pointcloud_to_workspace:
        assert args.workspace_id is not None, "--crop-pointcloud-to-workspace requires --workspace-id"
        assert (
            args.extrinsics_paths is not None
        ), "--crop-pointcloud-to-workspace needs --extrinsics-paths: the workspace bounds are in the robot-base frame"
        workspace_bounds = get_workspace_bounds(args.workspace_id)

    T_world_cams = None
    if args.extrinsics_paths is not None:
        assert (
            len(args.extrinsics_paths) == n_cameras
        ), f"--extrinsics-paths {args.extrinsics_paths} must have one entry per --cameras {args.cameras}"
        T_world_cams = []
        for path, camera, camera_model_id in zip(
            args.extrinsics_paths, args.cameras, args.camera_model_ids, strict=True
        ):
            T_world_cams.append(_load_camera_extrinsics_yaml(path, camera, camera_model_id))
            print(f"[info] Loaded T_world_cam for {camera} from {path}:\n{T_world_cams[-1]}")

    print("[info] Loading SAM 3 ...")
    t0 = time.perf_counter()
    predictor = SAM3Predictor()
    _log_elapsed("SAM 3 loaded", t0)
    observations = [
        _observe_camera(args, predictor, camera, camera_model_id)
        for camera, camera_model_id in zip(args.cameras, args.camera_model_ids, strict=True)
    ]

    # The first camera is the one tracked
    tracked = observations[0]
    camera = tracked.camera
    T_world_cam = None if T_world_cams is None else T_world_cams[0]
    rgb_all, depth_m_all, K, mask = tracked.rgb, tracked.depth_m, tracked.K, tracked.mask0
    num_frames, H, W = rgb_all.shape[:3]
    rgb0, depth0 = rgb_all[0], depth_m_all[0]
    image_bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)

    object_slug = args.object_description.replace(" ", "_") if given_mesh_path is None else given_mesh_path.stem
    asset_dir = args.output_dir / f"{object_slug}__{camera}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    masked_cropped_path = ImageUtils.save_masked_debug(image_bgr0, mask, asset_dir, object_slug)
    for obs in observations[1:]:
        bgr0 = cv2.cvtColor(obs.rgb[0], cv2.COLOR_RGB2BGR)
        ImageUtils.save_masked_debug(bgr0, obs.mask0, asset_dir, f"{object_slug}__scaling_view_{obs.camera}")

    if args.scaled_mesh_path is None:
        if args.unscaled_mesh_path is None:
            # Same path examples/generate_mesh.py writes for this object and --camera-name, so its mesh is reused
            glb_path = asset_dir / f"{object_slug}.unscaled.glb"
            print(f"[info] Generating mesh with Meshy (reuses {glb_path} if present) ...")
            glb_path = MeshUtils.generate_with_meshy([masked_cropped_path], glb_path)
            assert glb_path.is_file(), f"Mesh file not created at {glb_path}"
            print(f"[info] Mesh: {glb_path.resolve()} ({glb_path.stat().st_size} bytes)")
        else:
            glb_path = args.unscaled_mesh_path
            print(f"[info] Using --unscaled-mesh-path (no Meshy, PCA-scaled below): {glb_path.resolve()}")

        # Merge every camera's frame-0 object points. PCA eigenvalues are pose-invariant, so any common frame works:
        # the robot base when extrinsics are given, else the (single) camera's optical frame.
        if T_world_cams is None:
            scaling_pts, scene_pts = tracked.object_pts0, tracked.scene_pts0
        else:
            per_camera = [
                obs.object_pts0 @ T[:3, :3].T + T[:3, 3] for obs, T in zip(observations, T_world_cams, strict=True)
            ]
            for obs, pts in zip(observations, per_camera, strict=True):
                print(
                    f"[info] {obs.camera}: {len(pts)} object points, centroid (robot base) {np.round(pts.mean(0), 4)}"
                )
            scaling_pts = np.concatenate(per_camera, axis=0)
            scene_pts = np.concatenate(
                [obs.scene_pts0 @ T[:3, :3].T + T[:3, 3] for obs, T in zip(observations, T_world_cams, strict=True)],
                axis=0,
            )
        if args.object_pointcloud_processing == "floor-points+5mm-voxel":
            scaling_pts, summary = floor_points_and_voxelize(scaling_pts, scene_pts)
            print(f"[info] Object point cloud processing ({args.object_pointcloud_processing}): {summary}")
        scaling_pts_path = asset_dir / f"{object_slug}__scaling_points.ply"
        trimesh.PointCloud(scaling_pts).export(scaling_pts_path)
        print(f"[info] Saved the merged scaling point cloud ({len(scaling_pts)} points) to {scaling_pts_path}")

        print(f"[info] Scaling mesh to the masked depth point cloud of {args.cameras} (PCA) ...")
        t0 = time.perf_counter()
        scaled_glb_path = asset_dir / f"{object_slug}.scaled.glb"
        scaled_glb_path, scale = scale_glb_to_pointcloud(glb_path, scaling_pts, scaled_glb_path)
        _log_elapsed(
            f"PCA scale={scale:.6f}  scaling_pts={len(scaling_pts)}  scaled mesh: {scaled_glb_path.resolve()} "
            f"({scaled_glb_path.stat().st_size} bytes)",
            t0,
        )
    else:
        scaled_glb_path = args.scaled_mesh_path
        print(f"[info] Using --scaled-mesh-path as-is (no Meshy, no scaling): {scaled_glb_path.resolve()}")
    mesh_path_abs = str(scaled_glb_path.resolve())
    if args.gif:
        gif_path = asset_dir / f"{object_slug}__orbit.gif"
        gif_path = MeshUtils.save_orbit_gif(scaled_glb_path, gif_path)
        print(f"[info] Saved orbit GIF to {gif_path}")

    vis = None
    if args.visualize:
        vis = TrackingVisualizer(
            scaled_glb_path,
            rgb_all,
            depth_m_all,
            K,
            mesh_position=tracked.object_pts0.mean(axis=0),
            camera_name=camera,
            T_world_cam=T_world_cam,
            object_points=tracked.object_pts0,
            workspace_bounds=workspace_bounds,
            extra_cameras=[
                VisualizerCamera(obs.camera, obs.rgb, obs.depth_m, obs.K, T, obs.object_pts0)
                for obs, T in zip(observations[1:], ([] if T_world_cams is None else T_world_cams[1:]), strict=True)
            ],
        )

    print(f"[info] Connecting to FoundationPose server at {args.server_address} ...")
    client = FoundationPoseClient(args.server_address)
    poses = np.zeros((num_frames, 4, 4), dtype=np.float64)

    print(
        f"[info] Registering object on frame 0 (gRPC; first call can take a while; "
        f"use_2d_tracker={args.use_2d_tracker} use_kalman_filter={args.use_kalman_filter}) ..."
    )
    t0 = time.perf_counter()
    poses[0] = client.register(
        mesh_path=mesh_path_abs,
        color_rgb=rgb0,
        depth_m=depth0,
        mask=mask,
        K=K,
        iteration=args.est_refine_iter,
        use_2d_tracker=args.use_2d_tracker,
        use_kalman_filter=args.use_kalman_filter,
    )
    _log_elapsed(f"Registered pose_cam translation: {poses[0][:3, 3]}", t0)

    print(f"[info] Tracking frames 1..{num_frames - 1} ...")
    t0 = time.perf_counter()
    for i in tqdm(range(1, num_frames), desc="track"):
        poses[i] = client.track(
            color_rgb=rgb_all[i],
            depth_m=depth_m_all[i],
            K=K,
            iteration=args.track_refine_iter,
        )
    client.close()
    _log_elapsed(f"Tracked {num_frames - 1} frames", t0)

    poses_path = asset_dir / f"{object_slug}__poses_cam.npy"
    np.save(poses_path, poses)
    print(f"[info] Saved camera-frame poses to {poses_path} shape={poses.shape}")
    if T_world_cam is not None:
        poses_world = np.stack([transform_pose_cam_to_world(T_world_cam, T_cam) for T_cam in poses], axis=0)
        poses_world_path = asset_dir / f"{object_slug}__poses_world.npy"
        np.save(poses_world_path, poses_world)
        print(f"[info] Saved robot-base poses to {poses_world_path} shape={poses_world.shape}")

    if args.save_video:
        video_path = asset_dir / f"{object_slug}__track.mp4"
        print(f"[info] Writing tracking video ({num_frames} frames) to {video_path} ...")
        t0 = time.perf_counter()
        writer = H264VideoWriter(video_path, 15.0, W, H)
        for i in tqdm(range(num_frames), desc="write video"):
            overlay = _draw_pose_axes(rgb_all[i], poses[i], K)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        writer.close()
        _log_elapsed(f"Saved tracking video to {video_path}", t0)

    print("Tracking complete!")
    print(f"Object: {args.object_description}")
    print(f"Frames: {num_frames}")
    print(f"Mesh: {scaled_glb_path}")
    print(f"Poses: {poses_path}")
    if T_world_cam is not None:
        print(f"Poses (robot base): {poses_world_path}")

    if vis is not None:
        vis.set_poses(poses)
        vis.wait()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
