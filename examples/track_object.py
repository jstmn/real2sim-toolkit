import dataclasses
import pathlib
import shutil
import time

import cv2
import h5py
import numpy as np
import tyro
from tqdm import tqdm

from r2st.constants import get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics
from r2st.core import GroundedSAMPredictor
from r2st.geometry import (
    reproject_depth_to_color_frame,
    depth_mm_to_meters,
    masked_depth_to_points,
    scale_intrinsics,
)
from r2st.mesh_scaling import scale_glb_to_pointcloud
from r2st.pose_grpc.client import FoundationPoseClient
from r2st.utils import (
    ImageUtils,
    MeshUtils,
    TrackingVisualizer,
    validate_merged_camera_group,
)

"""
# Example usage (FoundationPose gRPC server must already be running in the container):
    uv run python examples/track_object.py \
        --h5-path data/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
        --camera cam_1 \
        --camera-model-id d435 \
        --object-description "mustard bottle" \
        --visualize
"""


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see examples/merge_camera_streams.py)."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    camera_model_id: str
    """Camera model id for calibration lookup, e.g. 'd435'. See the README camera-model table."""

    object_description: str
    """Language description of the object to segment and mesh, e.g. 'mustard bottle'."""

    output_dir: pathlib.Path = pathlib.Path("data/meshyai")
    """Directory to save generated mesh assets and tracking outputs."""

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


def _load_rgb_depth(h5_path: pathlib.Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        validate_merged_camera_group(f, h5_path, camera)
        group = f[f"obs/sensor_data/{camera}"]
        return group["rgb"][:], group["depth"][:]


def _log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


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
    assert args.max_frames is None or args.max_frames >= 1, f"max_frames must be >= 1, got {args.max_frames}"

    depth_intrinsics = get_depth_intrinsics(args.camera_model_id)
    color_intrinsics = get_color_intrinsics(args.camera_model_id)

    print(f"[info] Loading RGB-D from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb_all, depth_raw_all = _load_rgb_depth(args.h5_path, args.camera)
    num_frames = rgb_all.shape[0]
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
        rgb_all = rgb_all[:num_frames]
        depth_raw_all = depth_raw_all[:num_frames]
    assert rgb_all.ndim == 4 and rgb_all.shape[-1] == 3, f"Expected NxHxWx3 rgb, got {rgb_all.shape}"
    _log_elapsed(f"Loaded {num_frames} frames, resolution {rgb_all.shape[1:3]}", t0)

    H, W = rgb_all.shape[1], rgb_all.shape[2]
    K = scale_intrinsics(
        color_intrinsics.intrinsic_matrix,
        (color_intrinsics.height, color_intrinsics.width),
        (H, W),
    ).astype(np.float64)
    print(f"[info] Color K ({args.camera_model_id}):\n{K}")

    print(f"[info] Reprojecting depth into the color frame ({num_frames} frames) ...")
    t0 = time.perf_counter()
    R_dc, t_dc = get_depth_to_color_extrinsics(args.camera_model_id)
    depth_m_all = np.stack(
        [
            reproject_depth_to_color_frame(
                depth_mm_to_meters(depth_raw_all[i]),
                depth_intrinsics,
                color_intrinsics,
                R_dc,
                t_dc,
            )
            for i in tqdm(range(num_frames), desc="reproject depth")
        ],
        axis=0,
    )
    assert depth_m_all.shape == (num_frames, H, W), f"Bad aligned depth shape: {depth_m_all.shape}"
    _log_elapsed("Reprojected depth into the color frame", t0)

    rgb0 = rgb_all[0]
    depth0 = depth_m_all[0]
    image_bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)

    print("[info] Loading GroundedSAM (GroundingDINO + SAM weights) ...")
    t0 = time.perf_counter()
    predictor = GroundedSAMPredictor()
    _log_elapsed("GroundedSAM loaded", t0)
    print(f"[info] Segmenting '{args.object_description}' on frame 0 ...")
    t0 = time.perf_counter()
    mask = ImageUtils.get_sam_mask(predictor, image_bgr0, args.object_description)
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    assert mask.sum() > 0, f"Segmentation mask is empty for description '{args.object_description}'"
    _log_elapsed(f"Mask pixels: {int(mask.sum())} / {mask.size}", t0)

    object_slug = args.object_description.replace(" ", "_")
    asset_dir = args.output_dir / f"{args.camera}__{object_slug}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    masked_cropped_path = ImageUtils.save_masked_debug(image_bgr0, mask, asset_dir, object_slug)

    print("[info] Generating mesh with Meshy (reuses existing model_glb.glb if present) ...")
    glb_path = asset_dir / f"{object_slug}_glb.glb"
    meshy_result = MeshUtils.generate_with_meshy([masked_cropped_path], asset_dir)
    assert meshy_result.exists(), f"Mesh file not created at {meshy_result}"
    if meshy_result.resolve() != glb_path.resolve():
        shutil.copy2(str(meshy_result), str(glb_path))
    assert glb_path.is_file(), f"Mesh file not created at {glb_path}"
    print(f"[info] Mesh: {glb_path.resolve()} ({glb_path.stat().st_size} bytes)")

    print("[info] Scaling mesh to masked RealSense point cloud (PCA) ...")
    t0 = time.perf_counter()
    object_pts = masked_depth_to_points(depth0, mask, K)
    scaled_glb_path = asset_dir / f"{object_slug}_glb_scaled.glb"
    scaled_glb_path, scale = scale_glb_to_pointcloud(glb_path, object_pts, scaled_glb_path)
    mesh_path_abs = str(scaled_glb_path.resolve())
    _log_elapsed(
        f"PCA scale={scale:.6f}  object_pts={len(object_pts)}  scaled mesh: {mesh_path_abs} "
        f"({scaled_glb_path.stat().st_size} bytes)",
        t0,
    )
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
            mesh_position=object_pts.mean(axis=0),
        )

    print(f"[info] Connecting to FoundationPose server at {args.server_address} ...")
    client = FoundationPoseClient(args.server_address)
    poses = np.zeros((num_frames, 4, 4), dtype=np.float64)

    print("[info] Registering object on frame 0 (gRPC; first call can take a while) ...")
    t0 = time.perf_counter()
    poses[0] = client.register(
        mesh_path=mesh_path_abs,
        color_rgb=rgb0,
        depth_m=depth0,
        mask=mask,
        K=K,
        iteration=args.est_refine_iter,
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

    if args.save_video:
        video_path = asset_dir / f"{object_slug}__track.mp4"
        print(f"[info] Writing tracking video ({num_frames} frames) to {video_path} ...")
        t0 = time.perf_counter()
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            15.0,
            (W, H),
        )
        assert writer.isOpened(), f"Failed to open video writer at {video_path}"
        for i in tqdm(range(num_frames), desc="write video"):
            overlay = _draw_pose_axes(rgb_all[i], poses[i], K)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        writer.release()
        _log_elapsed(f"Saved tracking video to {video_path}", t0)

    print("Tracking complete!")
    print(f"Object: {args.object_description}")
    print(f"Frames: {num_frames}")
    print(f"Mesh: {scaled_glb_path}")
    print(f"Poses: {poses_path}")

    if vis is not None:
        vis.set_poses(poses)
        vis.wait()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
