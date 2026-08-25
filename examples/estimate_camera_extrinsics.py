"""
This script does the following:
1. Loads the demonstration data from the h5 file
2. Loads the robot from Jrl2
3. Runs an optimization procedure to estimate the camera extrinsics.
4. Saves the estimated camera extrinsics to a yaml file.

At a high level, the optimization procedure runs CMA-ES to estimate the SE(3) pose (in the robot's base frame) of the
specified camera. There are two pointclouds that we care about: pcd_real and pcd_sim. pcd_real is the measured
pointcloud from the camera (found by projecting the depth points with the camera's intrinsic matrix). pcd_sim is a
6-view render of the robot (cameras on +X/-X/+Y/-Y/+Z/-Z looking at the workspace), fused in the robot-base frame and
FPS-downsampled once per sampled joint configuration, then mapped into the candidate camera with T_world_cam. The cost
is the one-sided squared nearest-neighbor distance from pcd_real to pcd_sim.

Notes:
1. A preprocessing step is performed on the measured pointclouds to remove points that aren't part of the robot.
    Frame 0 is SAM 3 (union of top --sam-kmax masks). That union is then propagated through every later
    RGB frame with SAM 3 ``mask_input`` + the previous mask's bbox (text prompt only on frame 0).
    After unprojection, Open3D ``remove_radius_outlier`` drops points with fewer than
    ``--radius-outlier-nb-points`` neighbors in ``--radius-outlier-radius-m`` (default 10 in 10 cm).
2. If --visualize is set, a viser server is started. The server shows measured vs best-so-far
    simulated pointclouds in the robot-base frame, a camera frustum at the estimated pose, and a
    plot of lowest population cost vs CMA-ES iteration.
3. if --visualize-robot-masks is set, debug PNGs are written for the frame-0 SAM masks that enter the union.
4. If --cache-robot-masks is set (default), each propagated mask is written as soon as it is computed to
   `<h5_dir>/<h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy`.
    A later run loads a consecutive prefix of those files and resumes SAM from the last cached frame.
    Frame 0's SAM 3 union may also be read from the older
    `robot-mask__<camera>__idx=0__...npy` seed cache. After the last frame, a dimmed-mask video is written to
    `robot-mask-propagated__<camera>__kmax=<k>__score_threshold=<t>.mp4`.



The pseudo-code is as follows:

inputs:
- rgbds: array of RGBD images
- all_joint_angles: array of joint angles from the demonstration. Same length as rgbds; frame i of qpos is frame i of RGB-D.
- N: the number of timesteps to sample from the demonstration for the cost function
- rgb_to_pcd(rgbd): unprojects color-aligned depth with color K, then keeps SAM robot pixels
- S: number of poses in the CMA-ES population
- pcd_sim_world[t]: 6-view rendered robot cloud at joint_angles[t], FPS-downsampled, in the robot-base frame
- camera_from_world(pose, pcd_world): maps world points into the OpenCV camera frame
- compute_chamfer_distance(pcd_real, pcd_sim): one-sided mean min squared distance from pcd_real to pcd_sim

# Find the N joint angles that are the furthest apart from each other
timesteps, joint_angles = furthest_point_sample(all_joint_angles, N, distance='circular')

pcd_reals = [rgb_to_pcd(rgbd) for rgbd in rgbds]

until convergence:
    sample poses pi, ..., pN from CMA-ES
    total_costs = [0] * S
    for each pose pi in population:
        total_cost = 0
        for joint_angle, pcd_real, pcd_sim_world in zip(joint_angles, pcd_reals, pcd_sim_worlds):
            pcd_sim = camera_from_world(pose, pcd_sim_world)
            cost = compute_chamfer_distance(pcd_real, pcd_sim)
            total_cost += cost
        total_costs[i] = total_cost
    update CMA-ES with total_costs as fitness values

output:
- estimated pose
"""

from __future__ import annotations

import dataclasses
import pathlib
import threading
import time
from typing import Any, Literal

import cv2
import h5py
import numpy as np
import sapien
import tyro
import yaml
from jrl2.robots import NAME_TO_ROBOT, get_robot_by_name
from tqdm import tqdm
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from r2st.constants import (
    MIN_DEPTH_M,
    get_color_intrinsics,
    get_depth_intrinsics,
    get_depth_to_color_extrinsics,
)
from r2st.core import (
    SAM3Predictor,
    bbox_xyxy_from_mask,
    binary_mask_to_sam_mask_input,
)
from r2st.geometry import (
    camera_extrinsic_to_maniskill_pose,
    depth_mm_to_meters,
    masked_depth_to_points,
    one_sided_squared_nn_distance,
    reproject_depth_to_color_frame,
)
from r2st.utils import (
    ImageUtils,
    MeshUtils,
    farthest_point_sample_naive,
    farthest_point_sample_pyg_lib,
    validate_merged_camera_group,
)

"""
# Example usage:
uv run python examples/estimate_camera_extrinsics.py \
    --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
    --robot-id xarm7__gripper --camera cam_1 --camera-model-id d435 \
    --depth-intrinsics-source rgb \
    --output-path data/demonstrations/0802/extrinsics.yaml \
    --seed-automatically \
    --visualize --visualize-robot-masks
"""

_SEED_RADIUS_MIN_M = 1.0
_SEED_RADIUS_MAX_M = 1.5
_SEED_LOOKAT_TARGET = np.array([0.0, 0.0, 0.5], dtype=np.float64)
_SEED_ORIGIN = np.zeros(3, dtype=np.float64)
_SEED_MIN_LOOK_DIST_M = 0.05
_SEED_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_CAPTURE_AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_CAPTURE_DISTANCE_M = 1.0
_CAPTURE_FOV_DEG = 90.0
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 480
_CAPTURE_FAR_M = 5.0
_CAPTURE_CAMERA_NAME = "multiview_cam"
_CAPTURE_UP_WHEN_PARALLEL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_MASK_VIDEO_FPS = 15.0
_BEST_AXES_LENGTH = 0.08
_BEST_AXES_RADIUS = 0.004
_BEST_ORIGIN_RADIUS = 0.008
_POP_AXES_LENGTH = 0.045
_POP_AXES_RADIUS = 0.0025
_POP_ORIGIN_RADIUS = 0.005


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see examples/merge_camera_streams.py)."""

    robot_id: str
    """Jrl2 robot name, e.g. 'xarm7__gripper'. See the README `{robot}__{eef}` table."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    camera_model_id: str
    """Camera model id for calibration lookup, e.g. 'd435'. See the README camera-model table."""

    depth_intrinsics_source: Literal["rgb", "depth"]
    """Which camera-model K the h5 depth channel uses. `rgb`: color-aligned depth, unproject with color K. `depth`: native depth image, reproject into the color frame first."""

    output_path: pathlib.Path
    """YAML path to write the estimated 4x4 camera extrinsics (robot-base T camera)."""

    n_timesteps: int = 5
    """Number of furthest-apart joint configurations used in the chamfer cost."""

    n_pcd_samples_real: int = 4096
    """Number of FPS points kept from each measured robot cloud."""

    n_pcd_samples_sim: int = 4096
    """Number of FPS points kept from each fused 6-view sim robot cloud."""

    n_random_downsample_initial: int = 16384
    """Random pre-FPS cap for each measured robot cloud. Must be >= n_pcd_samples_real."""

    cma_sigma_pos: float = 0.10
    """CMA-ES initial std for camera translation (meters)."""

    cma_sigma_rot: float = 1.0
    """CMA-ES initial std for camera rotation-vector (radians)."""

    cma_maxiter: int = 100
    """Maximum CMA-ES generations."""

    seed_automatically: bool = False
    """If set, choose the CMA-ES seed with the spherical-grid search."""

    seed_from_gui: bool = False
    """If set, choose the CMA-ES seed interactively with Viser GUI controls."""

    seed_pose: tuple[float, float, float, float, float, float, float] | None = None
    """If set, seed CMA-ES with this robot-base camera pose: x y z qw qx qy qz."""

    gui_translation_step_m: float = 0.02
    """World-frame translation per GUI button press, in meters."""

    gui_rotation_step_deg: float = 5.0
    """World-frame rotation per GUI button press, in degrees."""

    n_seed_azimuth: int = 25
    """Number of azimuth steps (about world +Z) for seed camera positions on each sphere."""

    n_seed_polar: int = 25
    """Number of polar-angle steps (from world +Z) for seed camera positions on each sphere."""

    n_seed_radii: int = 3
    """Number of sphere radii, linspace from 1.0 m to 1.5 m inclusive."""

    n_seed_rolls: int = 10
    """Number of evenly spaced rolls about the look-at axis (toward (0, 0, 0.5)) per seed position."""

    cma_popsize: int = 10
    """CMA-ES population size. If unset, the cma library default is used."""

    robot_description: str = "robot arm"
    """SAM 3 text prompt used to mask the robot on frame 0. Later frames propagate that mask."""

    sam_kmax: int = 5
    """Union at most this many highest-confidence SAM masks into the frame-0 robot mask."""

    sam_score_threshold: float = 0.3
    """Keep a top-k SAM mask in the frame-0 union only if its confidence is strictly above this."""

    cache_robot_masks: bool = True
    """If set, load/save propagated robot masks as <h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy."""

    mask_erode_px: int = 9
    """Erode the robot mask with an elliptical kernel of this size (pixels) before unprojecting. 0 skips erosion."""

    radius_outlier_nb_points: int = 10
    """Keep a measured point only if this many points lie within radius_outlier_radius_m of it."""

    radius_outlier_radius_m: float = 0.10
    """Search radius (meters) for the measured-cloud radius-outlier filter."""

    visualize: bool = False
    """If set, start a viser server with robot-base pointclouds, camera frustum, and a cost plot."""

    visualize_robot_masks: bool = False
    """If set, write debug PNGs for the frame-0 SAM masks that enter the union (top --sam-kmax with score above --sam-score-threshold)."""


def _log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


def _validate_seed_mode(
    seed_automatically: bool,
    seed_from_gui: bool,
    seed_pose: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    n_modes = int(seed_automatically) + int(seed_from_gui) + int(seed_pose is not None)
    assert n_modes == 1, "Exactly one of --seed-automatically, --seed-from-gui, or --seed-pose must be passed"


def _robot_mask_cache_path(
    h5_path: pathlib.Path,
    camera: str,
    frame_idx: int,
    kmax: int,
    score_threshold: float,
) -> pathlib.Path:
    return (
        h5_path.parent
        / h5_path.stem
        / f"robot-mask__{camera}__idx={int(frame_idx)}__kmax={int(kmax)}__score_threshold={score_threshold}.npy"
    )


def _propagated_robot_mask_cache_path(
    h5_path: pathlib.Path,
    camera: str,
    frame_idx: int,
    kmax: int,
    score_threshold: float,
) -> pathlib.Path:
    return (
        h5_path.parent
        / h5_path.stem
        / f"robot-mask-propagated__{camera}__idx={int(frame_idx)}__kmax={int(kmax)}__score_threshold={score_threshold}.npy"
    )


def _propagated_robot_mask_video_path(
    h5_path: pathlib.Path,
    camera: str,
    kmax: int,
    score_threshold: float,
) -> pathlib.Path:
    return (
        h5_path.parent
        / h5_path.stem
        / f"robot-mask-propagated__{camera}__kmax={int(kmax)}__score_threshold={score_threshold}.mp4"
    )


def _load_cached_bool_mask(path: pathlib.Path, hw: tuple[int, int]) -> np.ndarray:
    mask = np.load(path)
    assert isinstance(mask, np.ndarray), f"{path}: expected ndarray, got {type(mask)}"
    assert mask.shape == hw, f"{path}: mask shape {mask.shape} != {hw}"
    assert mask.dtype == bool, f"{path}: mask dtype must be bool, got {mask.dtype}"
    assert mask.any(), f"{path}: cached robot mask is empty"
    return mask


def _n_consecutive_cached_masks(paths: list[pathlib.Path]) -> int:
    """Count how many leading cache files exist (0 if the first is missing)."""
    n = 0
    for path in paths:
        if not path.is_file():
            break
        n += 1
    return n


def _save_cached_bool_mask(path: pathlib.Path, mask: np.ndarray) -> None:
    assert mask.dtype == bool, f"{path}: mask dtype must be bool, got {mask.dtype}"
    assert mask.any(), f"{path}: cannot cache an empty robot mask"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)


def _load_sam3() -> SAM3Predictor:
    t0 = time.perf_counter()
    predictor = SAM3Predictor()
    _log_elapsed("SAM 3 loaded", t0)
    return predictor


def _propagate_robot_masks(
    rgb: np.ndarray,
    predictor: SAM3Predictor,
    seed_mask: np.ndarray,
    cache_paths: list[pathlib.Path] | None = None,
) -> np.ndarray:
    """Propagate ``seed_mask`` (frame 0) through ``rgb`` with SAM 3 mask_input + bbox."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert n_frames >= 1, f"rgb must contain at least 1 frame, got {rgb.shape}"
    assert seed_mask.shape == (height, width), f"seed_mask {seed_mask.shape} != image {(height, width)}"
    assert seed_mask.dtype == bool, f"seed_mask dtype must be bool, got {seed_mask.dtype}"
    assert seed_mask.any(), "seed_mask is empty"
    if cache_paths is not None:
        assert len(cache_paths) == n_frames, f"cache_paths length {len(cache_paths)} != n_frames {n_frames}"
    masks = np.zeros((n_frames, height, width), dtype=bool)
    masks[0] = seed_mask
    if cache_paths is not None:
        _save_cached_bool_mask(cache_paths[0], seed_mask)
    mask_input = binary_mask_to_sam_mask_input(seed_mask)
    prev_mask = seed_mask
    for t in tqdm(range(1, n_frames), desc="Propagate robot masks"):
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        box = bbox_xyxy_from_mask(prev_mask)
        mask, _score, low_res = predictor.propagate_from_mask(image_bgr, mask_input, box)
        assert mask.shape == (height, width), f"frame {t}: mask {mask.shape} != {(height, width)}"
        masks[t] = mask
        if cache_paths is not None:
            _save_cached_bool_mask(cache_paths[t], mask)
        mask_input = low_res
        prev_mask = mask
    return masks


def _compute_propagated_robot_masks(
    rgb: np.ndarray,
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    sam_kmax: int,
    sam_score_threshold: float,
    cache_robot_masks: bool,
    visualize_robot_masks: bool,
) -> np.ndarray:
    """Return ``(T, H, W)`` robot masks. Frame 0 is SAM 3; later frames are SAM 3-propagated."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert n_frames >= 1, f"rgb must contain at least 1 frame, got {rgb.shape}"
    hw = (height, width)
    propagated_paths = [
        _propagated_robot_mask_cache_path(h5_path, camera, t, sam_kmax, sam_score_threshold) for t in range(n_frames)
    ]
    n_cached = _n_consecutive_cached_masks(propagated_paths) if cache_robot_masks else 0
    if n_cached == n_frames:
        masks = np.stack([_load_cached_bool_mask(path, hw) for path in propagated_paths], axis=0)
        print(f"[info] Loaded {n_frames} propagated robot masks from cache")
    else:
        masks = np.zeros((n_frames, height, width), dtype=bool)
        cache_paths = propagated_paths if cache_robot_masks else None
        if n_cached >= 1:
            for t in range(n_cached):
                masks[t] = _load_cached_bool_mask(propagated_paths[t], hw)
            print(f"[info] Resuming robot-mask propagation from frame {n_cached} ({n_cached}/{n_frames} cached)")
            predictor = _load_sam3()
            rest = _propagate_robot_masks(
                rgb[n_cached - 1 :],
                predictor,
                masks[n_cached - 1],
                cache_paths=None if cache_paths is None else cache_paths[n_cached - 1 :],
            )
            assert rest.shape[0] == n_frames - n_cached + 1, f"rest frames {rest.shape[0]} != {n_frames - n_cached + 1}"
            masks[n_cached:] = rest[1:]
        else:
            seed_path = _robot_mask_cache_path(h5_path, camera, 0, sam_kmax, sam_score_threshold)
            predictor: SAM3Predictor | None = None
            if cache_robot_masks and seed_path.is_file():
                seed_mask = _load_cached_bool_mask(seed_path, hw)
                print(f"[info] Loaded frame-0 robot mask from {seed_path}")
            else:
                predictor = _load_sam3()
                image_bgr0 = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR)
                print(f"[info] Segmenting '{robot_description}' on frame 0 ...")
                t0 = time.perf_counter()
                ranked_masks, scores, phrases = ImageUtils.get_sam_masks_ranked(
                    predictor, image_bgr0, robot_description
                )
                _log_elapsed(f"Segmented '{robot_description}' on frame 0 ({ranked_masks.shape[0]} masks)", t0)
                for rank in range(min(sam_kmax, ranked_masks.shape[0])):
                    print(
                        f"[info]   rank={rank} score={scores[rank]:.4f} phrase={phrases[rank]!r} "
                        f"pixels={int(ranked_masks[rank].sum())}"
                    )
                seed_mask = _union_top_sam_masks(ranked_masks, scores, sam_kmax, sam_score_threshold)
                if visualize_robot_masks:
                    _dump_top_sam_masks(
                        image_bgr0,
                        ranked_masks,
                        scores,
                        phrases,
                        h5_path.parent,
                        camera,
                        0,
                        sam_kmax,
                        sam_score_threshold,
                    )
                if cache_robot_masks:
                    _save_cached_bool_mask(seed_path, seed_mask)
                    print(f"[info] Saved frame-0 robot mask to {seed_path}")

            if n_frames == 1:
                masks = seed_mask[None, ...]
                if cache_paths is not None:
                    _save_cached_bool_mask(cache_paths[0], seed_mask)
            else:
                if predictor is None:
                    predictor = _load_sam3()
                print(f"[info] Propagating frame-0 robot mask through {n_frames} frames ...")
                t0 = time.perf_counter()
                masks = _propagate_robot_masks(rgb, predictor, seed_mask, cache_paths=cache_paths)
                _log_elapsed(f"Propagated robot mask through {n_frames} frames", t0)
        assert masks.shape == (n_frames, height, width), f"masks {masks.shape} != {(n_frames, height, width)}"
        if cache_robot_masks:
            print(f"[info] Saved {n_frames} propagated robot masks")
    _save_propagated_mask_demo_overlays(rgb, masks, h5_path, camera, sam_kmax, sam_score_threshold)
    return masks


def _save_propagated_mask_demo_overlays(
    rgb: np.ndarray,
    masks: np.ndarray,
    h5_path: pathlib.Path,
    camera: str,
    kmax: int,
    score_threshold: float,
) -> None:
    """Write a dimmed-mask demo PNG for every trajectory frame and an mp4 of those overlays."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert masks.shape == (n_frames, height, width), f"masks {masks.shape} != {(n_frames, height, width)}"
    video_path = _propagated_robot_mask_video_path(h5_path, camera, kmax, score_threshold)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _MASK_VIDEO_FPS,
        (width, height),
    )
    assert writer.isOpened(), f"Failed to open video writer at {video_path}"
    for t in range(n_frames):
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        overlay_path = _propagated_robot_mask_cache_path(h5_path, camera, t, kmax, score_threshold).with_suffix(
            ".demo.png"
        )
        demo = ImageUtils.demo_overlay(image_bgr, masks[t])
        assert overlay_path.name.endswith(".demo.png"), f"demo overlay path must end with .demo.png, got {overlay_path}"
        assert cv2.imwrite(str(overlay_path), demo), f"Failed to write {overlay_path}"
        writer.write(demo)
    writer.release()
    print(f"[info] Saved {n_frames} robot-mask demo overlays and video to {video_path}")


def _union_top_sam_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    kmax: int,
    score_threshold: float,
) -> np.ndarray:
    """OR-union of the first `kmax` ranked masks whose score is strictly above `score_threshold`."""
    assert masks.ndim == 3, f"masks must be (N, H, W), got {masks.shape}"
    assert scores.shape == (masks.shape[0],), f"scores {scores.shape} != n_masks {masks.shape[0]}"
    assert kmax >= 1, f"kmax must be >= 1, got {kmax}"
    n = min(int(kmax), masks.shape[0])
    keep = scores[:n] > score_threshold
    n_keep = int(keep.sum())
    assert n_keep >= 1, f"No SAM masks in top {n} with score > {score_threshold} " f"(scores={scores[:n].tolist()})"
    union = np.any(masks[:n][keep], axis=0)
    print(
        f"[info] Unioned {n_keep}/{n} masks with score > {score_threshold} "
        f"(kept ranks={np.flatnonzero(keep).tolist()}, scores={scores[:n][keep].tolist()})"
    )
    return union


def _erode_mask(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    """Erode a bool mask with an elliptical kernel of size `kernel_px`. `kernel_px=0` is a no-op."""
    assert mask.ndim == 2, f"mask must be 2D, got {mask.shape}"
    assert mask.dtype == bool, f"mask dtype must be bool, got {mask.dtype}"
    assert mask.any(), "Cannot erode an empty mask"
    assert kernel_px == 0 or (
        kernel_px >= 1 and kernel_px % 2 == 1
    ), f"mask_erode_px must be 0 or a positive odd int, got {kernel_px}"
    if kernel_px == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    assert eroded.any(), f"Mask empty after erode with kernel_px={kernel_px} (had {int(mask.sum())} pixels)"
    return eroded


def _dump_top_sam_masks(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    phrases: list[str],
    out_dir: pathlib.Path,
    camera: str,
    frame_idx: int,
    kmax: int,
    score_threshold: float,
) -> None:
    n = min(int(kmax), masks.shape[0])
    keep = scores[:n] > score_threshold
    ranks = np.flatnonzero(keep)
    assert ranks.size >= 1, (
        f"No SAM masks in top {n} with score > {score_threshold} for frame {frame_idx} "
        f"(scores={scores[:n].tolist()})"
    )
    for rank in ranks:
        rank = int(rank)
        mask = masks[rank]
        print(
            f"[info] frame {frame_idx} rank={rank}/{n - 1} score={scores[rank]:.4f} "
            f"phrase={phrases[rank]!r} pixels={int(mask.sum())}/{mask.size}"
        )
        if not mask.any():
            print(f"[warning] frame {frame_idx} rank={rank} mask is empty, skipping PNG")
            continue
        prefix = f"{camera}__robot_mask__{int(frame_idx)}__rank={rank}__score={scores[rank]:.3f}"
        ImageUtils.save_masked_debug(image_bgr, mask, out_dir, prefix)


def _load_demo(h5_path: pathlib.Path, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rgb, depth, and qpos, aligned by frame index (same N)."""
    with h5py.File(h5_path, "r") as f:
        validate_merged_camera_group(f, h5_path, camera)
        assert (
            "obs/qpos" in f
        ), f"{h5_path}: missing 'obs/qpos' (re-merge the demo with examples/merge_camera_streams.py)"
        group = f[f"obs/sensor_data/{camera}"]
        rgb = group["rgb"][:]
        depth = group["depth"][:]
        qpos = np.asarray(f["obs/qpos"][:])
    assert qpos.ndim == 2, f"{h5_path}: obs/qpos must be NxD, got {qpos.shape}"
    assert qpos.shape[0] > 0, f"{h5_path}: obs/qpos is empty"
    assert qpos.shape[0] == rgb.shape[0], (
        f"{h5_path}: obs/qpos has {qpos.shape[0]} frames, {camera} rgb has {rgb.shape[0]} "
        "(qpos and RGB-D must share the same index axis)"
    )
    return rgb, depth, qpos


def _expand_demo_qpos_to_joint_names(qpos: np.ndarray, robot, target_joint_names: list[str]) -> np.ndarray:
    """Map demo qpos onto ``target_joint_names`` (Sapien active joints).

    Demo h5 stores arm qpos only. Columns are a leading prefix of
    ``robot.actuated_joint_names``. Extra EEF joints are pinned at 0 (closed gripper
    for xarm7__gripper). Jrl2 counts mimic joints as actuated; Sapien does not, so
    unused Jrl2 joints must have a URDF mimic.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    assert qpos.ndim == 2, f"qpos must be (T, dof), got {qpos.shape}"
    source_names = list(robot.actuated_joint_names)
    target_joint_names = list(target_joint_names)
    n_demo = int(qpos.shape[1])
    n_T = int(qpos.shape[0])
    n_source = len(source_names)
    assert n_demo >= 1, f"obs/qpos dim must be >= 1, got {qpos.shape}"
    assert n_source >= 1, f"robot {robot.name!r} has no actuated joints"
    assert len(target_joint_names) >= 1, "target_joint_names is empty"
    assert n_demo <= n_source, f"obs/qpos dim {n_demo} > Jrl2 {robot.name} actuators {n_source} ({source_names})"
    name_to_joint = {j.name: j for j in robot.actuated_joints}
    assert len(name_to_joint) == n_source, f"Duplicate Jrl2 actuated joint names: {source_names}"
    extra_names = source_names[n_demo:]
    for name in extra_names:
        assert name in name_to_joint, f"actuated joint {name!r} missing from robot.actuated_joints"
        limit = name_to_joint[name].limit
        assert limit is not None, f"joint {name!r} has no limit; cannot pin at 0"
        assert (
            limit.lower <= 0.0 <= limit.upper
        ), f"Cannot pin EEF joint {name!r} at 0 (limits=[{limit.lower}, {limit.upper}])"
    named: dict[str, np.ndarray] = {name: qpos[:, i] for i, name in enumerate(source_names[:n_demo])}
    zeros = np.zeros(n_T, dtype=np.float64)
    for name in extra_names:
        named[name] = zeros
    missing = [name for name in target_joint_names if name not in named]
    assert not missing, f"Sapien joints {missing} are not in Jrl2 actuated joints {source_names}"
    unused = [name for name in source_names if name not in target_joint_names]
    for name in unused:
        assert name_to_joint[name].mimic is not None, (
            f"Jrl2 joint {name!r} is actuated and has no mimic, but is not a Sapien active joint "
            f"{target_joint_names}"
        )
    if extra_names:
        print(f"[info] Demo qpos has {n_demo} joints {source_names[:n_demo]}; pinning EEF joints {extra_names} to 0")
    if unused:
        print(f"[info] Sapien omits mimic joints {unused}; they follow their parent")
    out = np.stack([named[name] for name in target_joint_names], axis=1)
    assert out.shape == (n_T, len(target_joint_names)), f"expanded qpos shape {out.shape}"
    return out


def _look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """4x4 T_world_cam in OpenCV / RealSense optical convention (+Z forward, +Y down)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = np.asarray(up, dtype=np.float64).reshape(3)
    z_fwd = target - eye
    z_norm = np.linalg.norm(z_fwd)
    assert z_norm > 1e-8, f"eye and target are coincident: eye={eye} target={target}"
    z_fwd = z_fwd / z_norm
    x_right = np.cross(up, z_fwd)
    x_norm = np.linalg.norm(x_right)
    assert x_norm > 1e-8, f"up is parallel to look direction: up={up} z={z_fwd}"
    x_right = x_right / x_norm
    y_down = np.cross(z_fwd, x_right)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_right
    T[:3, 1] = y_down
    T[:3, 2] = z_fwd
    T[:3, 3] = eye
    return T


def _axis_aligned_capture_poses(lookat: np.ndarray, distance_m: float) -> list[np.ndarray]:
    """Six OpenCV T_world_cam poses on ±X/±Y/±Z looking at `lookat` from `distance_m`."""
    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)
    assert distance_m > 0.0, f"distance_m must be > 0, got {distance_m}"
    assert _CAPTURE_AXES.shape == (6, 3), f"_CAPTURE_AXES must be (6, 3), got {_CAPTURE_AXES.shape}"
    poses: list[np.ndarray] = []
    up_default = _SEED_UP / np.linalg.norm(_SEED_UP)
    for axis in _CAPTURE_AXES:
        axis_norm = float(np.linalg.norm(axis))
        assert axis_norm > 0.0, f"capture axis must be non-zero, got {axis}"
        axis_u = axis / axis_norm
        eye = lookat + float(distance_m) * axis_u
        look = lookat - eye
        look_n = float(np.linalg.norm(look))
        assert look_n >= _SEED_MIN_LOOK_DIST_M, f"capture look-at distance {look_n} < {_SEED_MIN_LOOK_DIST_M}"
        look_u = look / look_n
        up = up_default
        if abs(float(look_u @ up)) > (1.0 - 1e-6):
            up = _CAPTURE_UP_WHEN_PARALLEL
        poses.append(_look_at_opencv(eye, lookat, up))
    assert len(poses) == 6, f"expected 6 capture poses, got {len(poses)}"
    return poses


def _seed_eyes_on_sphere(n_azimuth: int, n_polar: int, n_radii: int) -> np.ndarray:
    """Fixed upper-hemisphere grid with camera-position z > 0 about the robot base."""
    assert n_azimuth >= 1, f"n_azimuth must be >= 1, got {n_azimuth}"
    assert n_polar >= 1, f"n_polar must be >= 1, got {n_polar}"
    assert n_radii >= 1, f"n_radii must be >= 1, got {n_radii}"
    radii = np.linspace(_SEED_RADIUS_MIN_M, _SEED_RADIUS_MAX_M, n_radii, dtype=np.float64)
    azimuths = (np.arange(n_azimuth, dtype=np.float64) / n_azimuth) * (2.0 * np.pi)
    polars = ((np.arange(n_polar, dtype=np.float64) + 0.5) / n_polar) * np.pi
    polars = polars[polars < np.pi / 2.0]
    assert polars.size >= 1, f"n_polar={n_polar} produces no seed positions with camera z > 0; increase n_polar"
    eyes = np.empty((n_radii * len(polars) * n_azimuth, 3), dtype=np.float64)
    k = 0
    for radius_m in radii:
        for theta in polars:
            st = float(np.sin(theta))
            ct = float(np.cos(theta))
            for phi in azimuths:
                eyes[k] = _SEED_ORIGIN + float(radius_m) * np.array(
                    [st * np.cos(phi), st * np.sin(phi), ct], dtype=np.float64
                )
                k += 1
    assert k == eyes.shape[0]
    assert np.all(eyes[:, 2] > 0.0), f"All seed camera positions must have z > 0, got min z={eyes[:, 2].min()}"
    look = _SEED_LOOKAT_TARGET[None, :] - eyes
    look_n = np.linalg.norm(look, axis=1)
    assert np.all(
        look_n >= _SEED_MIN_LOOK_DIST_M
    ), f"Seed sphere grid has look-at distance < {_SEED_MIN_LOOK_DIST_M} m: min={float(look_n.min())}"
    look_u = look / look_n[:, None]
    up_n = _SEED_UP / np.linalg.norm(_SEED_UP)
    parallel = np.abs(look_u @ up_n) > (1.0 - 1e-6)
    assert not np.any(
        parallel
    ), f"Seed sphere grid has look-at parallel to up at indices {np.flatnonzero(parallel).tolist()}"
    return eyes


def _look_at_with_roll(eye: np.ndarray, target: np.ndarray, roll_rad: float, up: np.ndarray) -> np.ndarray:
    """Look-at pose, then roll about the camera optical axis (+Z, the look-at direction)."""
    T = _look_at_opencv(eye, target, up)
    Rz = axangle2mat(np.array([0.0, 0.0, 1.0], dtype=np.float64), float(roll_rad))
    T_roll = T.copy()
    T_roll[:3, :3] = T[:3, :3] @ Rz
    return T_roll


def _pose_translation_rotation_text(T: np.ndarray) -> str:
    """Markdown lines for camera translation and rotation in the robot-base frame."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    t = T[:3, 3]
    q = mat2quat(T[:3, :3])
    rotvec_deg = np.rad2deg(_T_to_vec(T)[3:])
    return (
        f"**camera t (robot base):** `[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]`  \n"
        f"**camera R (wxyz):** `[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]`  \n"
        f"**camera rotvec (deg):** `[{rotvec_deg[0]:.2f}, {rotvec_deg[1]:.2f}, {rotvec_deg[2]:.2f}]`"
    )


def _step_pose_world(
    T_world_cam: np.ndarray,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
) -> np.ndarray:
    """Apply world-frame translation and rotation-vector increments to `T_world_cam`."""
    assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
    translation_world = np.asarray(translation_world, dtype=np.float64).reshape(3)
    rotation_world = np.asarray(rotation_world, dtype=np.float64).reshape(3)
    T_next = T_world_cam.copy()
    T_next[:3, 3] += translation_world
    angle = float(np.linalg.norm(rotation_world))
    if angle > 0.0:
        T_next[:3, :3] = axangle2mat(rotation_world / angle, angle) @ T_world_cam[:3, :3]
    assert np.allclose(T_next[3], [0.0, 0.0, 0.0, 1.0]), f"Bad homogeneous row: {T_next[3]}"
    return T_next


def _generate_seed_poses(
    n_azimuth: int,
    n_polar: int,
    n_rolls: int,
    n_radii: int,
) -> list[np.ndarray]:
    """Camera poses above z=0 on concentric spheres, looking toward (0, 0, 0.5), with optical-axis rolls."""
    assert n_azimuth >= 1, f"n_azimuth must be >= 1, got {n_azimuth}"
    assert n_polar >= 1, f"n_polar must be >= 1, got {n_polar}"
    assert n_rolls >= 1, f"n_rolls must be >= 1, got {n_rolls}"
    assert n_radii >= 1, f"n_radii must be >= 1, got {n_radii}"
    eyes = _seed_eyes_on_sphere(n_azimuth, n_polar, n_radii)
    rolls = np.linspace(0.0, 2.0 * np.pi, n_rolls, endpoint=False)
    poses = [_look_at_with_roll(eye, _SEED_LOOKAT_TARGET, float(roll), _SEED_UP) for eye in eyes for roll in rolls]
    n_expected = len(eyes) * n_rolls
    assert len(poses) == n_expected, f"expected {n_expected} seeds, got {len(poses)}"
    assert all(T[2, 3] > 0.0 for T in poses), "All seed camera poses must have translation z > 0"
    return poses


def _pose_from_xyz_wxyz(xyz_wxyz: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """Build OpenCV T_world_cam from robot-base x y z qw qx qy qz."""
    xyz_wxyz = np.asarray(xyz_wxyz, dtype=np.float64).reshape(-1)
    assert xyz_wxyz.shape == (7,), f"seed pose must be x y z qw qx qy qz, got shape {xyz_wxyz.shape}"
    t = xyz_wxyz[:3]
    q = xyz_wxyz[3:]
    q_norm = float(np.linalg.norm(q))
    assert q_norm > 1e-8, f"seed quaternion is zero: {q}"
    assert abs(q_norm - 1.0) < 1e-3, f"seed quaternion must be unit length, got norm={q_norm} q={q}"
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat2mat(q / q_norm)
    T[:3, 3] = t
    return T


def _T_to_vec(T: np.ndarray) -> np.ndarray:
    """SE(3) 4x4 → 6D [tx, ty, tz, rx, ry, rz] with rotation-vector orientation."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    axis, angle = mat2axangle(T[:3, :3])
    rotvec = np.asarray(axis, dtype=np.float64) * float(angle)
    return np.concatenate([T[:3, 3].astype(np.float64), rotvec], axis=0)


def _vec_to_T(x: np.ndarray) -> np.ndarray:
    """6D [tx, ty, tz, rx, ry, rz] → SE(3) 4x4."""
    x = np.asarray(x, dtype=np.float64).reshape(6)
    rotvec = x[3:]
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        R = np.eye(3, dtype=np.float64)
    else:
        R = axangle2mat(rotvec / angle, angle)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = x[:3]
    return T


def _T_to_sapien_pose(T_world_cam: np.ndarray) -> sapien.Pose:
    """RealSense-optical T_world_cam → Sapien / ManiSkill camera pose."""
    t, R_ms = camera_extrinsic_to_maniskill_pose(T_world_cam)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_ms
    mat[:3, 3] = t
    return sapien.Pose(mat[:3, 3], mat2quat(mat[:3, :3]))


def _urdf_for_sapien(urdf_path: pathlib.Path) -> pathlib.Path:
    """Sapien cannot resolve `file://` mesh hrefs that xacrodoc emits. Strip the prefix."""
    text = urdf_path.read_text()
    rewritten = text.replace('filename="file://', 'filename="')
    if rewritten == text:
        return urdf_path
    out = urdf_path.with_name(urdf_path.stem + "__sapien.urdf")
    out.write_text(rewritten)
    return out


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE(3) `T` to `(N, 3)` points. Empty clouds stay empty."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    assert pts.ndim == 2 and pts.shape[1] == 3, f"pts must be (N, 3), got {pts.shape}"
    if pts.shape[0] == 0:
        return pts
    return (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def _world_points_to_camera(T_world_cam: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """Map robot-base points into the OpenCV camera frame: `(p - t) @ R`."""
    assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
    assert pts_world.ndim == 2 and pts_world.shape[1] == 3, f"pts_world must be (N, 3), got {pts_world.shape}"
    assert pts_world.shape[0] >= 1, "pts_world must contain at least 1 point"
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return ((pts_world.astype(np.float64) - t) @ R).astype(np.float32)


def _remove_radius_outliers(points: np.ndarray, nb_points: int, radius_m: float) -> np.ndarray:
    """Drop points with fewer than `nb_points` neighbors inside `radius_m` (Open3D radius outlier)."""
    import open3d as o3d

    assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    assert points.shape[0] >= 1, "points must contain at least 1 point"
    assert nb_points >= 1, f"nb_points must be >= 1, got {nb_points}"
    assert radius_m > 0.0, f"radius_m must be > 0, got {radius_m}"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    filtered, _ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius_m)
    out = np.asarray(filtered.points)
    assert out.ndim == 2 and out.shape[1] == 3, f"filtered points must be (N, 3), got {out.shape}"
    assert out.shape[0] >= 1, (
        f"Radius outlier filter removed all points (n_in={points.shape[0]}, nb_points={nb_points}, "
        f"radius_m={radius_m})"
    )
    return out.astype(np.float64, copy=False)


def _subsample_pcd(
    points: np.ndarray,
    n_samples: int,
    n_random_downsample_initial: int,
    device: str,
    rng: np.random.Generator,
    profile: dict[str, float] | None = None,
) -> np.ndarray:
    """Random subsample to `n_random_downsample_initial`, then FPS to `n_samples`."""
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    assert n_samples >= 1, f"n_samples must be >= 1, got {n_samples}"
    assert (
        n_random_downsample_initial >= n_samples
    ), f"n_random_downsample_initial ({n_random_downsample_initial}) must be >= n_samples ({n_samples})"
    n = int(points.shape[0])
    if n <= n_samples:
        return points.astype(np.float32, copy=False)
    if n > n_random_downsample_initial:
        idx_rand = rng.choice(n, size=n_random_downsample_initial, replace=False)
        points = points[idx_rand]
        n = int(points.shape[0])
    if profile is not None:
        profile["n_calls"] = profile.get("n_calls", 0.0) + 1.0
        profile["n_points_sum"] = profile.get("n_points_sum", 0.0) + float(n)
        profile["n_samples"] = float(n_samples)
    idx = farthest_point_sample_pyg_lib(points, n_samples, device=device)
    return points[idx].astype(np.float32, copy=False)


class SimRobotRenderer:
    """Sapien scene with the Jrl2 URDF. 6-view depth renders are fused in the robot-base frame."""

    def __init__(self, urdf_path: pathlib.Path) -> None:
        sapien_urdf = _urdf_for_sapien(urdf_path)
        assert sapien_urdf.is_file(), f"Sapien URDF not found: {sapien_urdf}"

        self._scene = sapien.Scene()
        self._scene.set_ambient_light([0.45, 0.45, 0.45])
        self._scene.add_directional_light([0.0, 1.0, -1.0], [1.0, 1.0, 1.0])
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(sapien_urdf))
        assert robot is not None, f"Failed to load URDF {sapien_urdf}"
        self._robot = robot
        self._robot.set_root_pose(sapien.Pose([0.0, 0.0, 0.0]))

        self._joint_names = [j.name for j in self._robot.get_active_joints()]
        self._dof = int(self._robot.dof)
        assert len(self._joint_names) >= 1, "Sapien robot has no active joints"
        assert self._dof == len(self._joint_names), f"dof {self._dof} != n joints {len(self._joint_names)}"
        print(f"[info] Sapien active joints ({self._dof}): {self._joint_names}")

        fovy = float(np.deg2rad(_CAPTURE_FOV_DEG))
        assert fovy > 0.0, f"capture fovy must be > 0, got {fovy}"
        self._cam = self._scene.add_camera(
            _CAPTURE_CAMERA_NAME,
            _CAPTURE_WIDTH,
            _CAPTURE_HEIGHT,
            fovy=fovy,
            near=MIN_DEPTH_M,
            far=_CAPTURE_FAR_M,
        )
        self._capture_Ts = _axis_aligned_capture_poses(_SEED_LOOKAT_TARGET, _CAPTURE_DISTANCE_M)
        print(
            f"[info] Multiview capture: 6 axis cameras at {_CAPTURE_DISTANCE_M:g} m, "
            f"{_CAPTURE_WIDTH}x{_CAPTURE_HEIGHT} fov={_CAPTURE_FOV_DEG:g} deg far={_CAPTURE_FAR_M:g} m"
        )

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def set_qpos(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        assert qpos.shape == (self._dof,), f"qpos must be ({self._dof},), got {qpos.shape}"
        self._robot.set_qpos(qpos)

    def get_links(self) -> list:
        links = list(self._robot.get_links())
        assert len(links) >= 1, "Sapien robot has no links"
        return links

    def _render_camera_frame(self, T_world_cam: np.ndarray) -> np.ndarray:
        """Robot-only cloud in the OpenCV camera frame of `T_world_cam`, `(M, 3)` float32."""
        self._cam.local_pose = _T_to_sapien_pose(T_world_cam)
        self._scene.update_render()
        self._cam.take_picture()
        position = self._cam.get_picture("Position")
        segmentation = self._cam.get_picture("Segmentation")
        assert position.ndim == 3 and position.shape[2] >= 3, f"Bad Position texture: {position.shape}"
        xyz_gl = position[:, :, :3].astype(np.float64)
        if segmentation.ndim == 3:
            seg = segmentation[:, :, 0]
        else:
            seg = segmentation
        robot_px = seg > 0
        xyz_cv = np.stack([xyz_gl[:, :, 0], -xyz_gl[:, :, 1], -xyz_gl[:, :, 2]], axis=-1)
        z = xyz_cv[:, :, 2]
        valid = robot_px & np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= _CAPTURE_FAR_M)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float32)
        return xyz_cv[valid].astype(np.float32)

    def render_multiview_world(self, qpos: np.ndarray, n_samples: int, device: str) -> np.ndarray:
        """Fuse 6 axis-aligned renders in the robot-base frame and FPS to `n_samples`."""
        assert n_samples >= 1, f"n_samples must be >= 1, got {n_samples}"
        assert isinstance(device, str) and len(device) > 0, f"device must be a non-empty str, got {device!r}"
        self.set_qpos(qpos)
        parts: list[np.ndarray] = []
        n_per_view: list[int] = []
        for T in self._capture_Ts:
            pts_cam = self._render_camera_frame(T)
            n_per_view.append(int(pts_cam.shape[0]))
            if pts_cam.shape[0] == 0:
                continue
            parts.append(_transform_points(T, pts_cam))
        assert len(parts) >= 1, f"All 6 capture views were empty, counts={n_per_view}"
        p_world = np.concatenate(parts, axis=0)
        assert p_world.ndim == 2 and p_world.shape[1] == 3 and p_world.shape[0] >= 1
        n_fused = int(p_world.shape[0])
        if n_fused > n_samples:
            idx = farthest_point_sample_pyg_lib(p_world, n_samples, device=device)
            p_world = p_world[idx]
        print(f"[info]   views[+x,-x,+y,-y,+z,-z]={n_per_view} fused={n_fused} fps={p_world.shape[0]}")
        return p_world.astype(np.float32, copy=False)


@dataclasses.dataclass
class _ManualSeedSelection:
    T_world_cam: np.ndarray
    translation_step_handle: Any
    rotation_step_handle: Any
    move_buttons: list[Any]
    command_handles: list[Any]
    selected_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    cost: float | None = None
    pcd_sims: list[np.ndarray] | None = None
    best_cost: float = np.inf
    t0: float = dataclasses.field(default_factory=time.perf_counter)


class ExtrinsicsVisualizer:
    """Viser overlay in the robot-base frame: clouds, camera frustum, CMA-ES cost plot."""

    def __init__(
        self,
        pcd_reals: list[np.ndarray],
        K: np.ndarray,
        height: int,
        width: int,
        rgb_frames: np.ndarray,
        cma_maxiter: int,
        fps_idx: np.ndarray,
        renderer: SimRobotRenderer,
        joint_angles: np.ndarray,
        pcd_sim_worlds: list[np.ndarray],
        chamfer_device: str,
    ) -> None:
        import viser
        from viser import uplot

        assert len(pcd_reals) >= 1
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0
        assert rgb_frames.ndim == 4 and rgb_frames.shape[0] == len(
            pcd_reals
        ), f"rgb_frames must be ({len(pcd_reals)}, H, W, 3), got {rgb_frames.shape}"
        assert cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {cma_maxiter}"
        fps_idx = np.asarray(fps_idx, dtype=np.int64).reshape(-1)
        assert fps_idx.shape == (len(pcd_reals),), f"fps_idx {fps_idx.shape} != n clouds {len(pcd_reals)}"
        fy = float(K[1, 1])
        assert fy > 0, f"K[1,1] (fy) must be > 0, got {fy}"

        self._pcd_reals = pcd_reals
        self._rgb_frames = rgb_frames
        self._fps_idx = fps_idx
        self._renderer = renderer
        self._joint_angles = np.asarray(joint_angles, dtype=np.float64)
        self._pcd_sim_worlds = pcd_sim_worlds
        self._chamfer_device = chamfer_device
        assert len(pcd_sim_worlds) == len(
            pcd_reals
        ), f"sim worlds {len(pcd_sim_worlds)} != real clouds {len(pcd_reals)}"
        for i, p_world in enumerate(pcd_sim_worlds):
            assert (
                p_world.ndim == 2 and p_world.shape[1] == 3 and p_world.shape[0] >= 1
            ), f"pcd_sim_worlds[{i}] must be (N, 3) with N>=1, got {p_world.shape}"
        assert self._joint_angles.ndim == 2, f"joint_angles must be (T, dof), got {self._joint_angles.shape}"
        assert self._joint_angles.shape == (
            len(pcd_reals),
            renderer.dof,
        ), f"joint_angles {self._joint_angles.shape} != ({len(pcd_reals)}, {renderer.dof})"
        self._T = _look_at_with_roll(
            np.array([0.80, 0.00, 0.50], dtype=np.float64),
            _SEED_LOOKAT_TARGET,
            np.pi,
            _SEED_UP,
        )
        self._maxiter = cma_maxiter
        self._xs = np.arange(1, cma_maxiter + 1, dtype=np.float64)
        self._ys_best = np.full(cma_maxiter, np.nan, dtype=np.float64)
        self._ys_pop = np.full(cma_maxiter, np.nan, dtype=np.float64)
        self._Ts: list[np.ndarray | None] = [None] * cma_maxiter
        self._pcd_sims_hist: list[list[np.ndarray] | None] = [None] * cma_maxiter
        self._pop_Ts_hist: list[list[np.ndarray] | None] = [None] * cma_maxiter
        self._best_costs: list[float | None] = [None] * cma_maxiter
        self._n_recorded = 0
        self._live_pcd_sims: list[np.ndarray] | None = None
        self._live_pop_Ts: list[np.ndarray] | None = None
        self._pop_axes: list = []
        self._manual_seed: _ManualSeedSelection | None = None
        self._manual_seed_status: Any | None = None
        self._timestep = 0
        self._iteration = 0
        self._T_init = self._T.copy()

        self._server = viser.ViserServer()
        self._server.scene.world_axes.visible = True
        self._server.scene.world_axes.scale = 0.15
        MeshUtils.add_xy_grid(self._server)
        self._server.initial_camera.position = (1.2, 1.2, 0.8)
        self._server.initial_camera.look_at = (0.0, 0.0, 0.3)
        self._server.initial_camera.up = (0.0, 0.0, 1.0)

        self._real_handle = self._server.scene.add_point_cloud(
            "/pcd_real",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=(220, 60, 60),
            point_size=0.003,
            point_shape="circle",
        )
        self._sim_handle = self._server.scene.add_point_cloud(
            "/pcd_sim",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=(40, 200, 120),
            point_size=0.003,
            point_shape="circle",
        )
        wxyz0, pos0 = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum = self._server.scene.add_camera_frustum(
            "/camera",
            fov=float(2.0 * np.arctan(height / (2.0 * fy))),
            aspect=width / height,
            scale=0.04,
            line_width=1.5,
            color=(40, 120, 255),
            image=rgb_frames[0],
            format="jpeg",
            wxyz=wxyz0,
            position=pos0,
        )
        self._cam_axes = self._server.scene.add_frame(
            "/camera_axes",
            axes_length=_BEST_AXES_LENGTH,
            axes_radius=_BEST_AXES_RADIUS,
            origin_radius=_BEST_ORIGIN_RADIUS,
            wxyz=wxyz0,
            position=pos0,
        )
        self._load_robot_meshes()
        self._cost_md = self._server.gui.add_markdown("best cost: —")
        self._pose_md = self._server.gui.add_markdown("camera pose: —")
        timestep_slider = self._server.gui.add_slider(
            "timestep (sampled)", min=0, max=max(len(pcd_reals) - 1, 0), step=1, initial_value=0
        )

        @timestep_slider.on_update
        def _on_timestep(_) -> None:
            self._timestep = int(timestep_slider.value)
            self._apply()

        # Viser sets step to min(step, max-min). max==min makes step 0 and the client sends NaN.
        self._iter_slider = self._server.gui.add_slider(
            "iteration", min=0, max=cma_maxiter, step=1, initial_value=0, disabled=True
        )

        @self._iter_slider.on_update
        def _on_iteration(_) -> None:
            value = self._iter_slider.value
            assert value == value, "iteration slider value is NaN"
            value = int(value)
            cap = int(self._n_recorded)
            assert 0 <= value <= self._maxiter, f"iteration {value} not in [0, {self._maxiter}]"
            if value > cap:
                value = cap
                self._iter_slider.value = cap
            self._iteration = value
            self._apply()

        empty = np.array([], dtype=np.float64)
        self._seed_t: list[float] = []
        self._seed_best: list[float] = []
        self._seed_plot = self._server.gui.add_uplot(
            data=(empty, empty),
            series=(
                uplot.Series(show=False),
                uplot.Series(label="best found", stroke="#e67e22", width=2),
            ),
            title="Seed search: best vs time",
            scales={"x": {"time": False}},
            axes=(
                uplot.Axis(label="time (s)"),
                uplot.Axis(label="cost"),
            ),
            aspect=1.8,
            height=220,
        )
        self._cost_plot = self._server.gui.add_uplot(
            data=(empty, empty, empty),
            series=(
                uplot.Series(show=False),
                uplot.Series(label="best found", stroke="#e67e22", width=2),
                uplot.Series(label="pop min", stroke="#3498db", width=1),
            ),
            title="CMA-ES cost vs iteration",
            scales={"x": {"time": False}},
            axes=(
                uplot.Axis(label="iteration"),
                uplot.Axis(label="cost"),
            ),
            aspect=1.8,
            height=220,
        )
        self._apply()
        print(f"[info] Viser extrinsics view at http://{self._server.get_host()}:{self._server.get_port()}")

    def select_seed_from_gui(
        self,
        translation_step_m: float,
        rotation_step_deg: float,
    ) -> tuple[np.ndarray, float, list[np.ndarray]]:
        """Block until the user adjusts and selects a CMA-ES seed in the Viser GUI."""
        assert self._manual_seed is None, "Manual seed selection has already been configured"
        assert translation_step_m > 0.0, f"translation_step_m must be > 0, got {translation_step_m}"
        assert rotation_step_deg > 0.0, f"rotation_step_deg must be > 0, got {rotation_step_deg}"
        with self._server.gui.add_folder("Manual seed selection", expand_by_default=True):
            self._manual_seed_status = self._server.gui.add_markdown("Initializing seed evaluation...")
            translation_step_handle = self._server.gui.add_number(
                "translation step (m)", translation_step_m, min=1.0e-4, step=0.005
            )
            rotation_step_handle = self._server.gui.add_number(
                "rotation step (deg)", rotation_step_deg, min=0.1, step=1.0
            )
            self._server.gui.add_markdown(
                "**Click the 3D view, then use keyboard hotkeys (world / robot-base).**  \n"
                "**Translation:** R/F +X/-X, T/G +Y/-Y, Y/H +Z/-Z  \n"
                "**Rotation:** U/J +roll/-roll, I/K +pitch/-pitch, O/L +yaw/-yaw  \n"
                "**Enter:** select seed and start CMA-ES"
            )
            state = _ManualSeedSelection(
                T_world_cam=self._T_init.copy(),
                translation_step_handle=translation_step_handle,
                rotation_step_handle=rotation_step_handle,
                move_buttons=[],
                command_handles=[],
            )
            self._manual_seed = state
            button_specs = (
                ("R", "R — +X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                ("F", "F — -X", [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                ("T", "T — +Y", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
                ("G", "G — -Y", [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]),
                ("Y", "Y — +Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                ("H", "H — -Z", [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]),
                ("U", "U — +roll (about X)", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                ("J", "J — -roll (about X)", [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                ("I", "I — +pitch (about Y)", [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                ("K", "K — -pitch (about Y)", [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
                ("O", "O — +yaw (about Z)", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                ("L", "L — -yaw (about Z)", [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            )
            for hotkey, label, translation_direction, rotation_direction in button_specs:
                translation = np.asarray(translation_direction, dtype=np.float64)
                rotation = np.asarray(rotation_direction, dtype=np.float64)

                def _on_move(_, translation=translation, rotation=rotation) -> None:
                    self._move_manual_seed(translation, rotation)

                button = self._server.gui.add_button(label)
                button.on_click(_on_move)
                state.move_buttons.append(button)
                command = self._server.gui.add_command(label, hotkey=hotkey)
                command.on_trigger(_on_move)
                state.command_handles.append(command)
            select_button = self._server.gui.add_button("Select seed and start CMA-ES", color="green")
            select_button.on_click(lambda _: self._confirm_manual_seed())
            state.move_buttons.append(select_button)
            select_command = self._server.gui.add_command("Select seed and start CMA-ES", hotkey="enter")
            select_command.on_trigger(lambda _: self._confirm_manual_seed())
            state.command_handles.append(select_command)

        with state.lock:
            self._evaluate_manual_seed()
        print("[info] Waiting for manual seed selection in Viser (click the 3D view, then RF/TG/YH/UJ/IK/OL/Enter)...")
        state.selected_event.wait()
        assert state.cost is not None and state.pcd_sims is not None, "Selected seed has not been evaluated"
        return state.T_world_cam.copy(), float(state.cost), [pcd.copy() for pcd in state.pcd_sims]

    def _move_manual_seed(self, translation_direction: np.ndarray, rotation_direction: np.ndarray) -> None:
        state = self._manual_seed
        assert state is not None, "Manual seed selection is not configured"
        with state.lock:
            if state.selected_event.is_set():
                return
            translation_step_m = float(state.translation_step_handle.value)
            rotation_step_rad = float(np.deg2rad(state.rotation_step_handle.value))
            assert translation_step_m > 0.0, f"translation step must be > 0, got {translation_step_m}"
            assert rotation_step_rad > 0.0, f"rotation step must be > 0, got {rotation_step_rad}"
            state.T_world_cam = _step_pose_world(
                state.T_world_cam,
                translation_direction * translation_step_m,
                rotation_direction * rotation_step_rad,
            )
            self._evaluate_manual_seed()

    def _evaluate_manual_seed(self) -> None:
        state = self._manual_seed
        assert state is not None, "Manual seed selection is not configured"
        assert self._manual_seed_status is not None
        self._manual_seed_status.content = "**Evaluating pose...**"
        cost, pcd_sims, _, _ = _pose_cost(
            state.T_world_cam,
            self._pcd_reals,
            self._pcd_sim_worlds,
            self._chamfer_device,
        )
        state.cost = float(cost)
        state.pcd_sims = pcd_sims
        state.best_cost = min(state.best_cost, state.cost)
        assert np.isfinite(state.best_cost), f"best_cost must be finite, got {state.best_cost}"
        self.record_seed_best(time.perf_counter() - state.t0, state.best_cost)
        self.set_best(pcd_sims, state.cost, state.T_world_cam)
        self._manual_seed_status.content = (
            f"**Current cost:** `{state.cost:.6f}`  \n"
            f"**Best cost:** `{state.best_cost:.6f}`  \n"
            f"{_pose_translation_rotation_text(state.T_world_cam)}"
        )

    def _confirm_manual_seed(self) -> None:
        state = self._manual_seed
        assert state is not None, "Manual seed selection is not configured"
        with state.lock:
            assert state.cost is not None and state.pcd_sims is not None, "Current seed has not been evaluated"
            for handle in state.move_buttons:
                handle.disabled = True
            for handle in state.command_handles:
                handle.remove()
            state.command_handles.clear()
            state.translation_step_handle.disabled = True
            state.rotation_step_handle.disabled = True
            assert self._manual_seed_status is not None
            self._manual_seed_status.content = f"**Selected seed cost:** `{state.cost:.6f}`"
            state.selected_event.set()

    def set_best(self, pcd_sims: list[np.ndarray], cost: float, T_world_cam: np.ndarray) -> None:
        """Live-update the scene when a new global best is found, if viewing the latest iteration."""
        assert len(pcd_sims) == len(
            self._pcd_reals
        ), f"sim clouds {len(pcd_sims)} != real clouds {len(self._pcd_reals)}"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        if self._n_recorded > 0 and self._iteration != self._n_recorded:
            return
        self._live_pcd_sims = [p.copy() for p in pcd_sims]
        self._apply(T=T_world_cam.copy(), pcd_sims=self._live_pcd_sims, cost=float(cost))

    def set_population(self, pop_Ts: list[np.ndarray]) -> None:
        """Live-update population axes if viewing the latest iteration."""
        assert len(pop_Ts) >= 1, "population must contain at least one pose"
        for i, T_pop in enumerate(pop_Ts):
            assert T_pop.shape == (4, 4), f"pop_Ts[{i}] must be 4x4, got {T_pop.shape}"
        if self._n_recorded > 0 and self._iteration != self._n_recorded:
            return
        self._live_pop_Ts = [T.copy() for T in pop_Ts]
        self._set_pop_axes(self._live_pop_Ts, hide_extra=True)

    def record_seed_best(self, elapsed_s: float, best_cost: float) -> None:
        """Append one seed-search sample: wall time (s) vs best chamfer so far."""
        assert elapsed_s >= 0.0, f"elapsed_s must be >= 0, got {elapsed_s}"
        assert np.isfinite(best_cost), f"best_cost must be finite, got {best_cost}"
        self._seed_t.append(float(elapsed_s))
        self._seed_best.append(float(best_cost))
        t = np.asarray(self._seed_t, dtype=np.float64)
        y = np.asarray(self._seed_best, dtype=np.float64)
        self._seed_plot.data = (t, y)

    def record_generation(
        self,
        generation: int,
        best_cost: float,
        pop_min: float,
        T_world_cam: np.ndarray,
        pcd_sims: list[np.ndarray],
        pop_Ts: list[np.ndarray],
    ) -> None:
        assert 1 <= generation <= self._maxiter, f"generation {generation} not in [1, {self._maxiter}]"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        assert len(pcd_sims) == len(
            self._pcd_reals
        ), f"sim clouds {len(pcd_sims)} != real clouds {len(self._pcd_reals)}"
        assert len(pop_Ts) >= 1, "population must contain at least one pose"
        for i, T_pop in enumerate(pop_Ts):
            assert T_pop.shape == (4, 4), f"pop_Ts[{i}] must be 4x4, got {T_pop.shape}"
        i = generation - 1
        self._ys_best[i] = float(best_cost)
        self._ys_pop[i] = float(pop_min)
        self._Ts[i] = T_world_cam.copy()
        self._pcd_sims_hist[i] = [p.copy() for p in pcd_sims]
        self._pop_Ts_hist[i] = [T.copy() for T in pop_Ts]
        self._best_costs[i] = float(best_cost)
        self._cost_plot.data = (self._xs[:generation], self._ys_best[:generation], self._ys_pop[:generation])
        follow = self._n_recorded == 0 or self._iteration == self._n_recorded
        self._n_recorded = generation
        self._iter_slider.disabled = False
        self._iter_slider.max = generation
        if follow:
            self._iteration = generation
            self._iter_slider.value = generation
            self._apply()

    def _snapshot_index(self) -> int | None:
        if self._n_recorded == 0 or self._iteration == 0:
            return None
        return min(int(self._iteration), self._n_recorded) - 1

    def _apply(
        self,
        T: np.ndarray | None = None,
        pcd_sims: list[np.ndarray] | None = None,
        cost: float | None = None,
    ) -> None:
        t = self._timestep
        snap_i = self._snapshot_index()
        pop_Ts: list[np.ndarray] | None = None
        if T is None and self._iteration == 0:
            T = self._T_init
            pcd_sims = None
            cost = None
            pop_Ts = []
        elif T is None and snap_i is not None:
            T = self._Ts[snap_i]
            pcd_sims = self._pcd_sims_hist[snap_i]
            cost = self._best_costs[snap_i]
            pop_Ts = self._pop_Ts_hist[snap_i]
            assert T is not None, f"Missing T snapshot at iteration {snap_i + 1}"
            assert pcd_sims is not None, f"Missing sim-cloud snapshot at iteration {snap_i + 1}"
            assert cost is not None, f"Missing cost snapshot at iteration {snap_i + 1}"
            assert pop_Ts is not None, f"Missing population snapshot at iteration {snap_i + 1}"
        elif T is None and pcd_sims is None:
            pcd_sims = self._live_pcd_sims
            pop_Ts = self._live_pop_Ts
        if T is not None:
            self._T = T
        demo_frame = int(self._fps_idx[t])
        if cost is not None:
            self._cost_md.content = f"**iter {self._iteration} best cost:** `{cost:.6f}`"
        elif self._iteration == 0:
            self._cost_md.content = "**iter 0 (init)**"
        self._pose_md.content = (
            f"**sampled t={t}  demo frame={demo_frame}**\n\n{_pose_translation_rotation_text(self._T)}"
        )
        self._real_handle.points = np.ascontiguousarray(_transform_points(self._T, self._pcd_reals[t]))
        if pcd_sims is None or pcd_sims[t].shape[0] == 0:
            sim_world = np.zeros((1, 3), dtype=np.float32)
        else:
            sim_world = _transform_points(self._T, pcd_sims[t])
        self._sim_handle.points = np.ascontiguousarray(sim_world)
        wxyz, pos = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum.wxyz = wxyz
        self._frustum.position = pos
        self._frustum.image = self._rgb_frames[t]
        self._cam_axes.wxyz = wxyz
        self._cam_axes.position = pos
        if pop_Ts is not None:
            self._set_pop_axes(pop_Ts, hide_extra=True)
        self._sync_robot()

    def _load_robot_meshes(self) -> None:
        """Parent each Sapien link's visual meshes under a Viser frame, matching ManiSkill's ViserVisualizer."""
        from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_meshes

        self._robot_links = self._renderer.get_links()
        self._robot_link_frames: list = []
        n_meshes = 0
        for link in self._robot_links:
            pose = link.pose
            position = tuple(np.asarray(pose.p, dtype=np.float64).tolist())
            wxyz = tuple(np.asarray(pose.q, dtype=np.float64).tolist())
            frame = self._server.scene.add_frame(
                f"/robot/{link.name}",
                show_axes=False,
                position=position,
                wxyz=wxyz,
            )
            self._robot_link_frames.append(frame)
            meshes = get_actor_visual_meshes(link.entity)
            for i, mesh in enumerate(meshes):
                self._server.scene.add_mesh_trimesh(f"/robot/{link.name}/visual/{i}", mesh)
                n_meshes += 1
        assert n_meshes >= 1, "Robot URDF produced no visual meshes for Viser"
        print(f"[info] Viser robot model: {len(self._robot_links)} links, {n_meshes} visual meshes")

    def _sync_robot(self) -> None:
        self._renderer.set_qpos(self._joint_angles[self._timestep])
        for link, frame in zip(self._robot_links, self._robot_link_frames, strict=True):
            pose = link.pose
            frame.position = tuple(np.asarray(pose.p, dtype=np.float64).tolist())
            frame.wxyz = tuple(np.asarray(pose.q, dtype=np.float64).tolist())

    def _set_pop_axes(self, pop_Ts: list[np.ndarray], *, hide_extra: bool) -> None:
        while len(self._pop_axes) < len(pop_Ts):
            i = len(self._pop_axes)
            self._pop_axes.append(
                self._server.scene.add_frame(
                    f"/pop_axes/{i}",
                    axes_length=_POP_AXES_LENGTH,
                    axes_radius=_POP_AXES_RADIUS,
                    origin_radius=_POP_ORIGIN_RADIUS,
                )
            )
        for i, handle in enumerate(self._pop_axes):
            if i < len(pop_Ts):
                wxyz, pos = MeshUtils._pose_mat_to_wxyz_position(pop_Ts[i])
                handle.wxyz = wxyz
                handle.position = pos
                handle.visible = True
            elif hide_extra:
                handle.visible = False

    def wait(self) -> None:
        self._server.sleep_forever()

    def close(self) -> None:
        self._server.stop()


def _pose_cost(
    T_world_cam: np.ndarray,
    pcd_reals: list[np.ndarray],
    pcd_sim_worlds: list[np.ndarray],
    chamfer_device: str,
) -> tuple[float, list[np.ndarray], float, float]:
    assert len(pcd_reals) == len(pcd_sim_worlds), f"real clouds {len(pcd_reals)} != sim worlds {len(pcd_sim_worlds)}"
    total = 0.0
    pcd_sims: list[np.ndarray] = []
    t_transform = 0.0
    t_chamfer = 0.0
    for pcd_real, p_world in zip(pcd_reals, pcd_sim_worlds, strict=True):
        t0 = time.perf_counter()
        pcd_sim = _world_points_to_camera(T_world_cam, p_world)
        t_transform += time.perf_counter() - t0
        pcd_sims.append(pcd_sim)
        t0 = time.perf_counter()
        total += float(one_sided_squared_nn_distance(pcd_real, pcd_sim, device=chamfer_device))
        t_chamfer += time.perf_counter() - t0
    return total, pcd_sims, t_transform, t_chamfer


def _select_seed_pose(
    seed_Ts: list[np.ndarray],
    pcd_reals: list[np.ndarray],
    pcd_sim_worlds: list[np.ndarray],
    chamfer_device: str,
    vis: ExtrinsicsVisualizer | None,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Evaluate look-at seed poses and return the lowest-chamfer (T, cost, pcd_sims)."""
    assert len(seed_Ts) >= 1, "seed_Ts must be non-empty"
    best_cost = np.inf
    best_T: np.ndarray | None = None
    best_pcd_sims: list[np.ndarray] | None = None
    xyz_Ts: list[np.ndarray] = []
    t0_seed = time.perf_counter()
    pbar = tqdm(seed_Ts, desc="seed poses")
    for T in pbar:
        cost, pcd_sims, _, _ = _pose_cost(T, pcd_reals, pcd_sim_worlds, chamfer_device)
        xyz = T[:3, 3]
        new_xyz = len(xyz_Ts) == 0 or not np.allclose(xyz_Ts[-1][:3, 3], xyz)
        if new_xyz:
            xyz_Ts.append(T.copy())
            if vis is not None:
                vis.set_population(xyz_Ts)
        if cost < best_cost:
            best_cost = cost
            best_T = T.copy()
            best_pcd_sims = pcd_sims
            if vis is not None:
                vis.set_best(pcd_sims, best_cost, best_T)
        if vis is not None and np.isfinite(best_cost):
            vis.record_seed_best(time.perf_counter() - t0_seed, best_cost)
        pbar.set_postfix(best=f"{best_cost:.6f}")
    assert best_T is not None and best_pcd_sims is not None, "Seed search produced no pose"
    return best_T, best_cost, best_pcd_sims


def _optimize_extrinsics(
    pcd_reals: list[np.ndarray],
    pcd_sim_worlds: list[np.ndarray],
    chamfer_device: str,
    seed_automatically: bool,
    seed_from_gui: bool,
    seed_pose: tuple[float, float, float, float, float, float, float] | None,
    gui_translation_step_m: float,
    gui_rotation_step_deg: float,
    n_seed_azimuth: int,
    n_seed_polar: int,
    n_seed_rolls: int,
    n_seed_radii: int,
    cma_sigma_pos: float,
    cma_sigma_rot: float,
    cma_maxiter: int,
    cma_popsize: int | None,
    vis: ExtrinsicsVisualizer | None,
) -> tuple[np.ndarray, float]:
    """CMA-ES over 6D camera pose. Returns (best 4x4 T_world_cam, cost)."""
    import cma

    assert n_seed_azimuth >= 1, f"n_seed_azimuth must be >= 1, got {n_seed_azimuth}"
    assert n_seed_polar >= 1, f"n_seed_polar must be >= 1, got {n_seed_polar}"
    assert n_seed_rolls >= 1, f"n_seed_rolls must be >= 1, got {n_seed_rolls}"
    assert n_seed_radii >= 1, f"n_seed_radii must be >= 1, got {n_seed_radii}"
    assert cma_sigma_pos > 0, f"cma_sigma_pos must be > 0, got {cma_sigma_pos}"
    assert cma_sigma_rot > 0, f"cma_sigma_rot must be > 0, got {cma_sigma_rot}"
    _validate_seed_mode(seed_automatically, seed_from_gui, seed_pose)
    if seed_automatically:
        seed_Ts = _generate_seed_poses(n_seed_azimuth, n_seed_polar, n_seed_rolls, n_seed_radii)
        radii = np.linspace(_SEED_RADIUS_MIN_M, _SEED_RADIUS_MAX_M, n_seed_radii, dtype=np.float64)
        print(
            f"[info] Seed search: n_azimuth={n_seed_azimuth} n_polar={n_seed_polar} n_rolls={n_seed_rolls} "
            f"n_radii={n_seed_radii} radii_m={radii.tolist()} n_total={len(seed_Ts)} "
            f"lookat={_SEED_LOOKAT_TARGET.tolist()}"
        )
        t0_seed = time.perf_counter()
        best_T, best_cost, best_pcd_sims = _select_seed_pose(
            seed_Ts,
            pcd_reals,
            pcd_sim_worlds,
            chamfer_device,
            vis,
        )
        _log_elapsed(f"Seed search finished, best={best_cost:.6f}", t0_seed)
    elif seed_from_gui:
        assert vis is not None, "GUI seed selection requires an ExtrinsicsVisualizer"
        best_T, best_cost, best_pcd_sims = vis.select_seed_from_gui(
            gui_translation_step_m,
            gui_rotation_step_deg,
        )
        print(f"[info] Manual seed selected: cost={best_cost:.6f} T_world_cam=\n{best_T}")
    else:
        assert seed_pose is not None, "seed_pose is required when not using --seed-automatically or --seed-from-gui"
        best_T = _pose_from_xyz_wxyz(seed_pose)
        t0_seed = time.perf_counter()
        best_cost, best_pcd_sims, _, _ = _pose_cost(best_T, pcd_reals, pcd_sim_worlds, chamfer_device)
        if vis is not None:
            vis.set_best(best_pcd_sims, best_cost, best_T)
            vis.record_seed_best(time.perf_counter() - t0_seed, best_cost)
        q_wxyz = mat2quat(best_T[:3, :3])
        print(
            f"[info] Seed pose xyz={best_T[:3, 3].tolist()} wxyz={q_wxyz.tolist()} "
            f"cost={best_cost:.6f} T_world_cam=\n{best_T}"
        )

    x0 = _T_to_vec(best_T)
    cma_stds = [cma_sigma_pos, cma_sigma_pos, cma_sigma_pos, cma_sigma_rot, cma_sigma_rot, cma_sigma_rot]
    cma_opts: dict = {"maxiter": cma_maxiter, "verbose": -1, "CMA_stds": cma_stds}
    if cma_popsize is not None:
        cma_opts["popsize"] = cma_popsize
    es = cma.CMAEvolutionStrategy(x0, 1.0, cma_opts)
    print(
        f"[info] CMA-ES x0={x0} sigma_pos={cma_sigma_pos} sigma_rot={cma_sigma_rot} "
        f"popsize={es.popsize} maxiter={cma_maxiter}"
    )

    generation = 0
    t0_opt = time.perf_counter()
    while not es.stop():
        generation += 1
        t0_gen = time.perf_counter()
        t0 = time.perf_counter()
        xs = es.ask()
        t_cma = time.perf_counter() - t0
        costs = []
        pop_Ts: list[np.ndarray] = []
        t_transform = 0.0
        t_chamfer = 0.0
        t_vis = 0.0
        pbar = tqdm(xs, desc=f"cma gen {generation}", leave=False)
        for x in pbar:
            T = _vec_to_T(x)
            cost, pcd_sims, dt_transform, dt_chamfer = _pose_cost(T, pcd_reals, pcd_sim_worlds, chamfer_device)
            t_transform += dt_transform
            t_chamfer += dt_chamfer
            costs.append(cost)
            pop_Ts.append(T.copy())
            if vis is not None:
                t0 = time.perf_counter()
                vis.set_population(pop_Ts)
                t_vis += time.perf_counter() - t0
            if cost < best_cost:
                best_cost = cost
                best_T = T.copy()
                best_pcd_sims = pcd_sims
                if vis is not None:
                    t0 = time.perf_counter()
                    vis.set_best(pcd_sims, best_cost, best_T)
                    t_vis += time.perf_counter() - t0
            pbar.set_postfix(best=f"{best_cost:.6f}")
        t0 = time.perf_counter()
        es.tell(xs, costs)
        t_cma += time.perf_counter() - t0
        pop_min = float(min(costs))
        assert best_T is not None and best_pcd_sims is not None, "CMA-ES produced no pose"
        if vis is not None:
            t0 = time.perf_counter()
            vis.record_generation(generation, best_cost, pop_min, best_T, best_pcd_sims, pop_Ts)
            t_vis += time.perf_counter() - t0
        dt_gen = time.perf_counter() - t0_gen
        t_other = dt_gen - (t_transform + t_chamfer + t_vis + t_cma)
        print(
            f"[info] gen {generation} ({dt_gen:.1f}s): best={best_cost:.6f}  gen_min={pop_min:.6f}  "
            f"gen_mean={float(np.mean(costs)):.6f}"
        )
        print(f"[info]   time_transform={t_transform:.1f}s")
        print(f"[info]   time_chamfer={t_chamfer:.1f}s")
        print(f"[info]   time_vis={t_vis:.1f}s")
        print(f"[info]   time_cma={t_cma:.1f}s")
        print(f"[info]   time_other={t_other:.1f}s")

    assert best_T is not None, "CMA-ES produced no pose"
    _log_elapsed(f"CMA-ES finished ({generation} gens)", t0_opt)
    return best_T, best_cost


def main(args: Args) -> None:
    import torch

    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert (
        args.robot_id.lower() in NAME_TO_ROBOT
    ), f"Unknown robot_id {args.robot_id!r}. Available: {sorted(NAME_TO_ROBOT)}"
    assert args.n_timesteps >= 1, f"n_timesteps must be >= 1, got {args.n_timesteps}"
    assert args.n_pcd_samples_real >= 1, f"n_pcd_samples_real must be >= 1, got {args.n_pcd_samples_real}"
    assert args.n_pcd_samples_sim >= 1, f"n_pcd_samples_sim must be >= 1, got {args.n_pcd_samples_sim}"
    assert args.n_random_downsample_initial >= args.n_pcd_samples_real, (
        f"n_random_downsample_initial ({args.n_random_downsample_initial}) must be >= "
        f"n_pcd_samples_real ({args.n_pcd_samples_real})"
    )
    assert args.cma_sigma_pos > 0, f"cma_sigma_pos must be > 0, got {args.cma_sigma_pos}"
    assert args.cma_sigma_rot > 0, f"cma_sigma_rot must be > 0, got {args.cma_sigma_rot}"
    assert args.cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {args.cma_maxiter}"
    assert args.cma_popsize is None or args.cma_popsize >= 2, f"cma_popsize must be >= 2, got {args.cma_popsize}"
    _validate_seed_mode(args.seed_automatically, args.seed_from_gui, args.seed_pose)
    assert args.gui_translation_step_m > 0.0, f"gui_translation_step_m must be > 0, got {args.gui_translation_step_m}"
    assert args.gui_rotation_step_deg > 0.0, f"gui_rotation_step_deg must be > 0, got {args.gui_rotation_step_deg}"
    assert args.n_seed_azimuth >= 1, f"n_seed_azimuth must be >= 1, got {args.n_seed_azimuth}"
    assert args.n_seed_polar >= 1, f"n_seed_polar must be >= 1, got {args.n_seed_polar}"
    assert args.n_seed_rolls >= 1, f"n_seed_rolls must be >= 1, got {args.n_seed_rolls}"
    assert args.n_seed_radii >= 1, f"n_seed_radii must be >= 1, got {args.n_seed_radii}"
    assert len(args.robot_description) > 0, "robot_description must not be empty"
    assert args.sam_kmax >= 1, f"sam_kmax must be >= 1, got {args.sam_kmax}"
    assert (
        0.0 <= args.sam_score_threshold < 1.0
    ), f"sam_score_threshold must be in [0, 1), got {args.sam_score_threshold}"
    assert args.mask_erode_px == 0 or (
        args.mask_erode_px >= 1 and args.mask_erode_px % 2 == 1
    ), f"mask_erode_px must be 0 or a positive odd int, got {args.mask_erode_px}"
    assert (
        args.radius_outlier_nb_points >= 1
    ), f"radius_outlier_nb_points must be >= 1, got {args.radius_outlier_nb_points}"
    assert args.radius_outlier_radius_m > 0.0, f"radius_outlier_radius_m must be > 0, got {args.radius_outlier_radius_m}"
    assert args.depth_intrinsics_source in (
        "rgb",
        "depth",
    ), f"depth_intrinsics_source must be 'rgb' or 'depth', got {args.depth_intrinsics_source!r}"
    chamfer_device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    color_intrinsics = get_color_intrinsics(args.camera_model_id)

    print(f"[info] Loading RGB-D + qpos from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb_all, depth_raw_all, qpos_all = _load_demo(args.h5_path, args.camera)
    _log_elapsed(f"Loaded rgb{rgb_all.shape} depth{depth_raw_all.shape} qpos{qpos_all.shape}", t0)

    n_frames = rgb_all.shape[0]
    assert args.n_timesteps <= n_frames, f"n_timesteps={args.n_timesteps} > n_frames={n_frames}"
    fps_idx = farthest_point_sample_naive(qpos_all.astype(np.float64), args.n_timesteps, distance="circular")
    fps_idx = np.asarray(fps_idx, dtype=np.int64)
    print(f"[info] FPS circular joint samples at RGB-D frames {fps_idx.tolist()}")

    rgb_sel = rgb_all[fps_idx]
    depth_raw_sel = depth_raw_all[fps_idx]
    qpos_sel = qpos_all[fps_idx]
    H, W = int(rgb_sel.shape[1]), int(rgb_sel.shape[2])
    assert (H, W) == (color_intrinsics.height, color_intrinsics.width), (
        f"{args.camera} rgb is {H}x{W} but {args.camera_model_id} color intrinsics are "
        f"{color_intrinsics.height}x{color_intrinsics.width}"
    )
    K = color_intrinsics.intrinsic_matrix.astype(np.float64)
    print(f"[info] Color K ({args.camera_model_id}):\n{K}")
    print(f"[info] Depth channel uses {args.depth_intrinsics_source} intrinsics")

    depth_m_sel = depth_mm_to_meters(depth_raw_sel)
    if args.depth_intrinsics_source == "depth":
        depth_cam_intrinsics = get_depth_intrinsics(args.camera_model_id)
        assert depth_raw_sel.shape[1:] == (depth_cam_intrinsics.height, depth_cam_intrinsics.width), (
            f"{args.camera} depth is {depth_raw_sel.shape[1]}x{depth_raw_sel.shape[2]} but "
            f"{args.camera_model_id} depth intrinsics are "
            f"{depth_cam_intrinsics.height}x{depth_cam_intrinsics.width}"
        )
        print(f"[info] Reprojecting native depth into the color frame ({args.n_timesteps} sampled frames) ...")
        t0 = time.perf_counter()
        R_dc, t_dc = get_depth_to_color_extrinsics(args.camera_model_id)
        depth_m_sel = np.stack(
            [
                reproject_depth_to_color_frame(
                    depth_m_sel[i],
                    depth_cam_intrinsics,
                    color_intrinsics,
                    R_dc,
                    t_dc,
                )
                for i in range(args.n_timesteps)
            ],
            axis=0,
        )
        _log_elapsed("Reprojected depth into the color frame", t0)
    assert depth_m_sel.shape == (
        args.n_timesteps,
        H,
        W,
    ), f"depth_m_sel shape {depth_m_sel.shape} != ({args.n_timesteps}, {H}, {W})"

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pcd_reals: list[np.ndarray] = []
    demo_frames: list[np.ndarray] = []
    t0_masks = time.perf_counter()
    masks_all = _compute_propagated_robot_masks(
        rgb_all,
        args.h5_path,
        args.camera,
        args.robot_description,
        args.sam_kmax,
        args.sam_score_threshold,
        args.cache_robot_masks,
        args.visualize_robot_masks,
    )
    assert masks_all.shape == (
        n_frames,
        H,
        W,
    ), f"masks_all {masks_all.shape} != ({n_frames}, {H}, {W})"
    for i, frame_idx in enumerate(fps_idx):
        frame_idx = int(frame_idx)
        mask = masks_all[frame_idx]
        assert mask.any(), f"Empty robot mask at frame {frame_idx} for prompt {args.robot_description!r}"
        n_mask = int(mask.sum())
        mask = _erode_mask(mask, args.mask_erode_px)
        print(
            f"[info] Frame {frame_idx} mask pixels: {n_mask} -> {int(mask.sum())} / {mask.size} "
            f"(erode_px={args.mask_erode_px})"
        )
        demo_frames.append(ImageUtils.demo_overlay(rgb_sel[i], mask))
        pts = masked_depth_to_points(depth_m_sel[i], mask, K)
        n_unproj = int(pts.shape[0])
        pts = _remove_radius_outliers(pts, args.radius_outlier_nb_points, args.radius_outlier_radius_m)
        print(
            f"[info] pcd_real[{i}] radius-outlier: {n_unproj} -> {pts.shape[0]} "
            f"(nb_points={args.radius_outlier_nb_points}, radius_m={args.radius_outlier_radius_m})"
        )
        pts = _subsample_pcd(pts, args.n_pcd_samples_real, args.n_random_downsample_initial, chamfer_device, rng)
        assert pts.shape[0] >= 1, f"No robot points at frame {frame_idx}"
        pcd_reals.append(pts)
        print(f"[info] pcd_real[{i}] n={pts.shape[0]}")
    assert len(demo_frames) == args.n_timesteps, f"demo_frames {len(demo_frames)} != n_timesteps {args.n_timesteps}"
    demo_sel = np.stack(demo_frames, axis=0)
    _log_elapsed(f"Built robot masks for {n_frames} frames; used {args.n_timesteps} sampled frames", t0_masks)

    print(f"[info] Loading Jrl2 robot '{args.robot_id}' ...")
    t0 = time.perf_counter()
    robot = get_robot_by_name(args.robot_id)
    urdf_path = pathlib.Path(robot._urdf_filepath)
    print(f"[info] URDF: {urdf_path}")
    renderer = SimRobotRenderer(urdf_path)
    qpos_sel = _expand_demo_qpos_to_joint_names(qpos_sel, robot, renderer.joint_names)
    pcd_sim_worlds: list[np.ndarray] = []
    t0_sim = time.perf_counter()
    for i, q in enumerate(qpos_sel):
        p_world = renderer.render_multiview_world(q, args.n_pcd_samples_sim, chamfer_device)
        pcd_sim_worlds.append(p_world)
        print(f"[info] pcd_sim_world[{i}] n={p_world.shape[0]}")
    _log_elapsed(f"Rendered 6-view sim clouds for {len(pcd_sim_worlds)} timesteps", t0_sim)
    _log_elapsed("Loaded robot + Sapien renderer", t0)

    vis = (
        ExtrinsicsVisualizer(
            pcd_reals, K, H, W, demo_sel, args.cma_maxiter, fps_idx, renderer, qpos_sel, pcd_sim_worlds, chamfer_device
        )
        if args.visualize or args.seed_from_gui
        else None
    )
    best_T, best_cost = _optimize_extrinsics(
        pcd_reals,
        pcd_sim_worlds,
        chamfer_device,
        args.seed_automatically,
        args.seed_from_gui,
        args.seed_pose,
        args.gui_translation_step_m,
        args.gui_rotation_step_deg,
        args.n_seed_azimuth,
        args.n_seed_polar,
        args.n_seed_rolls,
        args.n_seed_radii,
        args.cma_sigma_pos,
        args.cma_sigma_rot,
        args.cma_maxiter,
        args.cma_popsize,
        vis,
    )
    t, q_wxyz = best_T[:3, 3], mat2quat(best_T[:3, :3])
    payload = {
        "camera": args.camera,
        "robot_id": args.robot_id,
        "camera_model_id": args.camera_model_id,
        "depth_intrinsics_source": args.depth_intrinsics_source,
        "parent_frame": "robot_base",
        "child_frame": f"{args.camera}_optical",
        "h5_path": str(args.h5_path),
        "n_timesteps": args.n_timesteps,
        "sampled_frame_indices": fps_idx.tolist(),
        "cost": float(best_cost),
        "translation": t.tolist(),
        "quaternion_wxyz": q_wxyz.tolist(),
        "matrix": best_T.tolist(),
    }
    args.output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"✅ Estimated camera extrinsics written to {args.output_path}")
    print(f"[info] cost={best_cost:.6f}")
    print(f"[info] translation={t}")
    print(f"[info] quaternion_wxyz={q_wxyz}")

    if args.visualize:
        assert vis is not None
        vis.wait()
    elif vis is not None:
        vis.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
