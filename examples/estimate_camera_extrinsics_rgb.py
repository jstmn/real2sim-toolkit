"""
This script does the following:
1. Loads the demonstration data from the h5 file
2. Loads the robot from Jrl2
3. Runs an optimization procedure to estimate the camera extrinsics.
4. Saves the estimated camera extrinsics to a yaml file.

At a high level, the optimization procedure runs CMA-ES to estimate the SE(3) pose (in the robot's base frame) of the
specified camera by comparing robot silhouettes. There are two masks that we care about: sam_mask and sim_mask.
sam_mask is the SAM 3 robot mask on the real RGB frame. sim_mask is the ground-truth segmentation of the simulated robot,
rendered by Sapien from the candidate camera pose with the real color intrinsics at that frame's joint configuration.

A SAM pixel that is not in sim_mask is an error. A sim pixel that is not in sam_mask is not necessarily an error,
because SAM 3 can return partial masks. The per-frame cost is therefore mostly one-sided:

    cost = mean_{p in sam_mask} dist(p, sim_mask) + sim_outside_sam_weight * mean_{p in sim_mask} dist(p, sam_mask)

where dist(p, mask) is the pixel distance from p to the nearest pixel of mask (0 inside it). The distance transform
(rather than a pixel count) keeps the cost informative when the silhouettes do not overlap. The second term is small
but must be > 0 in practice: with the first term alone, any pose whose sim_mask covers sam_mask is optimal, so CMA-ES
drifts toward cameras pressed against the robot whose sim_mask fills the image.

Notes:
1. A preprocessing step builds the robot masks.
    Frame 0 is SAM 3 (union of top --sam-kmax masks). That union is then propagated through every later
    RGB frame with SAM 3 ``mask_input`` + the previous mask's bbox (text prompt only on frame 0).
    The SAM mask of each sampled frame is eroded by --mask-erode-px so that SAM boundary bleed onto the background
    is not counted as an error.
2. If --visualize is set, a viser server is started. The server shows the robot model, a camera frustum at the
    estimated pose with the SAM vs sim mask overlay, and a plot of lowest population cost vs CMA-ES iteration.
3. if --visualize-robot-masks is set, debug PNGs are written for the frame-0 SAM masks that enter the union.
4. If --cache-robot-masks is set (default), each propagated mask is written as soon as it is computed to
   `<h5_dir>/<h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy`.
    A later run loads a consecutive prefix of those files and resumes SAM from the last cached frame.
    Frame 0's SAM 3 union may also be read from the older
    `robot-mask__<camera>__idx=0__...npy` seed cache. After the last frame, a dimmed-mask video is written to
    `robot-mask-propagated__<camera>__kmax=<k>__score_threshold=<t>.mp4`.
    With --per-frame-robot-masks, masks are cached per frame as
    `robot-mask-per-frame__<camera>__prompt=<p>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy` and only uncached
    frames are segmented.
5. After optimization, a SAM vs sim overlay at the estimated pose is written for every sampled frame to
    `<h5_dir>/<h5_stem>/sampled_inputs/`.



The pseudo-code is as follows:

inputs:
- rgbs: array of RGB images
- all_joint_angles: array of joint angles from the demonstration. Same length as rgbs; frame i of qpos is frame i of RGB.
- N: the number of timesteps to sample from the demonstration for the cost function
- S: number of poses in the CMA-ES population
- render_sim_mask(pose, joint_angle): Sapien robot segmentation from `pose` with the real color K
- silhouette_cost(sam_mask, sim_mask): the per-frame cost above

# Find the N joint angles that are the furthest apart from each other
timesteps, joint_angles = furthest_point_sample(all_joint_angles, N, distance='circular')

sam_masks = [erode(sam3(rgb)) for rgb in rgbs[timesteps]]

until convergence:
    sample poses p1, ..., pS from CMA-ES
    total_costs = [0] * S
    for each pose pi in population:
        total_cost = 0
        for joint_angle, sam_mask in zip(joint_angles, sam_masks):
            sim_mask = render_sim_mask(pi, joint_angle)
            total_cost += silhouette_cost(sam_mask, sim_mask)
        total_costs[i] = total_cost / N
    update CMA-ES with total_costs as fitness values

output:
- estimated pose
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import threading
import time
from typing import Any

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

from r2st.constants import MIN_DEPTH_M, get_color_intrinsics
from r2st.core import (
    SAM3Predictor,
    bbox_xyxy_from_mask,
    binary_mask_to_sam_mask_input,
)
from r2st.geometry import camera_extrinsic_to_maniskill_pose
from r2st.utils import (
    ImageUtils,
    MeshUtils,
    farthest_point_sample_naive,
    validate_merged_camera_group,
)

"""
# See README.md for example usage
"""

_SEED_RADIUS_MIN_M = 1.0
_SEED_RADIUS_MAX_M = 1.5
_SEED_LOOKAT_TARGET = np.array([0.0, 0.0, 0.5], dtype=np.float64)
_SEED_ORIGIN = np.zeros(3, dtype=np.float64)
_SEED_MIN_LOOK_DIST_M = 0.05
_SEED_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_RENDER_FAR_M = 5.0
_RENDER_CAMERA_NAME = "mask_cam"
_MASK_VIDEO_FPS = 15.0
_MIN_ROBOT_MASK_PIXELS = 400
_OVERLAY_DIM = 0.4
_OVERLAY_SAM_ONLY_RGB = np.array([255.0, 40.0, 40.0], dtype=np.float32)
_OVERLAY_SIM_ONLY_RGB = np.array([60.0, 120.0, 255.0], dtype=np.float32)
_OVERLAY_BOTH_RGB = np.array([40.0, 220.0, 90.0], dtype=np.float32)
_BEST_AXES_LENGTH = 0.08
_BEST_AXES_RADIUS = 0.004
_BEST_ORIGIN_RADIUS = 0.008
_POP_AXES_LENGTH = 0.045
_POP_AXES_RADIUS = 0.0025
_POP_ORIGIN_RADIUS = 0.005


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see data/grail_xarm_july_2/merge_camera_streams.py)."""

    robot_id: str
    """Jrl2 robot name, e.g. 'xarm7__gripper'. See the README `{robot}__{eef}` table."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    camera_model_id: str
    """Camera model id for calibration lookup, e.g. 'd435'. See the README camera-model table."""

    output_path: pathlib.Path
    """YAML path to write the estimated 4x4 camera extrinsics (robot-base T camera)."""

    n_timesteps: int = 5
    """Number of furthest-apart joint configurations used in the silhouette cost."""

    sim_outside_sam_weight: float = 0.1
    """Weight of the sim-pixels-outside-the-SAM-mask term. Keep it small (SAM masks can be partial) but > 0: with 0,
    any pose whose sim mask covers the SAM mask is optimal, including cameras pressed against the robot."""

    cma_sigma_pos: float = 0.10
    """CMA-ES initial std for camera translation (meters)."""

    cma_sigma_rot: float = 1.0
    """CMA-ES initial std for camera rotation-vector (radians)."""

    cma_maxiter: int = 250
    """Maximum CMA-ES generations."""

    cma_early_stop_atol: float = 0.01
    """Stop CMA-ES if this generation's min cost (`gen_min`, pixels) changes by less than this for `cma_early_stop_patience` gens."""

    cma_early_stop_patience: int = 5
    """Consecutive generations whose `gen_min` range is < `cma_early_stop_atol` before early stop."""

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

    cma_popsize: int = 25
    """CMA-ES population size."""

    robot_description: str = "robot arm"
    """SAM 3 text prompt used to mask the robot: on frame 0 only (later frames propagate that mask), or on every frame
    with --per-frame-robot-masks."""

    sam_kmax: int = 5
    """Union at most this many highest-confidence SAM masks into the frame-0 robot mask."""

    sam_score_threshold: float = 0.3
    """Keep a top-k SAM mask in the frame-0 union only if its confidence is strictly above this."""

    per_frame_robot_masks: bool = False
    """If set, segment the robot with the SAM 3 text prompt on every frame (union of the top --sam-kmax masks) instead
    of propagating frame 0's mask. For recordings of separate arm poses (not a continuous trajectory), where the arm
    jumps between frames and propagation stays on the static base."""

    cache_robot_masks: bool = True
    """If set, load/save robot masks as <h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy,
    or <h5_stem>/robot-mask-per-frame__<camera>__prompt=<p>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy with
    --per-frame-robot-masks."""

    mask_erode_px: int = 9
    """Erode the SAM robot mask with an elliptical kernel of this size (pixels) before the cost, so SAM bleed onto the
    background is not counted as an error. 0 skips erosion."""

    visualize: bool = False
    """If set, start a viser server with the robot model, camera frustum, SAM vs sim mask overlay, and a cost plot."""

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


def _per_frame_robot_mask_stem(camera: str, robot_description: str) -> str:
    assert "/" not in robot_description, f"robot_description must not contain '/', got {robot_description!r}"
    prompt = robot_description.replace(" ", "_")
    return f"robot-mask-per-frame__{camera}__prompt={prompt}"


def _per_frame_robot_mask_cache_path(
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    frame_idx: int,
    kmax: int,
    score_threshold: float,
) -> pathlib.Path:
    stem = _per_frame_robot_mask_stem(camera, robot_description)
    return (
        h5_path.parent
        / h5_path.stem
        / f"{stem}__idx={int(frame_idx)}__kmax={int(kmax)}__score_threshold={score_threshold}.npy"
    )


def _per_frame_robot_mask_video_path(
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    kmax: int,
    score_threshold: float,
) -> pathlib.Path:
    stem = _per_frame_robot_mask_stem(camera, robot_description)
    return h5_path.parent / h5_path.stem / f"{stem}__kmax={int(kmax)}__score_threshold={score_threshold}.mp4"


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
                seed_mask, kept = _union_top_sam_masks(ranked_masks, scores, sam_kmax, sam_score_threshold)
                print(
                    f"[info] Unioned {kept.size}/{min(sam_kmax, ranked_masks.shape[0])} masks with score > "
                    f"{sam_score_threshold} (kept ranks={kept.tolist()}, scores={scores[kept].tolist()})"
                )
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
    _save_robot_mask_demo_overlays(
        rgb,
        masks,
        propagated_paths,
        _propagated_robot_mask_video_path(h5_path, camera, sam_kmax, sam_score_threshold),
    )
    return masks


def _segment_robot_masks_per_frame(
    rgb: np.ndarray,
    frame_indices: np.ndarray,
    robot_description: str,
    sam_kmax: int,
    sam_score_threshold: float,
    cache_paths: list[pathlib.Path] | None,
) -> np.ndarray:
    """Return ``(len(frame_indices), H, W)`` robot masks for ``rgb[frame_indices]``, each from SAM 3 on its own frame
    (union of the top ``sam_kmax`` masks).

    If ``cache_paths`` (one per entry of ``frame_indices``) is given, frames with an existing cache file are loaded and
    only the rest are segmented.
    """
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    assert frame_indices.ndim == 1, f"frame_indices must be 1D, got {frame_indices.shape}"
    n_frames, height, width, _ = rgb.shape
    n_sel = frame_indices.shape[0]
    masks = np.zeros((n_sel, height, width), dtype=bool)
    todo = list(range(n_sel))
    if cache_paths is not None:
        assert len(cache_paths) == n_sel, f"cache_paths length {len(cache_paths)} != n frames {n_sel}"
        todo = [i for i in range(n_sel) if not cache_paths[i].is_file()]
        for i in sorted(set(range(n_sel)) - set(todo)):
            masks[i] = _load_cached_bool_mask(cache_paths[i], (height, width))
        print(f"[info] Loaded {n_sel - len(todo)}/{n_sel} per-frame robot masks from cache")
    if len(todo) == 0:
        return masks
    predictor = _load_sam3()
    for i in todo:
        t = int(frame_indices[i])
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        ranked_masks, scores, _ = ImageUtils.get_sam_masks_ranked(predictor, image_bgr, robot_description)
        masks[i], kept = _union_top_sam_masks(ranked_masks, scores, sam_kmax, sam_score_threshold)
        if cache_paths is not None:
            _save_cached_bool_mask(cache_paths[i], masks[i])
        print(
            f"[info] frame {t}/{n_frames}: {int(masks[i].sum())} px from {kept.size}/{ranked_masks.shape[0]} masks "
            f"(scores={np.round(scores[kept], 2).tolist()})"
        )
    return masks


def _save_robot_mask_demo_overlays(
    rgb: np.ndarray,
    masks: np.ndarray,
    mask_paths: list[pathlib.Path],
    video_path: pathlib.Path,
) -> None:
    """Write a dimmed-mask demo PNG next to every frame's mask path and an mp4 of those overlays."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert masks.shape == (n_frames, height, width), f"masks {masks.shape} != {(n_frames, height, width)}"
    assert len(mask_paths) == n_frames, f"mask_paths length {len(mask_paths)} != n_frames {n_frames}"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 via ffmpeg: OpenCV's "mp4v" (MPEG-4 Part 2) does not play in VS Code or browsers.
    writer = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(_MASK_VIDEO_FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    for t in range(n_frames):
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        overlay_path = mask_paths[t].with_suffix(".demo.png")
        demo = ImageUtils.demo_overlay(image_bgr, masks[t])
        assert overlay_path.name.endswith(".demo.png"), f"demo overlay path must end with .demo.png, got {overlay_path}"
        assert cv2.imwrite(str(overlay_path), demo), f"Failed to write {overlay_path}"
        writer.stdin.write(demo.tobytes())
    writer.stdin.close()
    assert writer.wait() == 0, f"ffmpeg failed writing {video_path}"
    print(f"[info] Saved {n_frames} robot-mask demo overlays and video to {video_path}")


def _union_top_sam_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    kmax: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """OR-union of the first `kmax` ranked masks whose score is strictly above `score_threshold`.

    Returns the `(H, W)` union and the kept ranks.
    """
    assert masks.ndim == 3, f"masks must be (N, H, W), got {masks.shape}"
    assert scores.shape == (masks.shape[0],), f"scores {scores.shape} != n_masks {masks.shape[0]}"
    assert kmax >= 1, f"kmax must be >= 1, got {kmax}"
    n = min(int(kmax), masks.shape[0])
    keep = scores[:n] > score_threshold
    n_keep = int(keep.sum())
    assert n_keep >= 1, f"No SAM masks in top {n} with score > {score_threshold} " f"(scores={scores[:n].tolist()})"
    union = np.any(masks[:n][keep], axis=0)
    return union, np.flatnonzero(keep)


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


def _mask_comparison_overlay(rgb: np.ndarray, sam_mask: np.ndarray, sim_mask: np.ndarray | None) -> np.ndarray:
    """Dimmed RGB with SAM-only pixels (the error) in red, sim-only in blue, and both in green. ``sim_mask=None`` shows
    the SAM mask alone (in green)."""
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
    assert sam_mask.shape == rgb.shape[:2], f"sam_mask {sam_mask.shape} != rgb {rgb.shape[:2]}"
    if sim_mask is None:
        sim_mask = sam_mask
    assert sim_mask.shape == rgb.shape[:2], f"sim_mask {sim_mask.shape} != rgb {rgb.shape[:2]}"
    rgb_f = rgb.astype(np.float32)
    out = _OVERLAY_DIM * rgb_f
    for region, color in (
        (sam_mask & ~sim_mask, _OVERLAY_SAM_ONLY_RGB),
        (~sam_mask & sim_mask, _OVERLAY_SIM_ONLY_RGB),
        (sam_mask & sim_mask, _OVERLAY_BOTH_RGB),
    ):
        out[region] = _OVERLAY_DIM * rgb_f[region] + (1.0 - _OVERLAY_DIM) * color
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _sampled_frame_prefix(h5_path: pathlib.Path, camera: str, sample_idx: int, frame_idx: int) -> pathlib.Path:
    return h5_path.parent / h5_path.stem / "sampled_inputs" / f"{camera}__sample={sample_idx}__idx={frame_idx}"


def _save_sampled_frame_inputs(
    h5_path: pathlib.Path,
    camera: str,
    sample_idx: int,
    frame_idx: int,
    rgb: np.ndarray,
    mask: np.ndarray,
) -> pathlib.Path:
    """Write the RGB and eroded SAM mask overlay that a sampled frame contributes to the cost.

    Returns the output directory.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
    assert mask.shape == rgb.shape[:2], f"mask {mask.shape} != rgb {rgb.shape[:2]}"
    prefix = _sampled_frame_prefix(h5_path, camera, sample_idx, frame_idx)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = prefix.with_name(prefix.name + "__rgb.png")
    overlay_path = prefix.with_name(prefix.name + "__mask_overlay.png")
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(rgb_path), image_bgr), f"Failed to write {rgb_path}"
    assert cv2.imwrite(str(overlay_path), ImageUtils.demo_overlay(image_bgr, mask)), f"Failed to write {overlay_path}"
    return prefix.parent


def _save_best_pose_overlays(
    h5_path: pathlib.Path,
    camera: str,
    fps_idx: np.ndarray,
    rgb_sel: np.ndarray,
    sam_masks: list[np.ndarray],
    sim_masks: list[np.ndarray],
) -> None:
    """Write the SAM vs sim mask overlay at the estimated pose next to each sampled frame's inputs."""
    assert len(sam_masks) == len(sim_masks) == fps_idx.shape[0] == rgb_sel.shape[0], (
        f"sam_masks {len(sam_masks)}, sim_masks {len(sim_masks)}, fps_idx {fps_idx.shape[0]}, "
        f"rgb_sel {rgb_sel.shape[0]} must match"
    )
    for i, frame_idx in enumerate(fps_idx):
        prefix = _sampled_frame_prefix(h5_path, camera, i, int(frame_idx))
        path = prefix.with_name(prefix.name + "__best_pose_sam_vs_sim.png")
        overlay = _mask_comparison_overlay(rgb_sel[i], sam_masks[i], sim_masks[i])
        assert cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)), f"Failed to write {path}"
    print(f"[info] Saved best-pose SAM (red: outside sim) vs sim (blue) overlays to {path.parent}")


def _load_demo(h5_path: pathlib.Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """Return rgb and qpos, aligned by frame index (same N)."""
    with h5py.File(h5_path, "r") as f:
        validate_merged_camera_group(f, h5_path, camera)
        assert (
            "obs/qpos" in f
        ), f"{h5_path}: missing 'obs/qpos' (re-merge the demo with data/grail_xarm_july_2/merge_camera_streams.py)"
        rgb = f[f"obs/sensor_data/{camera}/rgb"][:]
        qpos = np.asarray(f["obs/qpos"][:])
    assert qpos.ndim == 2, f"{h5_path}: obs/qpos must be NxD, got {qpos.shape}"
    assert qpos.shape[0] > 0, f"{h5_path}: obs/qpos is empty"
    assert qpos.shape[0] == rgb.shape[0], (
        f"{h5_path}: obs/qpos has {qpos.shape[0]} frames, {camera} rgb has {rgb.shape[0]} "
        "(qpos and RGB must share the same index axis)"
    )
    return rgb, qpos


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


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Per-pixel L2 distance (pixels) to the nearest pixel of non-empty bool `mask`; 0 inside it."""
    assert mask.ndim == 2 and mask.dtype == bool, f"mask must be a 2D bool array, got {mask.shape} {mask.dtype}"
    assert mask.any(), "Cannot compute a distance transform to an empty mask"
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


class SimRobotRenderer:
    """Sapien scene with the Jrl2 URDF and one camera with the real color intrinsics. Renders robot silhouettes."""

    def __init__(self, urdf_path: pathlib.Path, K: np.ndarray, height: int, width: int) -> None:
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0, f"height and width must be > 0, got {height}x{width}"
        sapien_urdf = _urdf_for_sapien(urdf_path)
        assert sapien_urdf.is_file(), f"Sapien URDF not found: {sapien_urdf}"

        self._scene = sapien.Scene()
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
        # Segmentation channel 1 holds each entity's per_scene_id
        self._robot_seg_ids = np.array(
            sorted({link.entity.per_scene_id for link in self._robot.get_links()}), dtype=np.uint32
        )

        self._hw = (int(height), int(width))
        fx, fy, cx, cy, skew = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), float(K[0, 1])
        assert fx > 0.0 and fy > 0.0, f"K focal lengths must be > 0, got fx={fx} fy={fy}"
        self._cam = self._scene.add_camera(
            _RENDER_CAMERA_NAME,
            int(width),
            int(height),
            fovy=float(2.0 * np.arctan(height / (2.0 * fy))),
            near=MIN_DEPTH_M,
            far=_RENDER_FAR_M,
        )
        self._cam.set_perspective_parameters(MIN_DEPTH_M, _RENDER_FAR_M, fx, fy, cx, cy, skew)
        K_render = np.asarray(self._cam.get_intrinsic_matrix(), dtype=np.float64)
        assert np.allclose(K_render, K, atol=1e-3), f"Sapien camera K\n{K_render}\n!= color K\n{K}"

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

    def render_mask(self, T_world_cam: np.ndarray) -> np.ndarray:
        """`(H, W)` bool robot mask seen from OpenCV `T_world_cam` at the current qpos."""
        self._cam.local_pose = _T_to_sapien_pose(T_world_cam)
        self._scene.update_render()
        self._cam.take_picture()
        segmentation = self._cam.get_picture("Segmentation")
        assert segmentation.shape[:2] == self._hw, f"Segmentation {segmentation.shape} != {self._hw}"
        return np.isin(segmentation[:, :, 1], self._robot_seg_ids)


class MaskPoseCost:
    """Mean per-frame silhouette cost of a candidate `T_world_cam` over the sampled timesteps (see module docstring)."""

    def __init__(
        self,
        renderer: SimRobotRenderer,
        joint_angles: np.ndarray,
        sam_masks: list[np.ndarray],
        sim_outside_sam_weight: float,
    ) -> None:
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        assert len(sam_masks) >= 1, "sam_masks must be non-empty"
        assert joint_angles.shape == (
            len(sam_masks),
            renderer.dof,
        ), f"joint_angles {joint_angles.shape} != ({len(sam_masks)}, {renderer.dof})"
        assert sim_outside_sam_weight >= 0.0, f"sim_outside_sam_weight must be >= 0, got {sim_outside_sam_weight}"
        height, width = sam_masks[0].shape
        for i, mask in enumerate(sam_masks):
            assert mask.shape == (height, width), f"sam_masks[{i}] {mask.shape} != {(height, width)}"
            assert mask.dtype == bool and mask.any(), f"sam_masks[{i}] must be a non-empty bool mask"
        self._renderer = renderer
        self._joint_angles = joint_angles
        self._sam_masks = sam_masks
        self._sam_dists = [_distance_to_mask(mask) for mask in sam_masks]
        self._w = float(sim_outside_sam_weight)
        # Each term is at most the image diagonal, so an empty render costs more than any non-empty one
        self._empty_sim_cost = float(np.hypot(height, width)) * (1.0 + self._w)

    def __call__(self, T_world_cam: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Return (mean cost in pixels, per-timestep sim masks)."""
        total = 0.0
        sim_masks: list[np.ndarray] = []
        for q, sam_mask, sam_dist in zip(self._joint_angles, self._sam_masks, self._sam_dists, strict=True):
            self._renderer.set_qpos(q)
            sim_mask = self._renderer.render_mask(T_world_cam)
            sim_masks.append(sim_mask)
            if not sim_mask.any():
                total += self._empty_sim_cost
                continue
            total += float(_distance_to_mask(sim_mask)[sam_mask].mean())
            total += self._w * float(sam_dist[sim_mask].mean())
        return total / len(self._sam_masks), sim_masks


@dataclasses.dataclass
class _ManualSeedSelection:
    T_world_cam: np.ndarray
    translation_step_handle: Any
    rotation_step_handle: Any
    move_buttons: list[Any]
    command_handles: list[Any]
    selected_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    wake_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    T_evaluated: np.ndarray | None = None
    cost: float | None = None
    sim_masks: list[np.ndarray] | None = None
    best_cost: float = np.inf
    t0: float = dataclasses.field(default_factory=time.perf_counter)


class ExtrinsicsVisualizer:
    """Viser overlay in the robot-base frame: robot model, camera frustum, SAM vs sim masks, CMA-ES cost plot.

    `renderer` only poses the displayed robot meshes. Pass a different instance than the one inside `cost_fn`: the
    timestep slider sets its qpos from the Viser thread.
    """

    def __init__(
        self,
        sam_masks: list[np.ndarray],
        K: np.ndarray,
        height: int,
        width: int,
        rgb_frames: np.ndarray,
        cma_maxiter: int,
        fps_idx: np.ndarray,
        renderer: SimRobotRenderer,
        joint_angles: np.ndarray,
        cost_fn: MaskPoseCost,
    ) -> None:
        import viser
        from viser import uplot

        assert len(sam_masks) >= 1
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0
        assert rgb_frames.shape == (
            len(sam_masks),
            height,
            width,
            3,
        ), f"rgb_frames must be ({len(sam_masks)}, {height}, {width}, 3), got {rgb_frames.shape}"
        assert cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {cma_maxiter}"
        fps_idx = np.asarray(fps_idx, dtype=np.int64).reshape(-1)
        assert fps_idx.shape == (len(sam_masks),), f"fps_idx {fps_idx.shape} != n masks {len(sam_masks)}"
        fy = float(K[1, 1])
        assert fy > 0, f"K[1,1] (fy) must be > 0, got {fy}"

        self._sam_masks = sam_masks
        self._rgb_frames = rgb_frames
        self._fps_idx = fps_idx
        self._renderer = renderer
        self._joint_angles = np.asarray(joint_angles, dtype=np.float64)
        self._cost_fn = cost_fn
        assert cost_fn._renderer is not renderer, "Visualizer renderer must not be the cost renderer"
        assert self._joint_angles.ndim == 2, f"joint_angles must be (T, dof), got {self._joint_angles.shape}"
        assert self._joint_angles.shape == (
            len(sam_masks),
            renderer.dof,
        ), f"joint_angles {self._joint_angles.shape} != ({len(sam_masks)}, {renderer.dof})"
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
        self._sim_masks_hist: list[list[np.ndarray] | None] = [None] * cma_maxiter
        self._pop_Ts_hist: list[list[np.ndarray] | None] = [None] * cma_maxiter
        self._best_costs: list[float | None] = [None] * cma_maxiter
        self._n_recorded = 0
        self._live_sim_masks: list[np.ndarray] | None = None
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

        overlay0 = _mask_comparison_overlay(rgb_frames[0], sam_masks[0], None)
        wxyz0, pos0 = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum = self._server.scene.add_camera_frustum(
            "/camera",
            fov=float(2.0 * np.arctan(height / (2.0 * fy))),
            aspect=width / height,
            scale=0.04,
            line_width=1.5,
            color=(40, 120, 255),
            image=overlay0,
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
        self._server.gui.add_markdown("**red:** SAM outside sim (error) · **blue:** sim only · **green:** both")
        self._overlay_image = self._server.gui.add_image(overlay0, label="SAM vs sim mask", format="jpeg")
        timestep_slider = self._server.gui.add_slider(
            "timestep (sampled)", min=0, max=max(len(sam_masks) - 1, 0), step=1, initial_value=0
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
                uplot.Axis(label="cost (px)"),
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
                uplot.Axis(label="cost (px)"),
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
        """Block until the user adjusts and selects a CMA-ES seed in the Viser GUI.

        GUI callbacks only move the pose; this (calling) thread renders and evaluates it, so Sapien is only used from
        the thread that runs CMA-ES.
        """
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

        print("[info] Waiting for manual seed selection in Viser (click the 3D view, then RF/TG/YH/UJ/IK/OL/Enter)...")
        state.wake_event.set()
        while True:
            state.wake_event.wait()
            with state.lock:
                state.wake_event.clear()
                T = state.T_world_cam.copy()
                selected = state.selected_event.is_set()
            if state.T_evaluated is None or not np.array_equal(T, state.T_evaluated):
                self._evaluate_manual_seed(T)
            if selected:
                break
        assert state.cost is not None and state.sim_masks is not None and state.T_evaluated is not None
        self._manual_seed_status.content = f"**Selected seed cost:** `{state.cost:.4f}` px"
        return state.T_evaluated.copy(), float(state.cost), [mask.copy() for mask in state.sim_masks]

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
            state.wake_event.set()

    def _evaluate_manual_seed(self, T_world_cam: np.ndarray) -> None:
        state = self._manual_seed
        assert state is not None, "Manual seed selection is not configured"
        assert self._manual_seed_status is not None
        self._manual_seed_status.content = "**Evaluating pose...**"
        cost, sim_masks = self._cost_fn(T_world_cam)
        state.T_evaluated = T_world_cam.copy()
        state.cost = float(cost)
        state.sim_masks = sim_masks
        state.best_cost = min(state.best_cost, state.cost)
        assert np.isfinite(state.best_cost), f"best_cost must be finite, got {state.best_cost}"
        self.record_seed_best(time.perf_counter() - state.t0, state.best_cost)
        self.set_best(sim_masks, state.cost, T_world_cam)
        self._manual_seed_status.content = (
            f"**Current cost:** `{state.cost:.4f}` px  \n"
            f"**Best cost:** `{state.best_cost:.4f}` px  \n"
            f"{_pose_translation_rotation_text(T_world_cam)}"
        )

    def _confirm_manual_seed(self) -> None:
        state = self._manual_seed
        assert state is not None, "Manual seed selection is not configured"
        with state.lock:
            for handle in state.move_buttons:
                handle.disabled = True
            for handle in state.command_handles:
                handle.remove()
            state.command_handles.clear()
            state.translation_step_handle.disabled = True
            state.rotation_step_handle.disabled = True
            state.selected_event.set()
            state.wake_event.set()

    def set_best(self, sim_masks: list[np.ndarray], cost: float, T_world_cam: np.ndarray) -> None:
        """Live-update the scene when a new global best is found, if viewing the latest iteration."""
        assert len(sim_masks) == len(self._sam_masks), f"sim masks {len(sim_masks)} != SAM masks {len(self._sam_masks)}"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        if self._n_recorded > 0 and self._iteration != self._n_recorded:
            return
        self._live_sim_masks = [m.copy() for m in sim_masks]
        self._apply(T=T_world_cam.copy(), sim_masks=self._live_sim_masks, cost=float(cost))

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
        """Append one seed-search sample: wall time (s) vs best silhouette cost so far."""
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
        sim_masks: list[np.ndarray],
        pop_Ts: list[np.ndarray],
    ) -> None:
        assert 1 <= generation <= self._maxiter, f"generation {generation} not in [1, {self._maxiter}]"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        assert len(sim_masks) == len(self._sam_masks), f"sim masks {len(sim_masks)} != SAM masks {len(self._sam_masks)}"
        assert len(pop_Ts) >= 1, "population must contain at least one pose"
        for i, T_pop in enumerate(pop_Ts):
            assert T_pop.shape == (4, 4), f"pop_Ts[{i}] must be 4x4, got {T_pop.shape}"
        i = generation - 1
        self._ys_best[i] = float(best_cost)
        self._ys_pop[i] = float(pop_min)
        self._Ts[i] = T_world_cam.copy()
        self._sim_masks_hist[i] = [m.copy() for m in sim_masks]
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
        sim_masks: list[np.ndarray] | None = None,
        cost: float | None = None,
    ) -> None:
        t = self._timestep
        snap_i = self._snapshot_index()
        pop_Ts: list[np.ndarray] | None = None
        if T is None and self._iteration == 0 and self._manual_seed is not None and self._manual_seed.cost is not None:
            # The GUI seed is the CMA-ES start, so iteration 0 shows it (and keeps it when the timestep changes)
            T = self._manual_seed.T_evaluated.copy()
            sim_masks = self._manual_seed.sim_masks
            cost = self._manual_seed.cost
            pop_Ts = []
        elif T is None and self._iteration == 0:
            T = self._T_init
            sim_masks = None
            cost = None
            pop_Ts = []
        elif T is None and snap_i is not None:
            T = self._Ts[snap_i]
            sim_masks = self._sim_masks_hist[snap_i]
            cost = self._best_costs[snap_i]
            pop_Ts = self._pop_Ts_hist[snap_i]
            assert T is not None, f"Missing T snapshot at iteration {snap_i + 1}"
            assert sim_masks is not None, f"Missing sim-mask snapshot at iteration {snap_i + 1}"
            assert cost is not None, f"Missing cost snapshot at iteration {snap_i + 1}"
            assert pop_Ts is not None, f"Missing population snapshot at iteration {snap_i + 1}"
        elif T is None and sim_masks is None:
            sim_masks = self._live_sim_masks
            pop_Ts = self._live_pop_Ts
        if T is not None:
            self._T = T
        demo_frame = int(self._fps_idx[t])
        if cost is not None:
            self._cost_md.content = f"**iter {self._iteration} best cost:** `{cost:.4f}` px"
        elif self._iteration == 0:
            self._cost_md.content = "**iter 0 (init)**"
        self._pose_md.content = (
            f"**sampled t={t}  demo frame={demo_frame}**\n\n{_pose_translation_rotation_text(self._T)}"
        )
        overlay = _mask_comparison_overlay(
            self._rgb_frames[t], self._sam_masks[t], None if sim_masks is None else sim_masks[t]
        )
        self._overlay_image.image = overlay
        wxyz, pos = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum.wxyz = wxyz
        self._frustum.position = pos
        self._frustum.image = overlay
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


def _select_seed_pose(
    seed_Ts: list[np.ndarray],
    cost_fn: MaskPoseCost,
    vis: ExtrinsicsVisualizer | None,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Evaluate look-at seed poses and return the lowest-cost (T, cost, sim_masks)."""
    assert len(seed_Ts) >= 1, "seed_Ts must be non-empty"
    best_cost = np.inf
    best_T: np.ndarray | None = None
    best_sim_masks: list[np.ndarray] | None = None
    xyz_Ts: list[np.ndarray] = []
    t0_seed = time.perf_counter()
    pbar = tqdm(seed_Ts, desc="seed poses")
    for T in pbar:
        cost, sim_masks = cost_fn(T)
        xyz = T[:3, 3]
        new_xyz = len(xyz_Ts) == 0 or not np.allclose(xyz_Ts[-1][:3, 3], xyz)
        if new_xyz:
            xyz_Ts.append(T.copy())
            if vis is not None:
                vis.set_population(xyz_Ts)
        if cost < best_cost:
            best_cost = cost
            best_T = T.copy()
            best_sim_masks = sim_masks
            if vis is not None:
                vis.set_best(sim_masks, best_cost, best_T)
        if vis is not None and np.isfinite(best_cost):
            vis.record_seed_best(time.perf_counter() - t0_seed, best_cost)
        pbar.set_postfix(best=f"{best_cost:.4f}")
    assert best_T is not None and best_sim_masks is not None, "Seed search produced no pose"
    return best_T, best_cost, best_sim_masks


def _optimize_extrinsics(
    cost_fn: MaskPoseCost,
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
    cma_early_stop_atol: float,
    cma_early_stop_patience: int,
    vis: ExtrinsicsVisualizer | None,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """CMA-ES over 6D camera pose. Returns (best 4x4 T_world_cam, cost, sim masks at that pose)."""
    import cma

    assert n_seed_azimuth >= 1, f"n_seed_azimuth must be >= 1, got {n_seed_azimuth}"
    assert n_seed_polar >= 1, f"n_seed_polar must be >= 1, got {n_seed_polar}"
    assert n_seed_rolls >= 1, f"n_seed_rolls must be >= 1, got {n_seed_rolls}"
    assert n_seed_radii >= 1, f"n_seed_radii must be >= 1, got {n_seed_radii}"
    assert cma_sigma_pos > 0, f"cma_sigma_pos must be > 0, got {cma_sigma_pos}"
    assert cma_sigma_rot > 0, f"cma_sigma_rot must be > 0, got {cma_sigma_rot}"
    assert cma_early_stop_atol > 0, f"cma_early_stop_atol must be > 0, got {cma_early_stop_atol}"
    assert cma_early_stop_patience >= 1, f"cma_early_stop_patience must be >= 1, got {cma_early_stop_patience}"
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
        best_T, best_cost, best_sim_masks = _select_seed_pose(seed_Ts, cost_fn, vis)
        _log_elapsed(f"Seed search finished, best={best_cost:.4f}", t0_seed)
    elif seed_from_gui:
        assert vis is not None, "GUI seed selection requires an ExtrinsicsVisualizer"
        best_T, best_cost, best_sim_masks = vis.select_seed_from_gui(
            gui_translation_step_m,
            gui_rotation_step_deg,
        )
        print(f"[info] Manual seed selected: cost={best_cost:.4f} T_world_cam=\n{best_T}")
    else:
        assert seed_pose is not None, "seed_pose is required when not using --seed-automatically or --seed-from-gui"
        best_T = _pose_from_xyz_wxyz(seed_pose)
        t0_seed = time.perf_counter()
        best_cost, best_sim_masks = cost_fn(best_T)
        if vis is not None:
            vis.set_best(best_sim_masks, best_cost, best_T)
            vis.record_seed_best(time.perf_counter() - t0_seed, best_cost)
        q_wxyz = mat2quat(best_T[:3, :3])
        print(
            f"[info] Seed pose xyz={best_T[:3, 3].tolist()} wxyz={q_wxyz.tolist()} "
            f"cost={best_cost:.4f} T_world_cam=\n{best_T}"
        )

    x0 = _T_to_vec(best_T)
    cma_stds = [cma_sigma_pos, cma_sigma_pos, cma_sigma_pos, cma_sigma_rot, cma_sigma_rot, cma_sigma_rot]
    cma_opts: dict = {"maxiter": cma_maxiter, "verbose": -1, "CMA_stds": cma_stds}
    if cma_popsize is not None:
        cma_opts["popsize"] = cma_popsize
    es = cma.CMAEvolutionStrategy(x0, 1.0, cma_opts)
    print(
        f"[info] CMA-ES x0={x0} sigma_pos={cma_sigma_pos} sigma_rot={cma_sigma_rot} "
        f"popsize={es.popsize} maxiter={cma_maxiter} "
        f"early_stop_atol={cma_early_stop_atol} early_stop_patience={cma_early_stop_patience}"
    )

    generation = 0
    recent_gen_mins: list[float] = []
    t0_opt = time.perf_counter()
    while not es.stop():
        generation += 1
        t0_gen = time.perf_counter()
        t0 = time.perf_counter()
        xs = es.ask()
        t_cma = time.perf_counter() - t0
        costs = []
        pop_Ts: list[np.ndarray] = []
        t_cost = 0.0
        t_vis = 0.0
        pbar = tqdm(xs, desc=f"cma gen {generation}", leave=False)
        for x in pbar:
            T = _vec_to_T(x)
            t0 = time.perf_counter()
            cost, sim_masks = cost_fn(T)
            t_cost += time.perf_counter() - t0
            costs.append(cost)
            pop_Ts.append(T.copy())
            if vis is not None:
                t0 = time.perf_counter()
                vis.set_population(pop_Ts)
                t_vis += time.perf_counter() - t0
            if cost < best_cost:
                best_cost = cost
                best_T = T.copy()
                best_sim_masks = sim_masks
                if vis is not None:
                    t0 = time.perf_counter()
                    vis.set_best(sim_masks, best_cost, best_T)
                    t_vis += time.perf_counter() - t0
            pbar.set_postfix(best=f"{best_cost:.4f}")
        t0 = time.perf_counter()
        es.tell(xs, costs)
        t_cma += time.perf_counter() - t0
        pop_min = float(min(costs))
        assert best_T is not None and best_sim_masks is not None, "CMA-ES produced no pose"
        if vis is not None:
            t0 = time.perf_counter()
            vis.record_generation(generation, best_cost, pop_min, best_T, best_sim_masks, pop_Ts)
            t_vis += time.perf_counter() - t0
        dt_gen = time.perf_counter() - t0_gen
        t_other = dt_gen - (t_cost + t_vis + t_cma)
        print(
            f"[info] gen {generation} ({dt_gen:.3f}s): best={best_cost:.4f}  gen_min={pop_min:.4f}  "
            f"gen_mean={float(np.mean(costs)):.4f}"
        )
        print(f"[info]   time_cost={t_cost:.3f}s")
        print(f"[info]   time_vis={t_vis:.3f}s")
        print(f"[info]   time_cma={t_cma:.3f}s")
        print(f"[info]   time_other={t_other:.3f}s")
        recent_gen_mins.append(pop_min)
        if len(recent_gen_mins) > cma_early_stop_patience:
            recent_gen_mins = recent_gen_mins[-cma_early_stop_patience:]
        if len(recent_gen_mins) == cma_early_stop_patience:
            gen_min_spread = max(recent_gen_mins) - min(recent_gen_mins)
            if gen_min_spread < cma_early_stop_atol:
                print(
                    f"[info] CMA-ES early stop at gen {generation}: gen_min changed by < {cma_early_stop_atol} "
                    f"over {cma_early_stop_patience} consecutive gens "
                    f"(gen_min={pop_min:.4f} spread={gen_min_spread:.4f} best={best_cost:.4f})"
                )
                break

    assert best_T is not None and best_sim_masks is not None, "CMA-ES produced no pose"
    _log_elapsed(f"CMA-ES finished ({generation} gens)", t0_opt)
    return best_T, best_cost, best_sim_masks


def main(args: Args) -> None:
    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert (
        args.robot_id.lower() in NAME_TO_ROBOT
    ), f"Unknown robot_id {args.robot_id!r}. Available: {sorted(NAME_TO_ROBOT)}"
    assert args.n_timesteps >= 1, f"n_timesteps must be >= 1, got {args.n_timesteps}"
    assert args.sim_outside_sam_weight >= 0.0, f"sim_outside_sam_weight must be >= 0, got {args.sim_outside_sam_weight}"
    assert args.cma_sigma_pos > 0, f"cma_sigma_pos must be > 0, got {args.cma_sigma_pos}"
    assert args.cma_sigma_rot > 0, f"cma_sigma_rot must be > 0, got {args.cma_sigma_rot}"
    assert args.cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {args.cma_maxiter}"
    assert args.cma_popsize is None or args.cma_popsize >= 2, f"cma_popsize must be >= 2, got {args.cma_popsize}"
    assert args.cma_early_stop_atol > 0, f"cma_early_stop_atol must be > 0, got {args.cma_early_stop_atol}"
    assert (
        args.cma_early_stop_patience >= 1
    ), f"cma_early_stop_patience must be >= 1, got {args.cma_early_stop_patience}"
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

    color_intrinsics = get_color_intrinsics(args.camera_model_id)

    print(f"[info] Loading RGB + qpos from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb_all, qpos_all = _load_demo(args.h5_path, args.camera)
    _log_elapsed(f"Loaded rgb{rgb_all.shape} qpos{qpos_all.shape}", t0)

    n_frames = rgb_all.shape[0]
    assert args.n_timesteps <= n_frames, f"n_timesteps={args.n_timesteps} > n_frames={n_frames}"
    H, W = int(rgb_all.shape[1]), int(rgb_all.shape[2])
    assert (H, W) == (color_intrinsics.height, color_intrinsics.width), (
        f"{args.camera} rgb is {H}x{W} but {args.camera_model_id} color intrinsics are "
        f"{color_intrinsics.height}x{color_intrinsics.width}"
    )
    K = color_intrinsics.intrinsic_matrix.astype(np.float64)
    print(f"[info] Color K ({args.camera_model_id}):\n{K}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    t0_masks = time.perf_counter()
    if args.per_frame_robot_masks:
        per_frame_paths = [
            _per_frame_robot_mask_cache_path(
                args.h5_path, args.camera, args.robot_description, t, args.sam_kmax, args.sam_score_threshold
            )
            for t in range(n_frames)
        ]
        masks_all = _segment_robot_masks_per_frame(
            rgb_all,
            np.arange(n_frames),
            args.robot_description,
            args.sam_kmax,
            args.sam_score_threshold,
            per_frame_paths if args.cache_robot_masks else None,
        )
        _save_robot_mask_demo_overlays(
            rgb_all,
            masks_all,
            per_frame_paths,
            _per_frame_robot_mask_video_path(
                args.h5_path, args.camera, args.robot_description, args.sam_kmax, args.sam_score_threshold
            ),
        )
    else:
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
    assert masks_all.shape == (n_frames, H, W), f"masks_all {masks_all.shape} != ({n_frames}, {H}, {W})"
    mask_pixels = masks_all.reshape(n_frames, -1).sum(axis=1)
    lost = np.flatnonzero(mask_pixels <= _MIN_ROBOT_MASK_PIXELS)
    assert lost.size == 0, (
        f"Robot mask has <= {_MIN_ROBOT_MASK_PIXELS} pixels at {lost.size}/{n_frames} frames "
        f"(lost tracking or bad segmentation). First frames: {[(int(t), int(mask_pixels[t])) for t in lost[:10]]}"
    )
    _log_elapsed(f"Built robot masks for {n_frames} frames", t0_masks)

    fps_local = farthest_point_sample_naive(qpos_all.astype(np.float64), args.n_timesteps, distance="circular")
    fps_idx = np.asarray(fps_local, dtype=np.int64)
    print(f"[info] FPS circular joint samples at RGB frames {fps_idx.tolist()}")

    rgb_sel = rgb_all[fps_idx]
    qpos_sel = qpos_all[fps_idx]
    sam_masks: list[np.ndarray] = []
    for i, frame_idx in enumerate(fps_idx):
        frame_idx = int(frame_idx)
        mask = masks_all[frame_idx]
        n_mask = int(mask.sum())
        mask = _erode_mask(mask, args.mask_erode_px)
        sam_masks.append(mask)
        inputs_dir = _save_sampled_frame_inputs(args.h5_path, args.camera, i, frame_idx, rgb_sel[i], mask)
        print(f"[info] sam_mask[{i}] frame {frame_idx}: {n_mask} -> {int(mask.sum())} px eroded")
    print(f"[info] Saved sampled-frame RGB / mask images to {inputs_dir}")

    print(f"[info] Loading Jrl2 robot '{args.robot_id}' ...")
    t0 = time.perf_counter()
    robot = get_robot_by_name(args.robot_id)
    urdf_path = pathlib.Path(robot._urdf_filepath)
    print(f"[info] URDF: {urdf_path}")
    renderer = SimRobotRenderer(urdf_path, K, H, W)
    qpos_sel = _expand_demo_qpos_to_joint_names(qpos_sel, robot, renderer.joint_names)
    cost_fn = MaskPoseCost(renderer, qpos_sel, sam_masks, args.sim_outside_sam_weight)
    _log_elapsed("Loaded robot + Sapien renderer", t0)

    vis = (
        ExtrinsicsVisualizer(
            sam_masks,
            K,
            H,
            W,
            rgb_sel,
            args.cma_maxiter,
            fps_idx,
            SimRobotRenderer(urdf_path, K, H, W),
            qpos_sel,
            cost_fn,
        )
        if args.visualize or args.seed_from_gui
        else None
    )
    best_T, best_cost, best_sim_masks = _optimize_extrinsics(
        cost_fn,
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
        args.cma_early_stop_atol,
        args.cma_early_stop_patience,
        vis,
    )
    _save_best_pose_overlays(args.h5_path, args.camera, fps_idx, rgb_sel, sam_masks, best_sim_masks)
    t, q_wxyz = best_T[:3, 3], mat2quat(best_T[:3, :3])
    payload = {
        "camera": args.camera,
        "robot_id": args.robot_id,
        "camera_model_id": args.camera_model_id,
        "parent_frame": "robot_base",
        "child_frame": f"{args.camera}_optical",
        "h5_path": str(args.h5_path),
        "n_timesteps": args.n_timesteps,
        "sampled_frame_indices": fps_idx.tolist(),
        "cost": float(best_cost),
        "sim_outside_sam_weight": float(args.sim_outside_sam_weight),
        "translation": t.tolist(),
        "quaternion_wxyz": q_wxyz.tolist(),
        "matrix": best_T.tolist(),
    }
    args.output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"✅ Estimated camera extrinsics written to {args.output_path}")
    print(f"[info] cost={best_cost:.4f} px")
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
