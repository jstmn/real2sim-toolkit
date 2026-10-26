"""Shared camera-extrinsics calibration code for examples/estimate_camera_extrinsics.py (pointcloud cost) and
examples/estimate_camera_extrinsics_rgb.py (silhouette cost).

Both scripts estimate the robot-base T camera pose with CMA-ES. They share the robot masks (SAM 3), demo loading,
seed poses, the CMA-ES loop, and the Viser view. Each script supplies a cost function ``cost_fn(T_world_cam) ->
(cost, sims)``, where ``sims`` holds one simulated observation per sampled timestep (sim clouds or sim masks), and an
``ExtrinsicsVisualizerBase`` subclass that draws those observations.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import pathlib
import shutil
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import h5py
import numpy as np
import sapien
import yaml
from termcolor import colored
from tqdm import tqdm
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from r2st.constants import (
    CAMERA_MOUNT_RADIUS_M,
    MIN_DEPTH_M,
    get_depth_intrinsics,
    get_depth_to_color_extrinsics,
    get_workspace_bounds,
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
    reproject_depth_to_color_frame,
)
from r2st.types import CameraIntrinsics, WorkspaceBounds
from r2st.utils import (
    H264VideoWriter,
    ImageUtils,
    MeshUtils,
    farthest_point_sample_pyg_lib,
    validate_merged_camera_group,
)

CostFn = Callable[[np.ndarray], tuple[float, list[Any]]]
"""``cost_fn(T_world_cam) -> (cost, sims)``: lower is better, ``sims`` has one entry per sampled timestep."""

SEED_RADIUS_MIN_M = 1.0
SEED_RADIUS_MAX_M = 1.5
SEED_LOOKAT_TARGET = np.array([0.0, 0.0, 0.5], dtype=np.float64)
SEED_ORIGIN = np.zeros(3, dtype=np.float64)
SEED_MIN_LOOK_DIST_M = 0.05
SEED_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
MASK_VIDEO_FPS = 15.0
MIN_ROBOT_MASK_PIXELS = 400
BEST_AXES_LENGTH = 0.08
BEST_AXES_RADIUS = 0.004
BEST_ORIGIN_RADIUS = 0.008
POP_AXES_LENGTH = 0.045
POP_AXES_RADIUS = 0.0025
POP_ORIGIN_RADIUS = 0.005
# Padding on every side of the workspace bounds except the bottom. The arm reaches past the table edge toward a camera,
# and the prior extrinsics are approximate.
WORKSPACE_PADDING_M = 0.0
# Padding below the workspace bounds. Small, so the floor below the table stays background, but under the prior
# extrinsics the table surface measures a few cm below its true height (median -2.5 cm, 1st percentile -4.5 cm).
WORKSPACE_BOTTOM_PADDING_M = 0.05


def log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


def validate_seed_mode(
    seed_automatically: bool,
    seed_from_gui: bool,
    seed_pose: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    n_modes = int(seed_automatically) + int(seed_from_gui) + int(seed_pose is not None)
    assert n_modes == 1, "Exactly one of --seed-automatically, --seed-from-gui, or --seed-pose must be passed"


def _filtering_dir(h5_path: pathlib.Path, workspace_tag: str | None) -> pathlib.Path:
    """``<h5_dir>/<h5_stem>/default/``, or ``.../workspace_filtered__pad=<m>__bottompad=<m>__mountradius=<m>/`` for
    masks segmented on workspace-filtered RGB."""
    if workspace_tag is None:
        return h5_path.parent / h5_path.stem / "default"
    return (
        h5_path.parent
        / h5_path.stem
        / (
            f"workspace_filtered__pad={WORKSPACE_PADDING_M}__bottompad={WORKSPACE_BOTTOM_PADDING_M}"
            f"__mountradius={CAMERA_MOUNT_RADIUS_M}"
        )
    )


def workspace_filtered_rgb_dir(h5_path: pathlib.Path, camera: str, workspace_tag: str) -> pathlib.Path:
    """``<filtering>/blacked_out_rgb/<camera>__<workspace_tag>/``, holding ``rgb_depth.mp4`` and
    ``rgb_depth/<frame>.png``: original RGB | depth | the RGB that SAM 3 sees, with pixels outside the workspace or
    without depth blacked out."""
    return _filtering_dir(h5_path, workspace_tag) / "blacked_out_rgb" / f"{camera}__{workspace_tag}"


def robot_mask_cache_dir(
    h5_path: pathlib.Path,
    camera: str,
    workspace_tag: str | None,
    robot_description: str | None,
    kmax: int,
    score_threshold: float,
) -> pathlib.Path:
    """``<h5_dir>/<h5_stem>/<filtering>/<propagated_mask|per_frame_mask>/<params>/``, holding ``masks.mp4``,
    ``npy/<frame>.npy`` (cached bool masks) and ``rgb/<frame>.png`` (dimmed-mask overlays). ``<filtering>`` is
    ``default`` or, for masks segmented on workspace-filtered RGB, ``workspace_filtered__pad=<m>__bottompad=<m>__mountradius=<m>``.
    ``workspace_tag`` names the workspace and prior extrinsics (see ``workspace_mask_tag``).
    ``robot_description`` is the per-frame SAM 3 prompt, None for propagated masks."""
    method = "propagated_mask" if robot_description is None else "per_frame_mask"
    params = [camera]
    if workspace_tag is not None:
        params.append(workspace_tag)
    if robot_description is not None:
        assert "/" not in robot_description, f"robot_description must not contain '/', got {robot_description!r}"
        params.append(f"prompt={robot_description.replace(' ', '_')}")
    params += [f"kmax={int(kmax)}", f"score_threshold={score_threshold}"]
    return _filtering_dir(h5_path, workspace_tag) / method / "__".join(params)


def propagated_robot_mask_cache_path(
    h5_path: pathlib.Path,
    camera: str,
    frame_idx: int,
    kmax: int,
    score_threshold: float,
    workspace_tag: str | None,
) -> pathlib.Path:
    cache_dir = robot_mask_cache_dir(h5_path, camera, workspace_tag, None, kmax, score_threshold)
    return cache_dir / "npy" / f"{int(frame_idx):05d}.npy"


def propagated_robot_mask_video_path(
    h5_path: pathlib.Path,
    camera: str,
    kmax: int,
    score_threshold: float,
    workspace_tag: str | None,
) -> pathlib.Path:
    return robot_mask_cache_dir(h5_path, camera, workspace_tag, None, kmax, score_threshold) / "masks.mp4"


def per_frame_robot_mask_cache_path(
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    frame_idx: int,
    kmax: int,
    score_threshold: float,
    workspace_tag: str | None,
) -> pathlib.Path:
    cache_dir = robot_mask_cache_dir(h5_path, camera, workspace_tag, robot_description, kmax, score_threshold)
    return cache_dir / "npy" / f"{int(frame_idx):05d}.npy"


def per_frame_robot_mask_video_path(
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    kmax: int,
    score_threshold: float,
    workspace_tag: str | None,
) -> pathlib.Path:
    return robot_mask_cache_dir(h5_path, camera, workspace_tag, robot_description, kmax, score_threshold) / "masks.mp4"


def robot_mask_overlay_path(mask_path: pathlib.Path) -> pathlib.Path:
    """``rgb/<frame>.png`` overlay next to the ``npy/<frame>.npy`` mask cache file."""
    assert mask_path.parent.name == "npy", f"mask cache path must be in an npy/ dir, got {mask_path}"
    return mask_path.parent.parent / "rgb" / f"{mask_path.stem}.png"


def optimization_frames_dir(h5_path: pathlib.Path) -> pathlib.Path:
    """``<h5_dir>/<h5_stem>/optimization_intermediate_frames/<day_month_year__hour_min_sec>/``: a new directory for one
    extrinsics run's sampled-frame images."""
    stamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%d_%m_%Y__%H_%M_%S")  # local time
    frames_dir = h5_path.parent / h5_path.stem / "optimization_intermediate_frames" / stamp
    assert not frames_dir.exists(), f"{frames_dir} already exists"
    frames_dir.mkdir(parents=True)
    return frames_dir


def write_extrinsics_yaml(output_path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``output_path``. An existing file there is first copied to ``<stem>.backup.yaml``, replacing
    any older backup."""
    assert output_path.suffix == ".yaml", f"output_path must end in .yaml, got {output_path}"
    if output_path.exists():
        backup_path = output_path.with_suffix(".backup.yaml")
        shutil.copy2(output_path, backup_path)
        print(f"[info] Backed up previous extrinsics to {colored(str(backup_path), 'blue')}")
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def load_cached_bool_mask(path: pathlib.Path, hw: tuple[int, int], allow_empty: bool = False) -> np.ndarray:
    mask = np.load(path)
    assert isinstance(mask, np.ndarray), f"{path}: expected ndarray, got {type(mask)}"
    assert mask.shape == hw, f"{path}: mask shape {mask.shape} != {hw}"
    assert mask.dtype == bool, f"{path}: mask dtype must be bool, got {mask.dtype}"
    assert allow_empty or mask.any(), f"{path}: cached robot mask is empty"
    return mask


def n_consecutive_cached_masks(paths: list[pathlib.Path]) -> int:
    """Count how many leading cache files exist (0 if the first is missing)."""
    n = 0
    for path in paths:
        if not path.is_file():
            break
        n += 1
    return n


def save_cached_bool_mask(path: pathlib.Path, mask: np.ndarray, allow_empty: bool = False) -> None:
    assert mask.dtype == bool, f"{path}: mask dtype must be bool, got {mask.dtype}"
    assert allow_empty or mask.any(), f"{path}: cannot cache an empty robot mask"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)


def load_sam3() -> SAM3Predictor:
    t0 = time.perf_counter()
    predictor = SAM3Predictor()
    log_elapsed("SAM 3 loaded", t0)
    return predictor


def propagate_robot_masks(
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
        save_cached_bool_mask(cache_paths[0], seed_mask)
    mask_input = binary_mask_to_sam_mask_input(seed_mask)
    prev_mask = seed_mask
    for t in tqdm(range(1, n_frames), desc="Propagate robot masks"):
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        box = bbox_xyxy_from_mask(prev_mask)
        mask, _score, low_res = predictor.propagate_from_mask(image_bgr, mask_input, box)
        assert mask.shape == (height, width), f"frame {t}: mask {mask.shape} != {(height, width)}"
        masks[t] = mask
        if cache_paths is not None:
            save_cached_bool_mask(cache_paths[t], mask)
        mask_input = low_res
        prev_mask = mask
    return masks


def compute_propagated_robot_masks(
    rgb: np.ndarray,
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    sam_kmax: int,
    sam_score_threshold: float,
    cache_robot_masks: bool,
    visualize_robot_masks: bool,
    workspace_tag: str | None,
) -> np.ndarray:
    """Return ``(T, H, W)`` robot masks. Frame 0 is SAM 3; later frames are SAM 3-propagated. ``workspace_tag`` names
    the cache files of masks segmented on workspace-filtered ``rgb`` (None for unfiltered RGB)."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert n_frames >= 1, f"rgb must contain at least 1 frame, got {rgb.shape}"
    hw = (height, width)
    propagated_paths = [
        propagated_robot_mask_cache_path(h5_path, camera, t, sam_kmax, sam_score_threshold, workspace_tag)
        for t in range(n_frames)
    ]
    n_cached = n_consecutive_cached_masks(propagated_paths) if cache_robot_masks else 0
    if n_cached == n_frames:
        masks = np.stack([load_cached_bool_mask(path, hw) for path in propagated_paths], axis=0)
        print(f"[info] Loaded {n_frames} propagated robot masks from cache")
    else:
        masks = np.zeros((n_frames, height, width), dtype=bool)
        cache_paths = propagated_paths if cache_robot_masks else None
        if n_cached >= 1:
            for t in range(n_cached):
                masks[t] = load_cached_bool_mask(propagated_paths[t], hw)
            print(f"[info] Resuming robot-mask propagation from frame {n_cached} ({n_cached}/{n_frames} cached)")
            predictor = load_sam3()
            rest = propagate_robot_masks(
                rgb[n_cached - 1 :],
                predictor,
                masks[n_cached - 1],
                cache_paths=None if cache_paths is None else cache_paths[n_cached - 1 :],
            )
            assert rest.shape[0] == n_frames - n_cached + 1, f"rest frames {rest.shape[0]} != {n_frames - n_cached + 1}"
            masks[n_cached:] = rest[1:]
        else:
            predictor = load_sam3()
            image_bgr0 = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR)
            print(f"[info] Segmenting '{robot_description}' on frame 0 ...")
            t0 = time.perf_counter()
            ranked_masks, scores, phrases = ImageUtils.get_sam_masks_ranked(predictor, image_bgr0, robot_description)
            log_elapsed(f"Segmented '{robot_description}' on frame 0 ({ranked_masks.shape[0]} masks)", t0)
            for rank in range(min(sam_kmax, ranked_masks.shape[0])):
                print(
                    f"[info]   rank={rank} score={scores[rank]:.4f} phrase={phrases[rank]!r} "
                    f"pixels={int(ranked_masks[rank].sum())}"
                )
            seed_mask, kept = union_top_sam_masks(ranked_masks, scores, sam_kmax, sam_score_threshold)
            print(
                f"[info] Unioned {kept.size}/{min(sam_kmax, ranked_masks.shape[0])} masks with score > "
                f"{sam_score_threshold} (kept ranks={kept.tolist()}, scores={scores[kept].tolist()})"
            )
            if visualize_robot_masks:
                dump_top_sam_masks(
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

            if n_frames == 1:
                masks = seed_mask[None, ...]
                if cache_paths is not None:
                    save_cached_bool_mask(cache_paths[0], seed_mask)
            else:
                print(f"[info] Propagating frame-0 robot mask through {n_frames} frames ...")
                t0 = time.perf_counter()
                masks = propagate_robot_masks(rgb, predictor, seed_mask, cache_paths=cache_paths)
                log_elapsed(f"Propagated robot mask through {n_frames} frames", t0)
        assert masks.shape == (n_frames, height, width), f"masks {masks.shape} != {(n_frames, height, width)}"
        if cache_robot_masks:
            print(
                f"[info] Saved {n_frames} propagated robot masks to {colored(str(propagated_paths[0].parent), 'blue')}"
            )
    save_robot_mask_demo_overlays(
        rgb,
        masks,
        propagated_paths,
        propagated_robot_mask_video_path(h5_path, camera, sam_kmax, sam_score_threshold, workspace_tag),
    )
    return masks


def segment_robot_masks_per_frame(
    rgb: np.ndarray,
    frame_indices: np.ndarray,
    robot_description: str,
    sam_kmax: int,
    sam_score_threshold: float,
    cache_paths: list[pathlib.Path] | None,
) -> np.ndarray:
    """Return ``(len(frame_indices), H, W)`` robot masks for ``rgb[frame_indices]``, each from SAM 3 on its own frame
    (union of the top ``sam_kmax`` masks). A frame where none of the top ``sam_kmax`` SAM 3 masks scores above
    ``sam_score_threshold`` (including no masks at all) gets an empty mask.

    If ``cache_paths`` (one per entry of ``frame_indices``) is given, frames with an existing cache file are loaded and
    only the rest are segmented. Empty masks are cached too.
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
            masks[i] = load_cached_bool_mask(cache_paths[i], (height, width), allow_empty=True)
        print(f"[info] Loaded {n_sel - len(todo)}/{n_sel} per-frame robot masks from cache")
    if len(todo) == 0:
        return masks
    predictor = load_sam3()
    for i in todo:
        t = int(frame_indices[i])
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        ranked_masks, scores, _ = ImageUtils.get_sam_masks_ranked(
            predictor, image_bgr, robot_description, allow_no_masks=True
        )
        if not np.any(scores[:sam_kmax] > sam_score_threshold):
            # The robot is not found in every frame (e.g. arm out of view). main() ignores frames with an empty mask
            if cache_paths is not None:
                save_cached_bool_mask(cache_paths[i], masks[i], allow_empty=True)
            print(
                f"[warn] frame {t}/{n_frames}: no SAM 3 mask with score > {sam_score_threshold} "
                f"(scores={np.round(scores[:sam_kmax], 2).tolist()}), frame will be ignored"
            )
            continue
        masks[i], kept = union_top_sam_masks(ranked_masks, scores, sam_kmax, sam_score_threshold)
        if cache_paths is not None:
            save_cached_bool_mask(cache_paths[i], masks[i])
        print(
            f"[info] frame {t}/{n_frames}: {int(masks[i].sum())} px from {kept.size}/{ranked_masks.shape[0]} masks "
            f"(scores={np.round(scores[kept], 2).tolist()})"
        )
    return masks


def save_robot_mask_demo_overlays(
    rgb: np.ndarray,
    masks: np.ndarray,
    mask_paths: list[pathlib.Path],
    video_path: pathlib.Path,
) -> None:
    """Write a dimmed-mask demo PNG for every frame's mask path (see ``robot_mask_overlay_path``) and an mp4 of those
    overlays. Frames with an empty mask are fully dimmed."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert masks.shape == (n_frames, height, width), f"masks {masks.shape} != {(n_frames, height, width)}"
    assert len(mask_paths) == n_frames, f"mask_paths length {len(mask_paths)} != n_frames {n_frames}"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = H264VideoWriter(video_path, MASK_VIDEO_FPS, width, height)
    for t in range(n_frames):
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        overlay_path = robot_mask_overlay_path(mask_paths[t])
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        demo = ImageUtils.demo_overlay(image_bgr, masks[t]) if masks[t].any() else image_bgr // 4
        assert cv2.imwrite(str(overlay_path), demo), f"Failed to write {overlay_path}"
        writer.write(demo)
    writer.close()
    print(f"[info] Saved {n_frames} robot-mask demo overlays and video to {colored(str(video_path), 'blue')}")


def depth_colormap(depth_m: np.ndarray) -> np.ndarray:
    """BGR turbo colormap of ``depth_m`` (meters), scaled to its 99th percentile. Pixels without depth are black."""
    assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
    finite = np.isfinite(depth_m) & (depth_m > 0)
    assert finite.any(), "depth has no finite positive values"
    depth_max = float(np.percentile(depth_m[finite], 99))
    depth_u8 = np.zeros(depth_m.shape, dtype=np.uint8)
    depth_u8[finite] = (255.0 * np.clip(depth_m[finite] / depth_max, 0.0, 1.0)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~finite] = 0
    return depth_color


def union_top_sam_masks(
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


def erode_mask(mask: np.ndarray, kernel_px: int) -> np.ndarray:
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


def dump_top_sam_masks(
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


def frame_depth_m(
    depth_raw: np.ndarray,
    color_intrinsics: CameraIntrinsics,
    depth_reprojection: tuple[CameraIntrinsics, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """One frame's uint16 mm depth as float32 meters in the color frame. ``depth_reprojection`` is
    ``(depth_intrinsics, R_dc, t_dc)`` for native depth, or None when depth already uses the color intrinsics."""
    depth_m = depth_mm_to_meters(depth_raw)
    if depth_reprojection is None:
        return depth_m
    depth_intrinsics, R_dc, t_dc = depth_reprojection
    return reproject_depth_to_color_frame(depth_m, depth_intrinsics, color_intrinsics, R_dc, t_dc)


def mask_valid_depth_fraction(mask: np.ndarray, depth_m: np.ndarray, max_depth_m: float) -> float:
    """Fraction of ``mask`` pixels whose depth is finite and in ``[MIN_DEPTH_M, max_depth_m]``."""
    assert mask.dtype == bool and mask.any(), "mask must be a non-empty bool array"
    assert mask.shape == depth_m.shape, f"mask {mask.shape} != depth {depth_m.shape}"
    z = depth_m[mask]
    return float(np.mean(np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= max_depth_m)))


def index_ranges_text(indices: np.ndarray) -> str:
    """Compact text for sorted ints, e.g. [3, 4, 5, 9] -> '3-5, 9'."""
    if indices.size == 0:
        return "none"
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in zip(starts.tolist(), ends.tolist()))


def load_demo(h5_path: pathlib.Path, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rgb, depth, and qpos, aligned by frame index (same N)."""
    with h5py.File(h5_path, "r") as f:
        validate_merged_camera_group(f, h5_path, camera)
        assert (
            "obs/qpos" in f
        ), f"{h5_path}: missing 'obs/qpos' (re-merge the demo with data/demonstrations/grail_xarm_july_2/merge_camera_streams.py)"
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


def expand_demo_qpos_to_joint_names(qpos: np.ndarray, robot, target_joint_names: list[str]) -> np.ndarray:
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


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
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


def seed_eyes_on_sphere(n_azimuth: int, n_polar: int, n_radii: int) -> np.ndarray:
    """Fixed upper-hemisphere grid with camera-position z > 0 about the robot base."""
    assert n_azimuth >= 1, f"n_azimuth must be >= 1, got {n_azimuth}"
    assert n_polar >= 1, f"n_polar must be >= 1, got {n_polar}"
    assert n_radii >= 1, f"n_radii must be >= 1, got {n_radii}"
    radii = np.linspace(SEED_RADIUS_MIN_M, SEED_RADIUS_MAX_M, n_radii, dtype=np.float64)
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
                eyes[k] = SEED_ORIGIN + float(radius_m) * np.array(
                    [st * np.cos(phi), st * np.sin(phi), ct], dtype=np.float64
                )
                k += 1
    assert k == eyes.shape[0]
    assert np.all(eyes[:, 2] > 0.0), f"All seed camera positions must have z > 0, got min z={eyes[:, 2].min()}"
    look = SEED_LOOKAT_TARGET[None, :] - eyes
    look_n = np.linalg.norm(look, axis=1)
    assert np.all(
        look_n >= SEED_MIN_LOOK_DIST_M
    ), f"Seed sphere grid has look-at distance < {SEED_MIN_LOOK_DIST_M} m: min={float(look_n.min())}"
    look_u = look / look_n[:, None]
    up_n = SEED_UP / np.linalg.norm(SEED_UP)
    parallel = np.abs(look_u @ up_n) > (1.0 - 1e-6)
    assert not np.any(
        parallel
    ), f"Seed sphere grid has look-at parallel to up at indices {np.flatnonzero(parallel).tolist()}"
    return eyes


def look_at_with_roll(eye: np.ndarray, target: np.ndarray, roll_rad: float, up: np.ndarray) -> np.ndarray:
    """Look-at pose, then roll about the camera optical axis (+Z, the look-at direction)."""
    T = look_at_opencv(eye, target, up)
    Rz = axangle2mat(np.array([0.0, 0.0, 1.0], dtype=np.float64), float(roll_rad))
    T_roll = T.copy()
    T_roll[:3, :3] = T[:3, :3] @ Rz
    return T_roll


def pose_translation_rotation_text(T: np.ndarray) -> str:
    """Markdown lines for camera translation and rotation in the robot-base frame."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    t = T[:3, 3]
    q = mat2quat(T[:3, :3])
    rotvec_deg = np.rad2deg(T_to_vec(T)[3:])
    return (
        f"**camera t (robot base):** `[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]`  \n"
        f"**camera R (wxyz):** `[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]`  \n"
        f"**camera rotvec (deg):** `[{rotvec_deg[0]:.2f}, {rotvec_deg[1]:.2f}, {rotvec_deg[2]:.2f}]`"
    )


def step_pose_world(
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


def generate_seed_poses(
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
    eyes = seed_eyes_on_sphere(n_azimuth, n_polar, n_radii)
    rolls = np.linspace(0.0, 2.0 * np.pi, n_rolls, endpoint=False)
    poses = [look_at_with_roll(eye, SEED_LOOKAT_TARGET, float(roll), SEED_UP) for eye in eyes for roll in rolls]
    n_expected = len(eyes) * n_rolls
    assert len(poses) == n_expected, f"expected {n_expected} seeds, got {len(poses)}"
    assert all(T[2, 3] > 0.0 for T in poses), "All seed camera poses must have translation z > 0"
    return poses


def pose_from_xyz_wxyz(xyz_wxyz: np.ndarray | tuple[float, ...]) -> np.ndarray:
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


def T_to_vec(T: np.ndarray) -> np.ndarray:
    """SE(3) 4x4 → 6D [tx, ty, tz, rx, ry, rz] with rotation-vector orientation."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    axis, angle = mat2axangle(T[:3, :3])
    rotvec = np.asarray(axis, dtype=np.float64) * float(angle)
    return np.concatenate([T[:3, 3].astype(np.float64), rotvec], axis=0)


def vec_to_T(x: np.ndarray) -> np.ndarray:
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


def T_to_sapien_pose(T_world_cam: np.ndarray) -> sapien.Pose:
    """RealSense-optical T_world_cam → Sapien / ManiSkill camera pose."""
    t, R_ms = camera_extrinsic_to_maniskill_pose(T_world_cam)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_ms
    mat[:3, 3] = t
    return sapien.Pose(mat[:3, 3], mat2quat(mat[:3, :3]))


def urdf_for_sapien(urdf_path: pathlib.Path) -> pathlib.Path:
    """Sapien cannot resolve `file://` mesh hrefs that xacrodoc emits. Strip the prefix."""
    text = urdf_path.read_text()
    rewritten = text.replace('filename="file://', 'filename="')
    if rewritten == text:
        return urdf_path
    out = urdf_path.with_name(urdf_path.stem + "__sapien.urdf")
    out.write_text(rewritten)
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE(3) `T` to `(N, 3)` points. Empty clouds stay empty."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    assert pts.ndim == 2 and pts.shape[1] == 3, f"pts must be (N, 3), got {pts.shape}"
    if pts.shape[0] == 0:
        return pts
    return (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def world_points_to_camera(T_world_cam: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """Map robot-base points into the OpenCV camera frame: `(p - t) @ R`."""
    assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
    assert pts_world.ndim == 2 and pts_world.shape[1] == 3, f"pts_world must be (N, 3), got {pts_world.shape}"
    assert pts_world.shape[0] >= 1, "pts_world must contain at least 1 point"
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return ((pts_world.astype(np.float64) - t) @ R).astype(np.float32)


def remove_radius_outliers(points: np.ndarray, nb_points: int, radius_m: float) -> np.ndarray:
    """Drop points with fewer than `nb_points` neighbors inside `radius_m` (Open3D radius outlier). The result can be
    empty."""
    import open3d as o3d

    assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    assert points.shape[0] >= 1, "points must contain at least 1 point"
    assert nb_points >= 1, f"nb_points must be >= 1, got {nb_points}"
    assert radius_m > 0.0, f"radius_m must be > 0, got {radius_m}"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    filtered, _ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius_m)
    out = np.asarray(filtered.points).reshape(-1, 3)
    return out.astype(np.float64, copy=False)


def subsample_pcd(
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


def compute_robot_masks(
    rgb_all: np.ndarray,
    h5_path: pathlib.Path,
    camera: str,
    robot_description: str,
    sam_kmax: int,
    sam_score_threshold: float,
    per_frame_robot_masks: bool,
    cache_robot_masks: bool,
    visualize_robot_masks: bool,
    workspace_masks: np.ndarray | None,
    workspace_tag: str | None,
) -> np.ndarray:
    """``(T, H, W)`` robot masks for every frame. With ``per_frame_robot_masks``, SAM 3 runs on every frame (empty mask
    where it finds no robot) and a demo overlay video is written. Otherwise frame 0's SAM 3 mask is propagated.

    With ``workspace_masks`` (``(T, H, W)`` bool from ``compute_workspace_masks``), SAM 3 sees the RGB with pixels
    outside the workspace blacked out, its masks are cut to the workspace, and the cache names carry ``workspace_tag``.
    """
    n_frames, height, width, _ = rgb_all.shape
    assert (workspace_masks is None) == (workspace_tag is None), "workspace_masks and workspace_tag go together"
    rgb_sam = rgb_all
    if workspace_masks is not None:
        assert workspace_masks.shape == (
            n_frames,
            height,
            width,
        ), f"workspace_masks {workspace_masks.shape} != {(n_frames, height, width)}"
        rgb_sam = np.where(workspace_masks[..., None], rgb_all, 0).astype(np.uint8)
    if per_frame_robot_masks:
        # Every frame is segmented (and cached) so that unusable frames can be excluded before sampling
        per_frame_paths = [
            per_frame_robot_mask_cache_path(
                h5_path, camera, robot_description, t, sam_kmax, sam_score_threshold, workspace_tag
            )
            for t in range(n_frames)
        ]
        masks_all = segment_robot_masks_per_frame(
            rgb_sam,
            np.arange(n_frames),
            robot_description,
            sam_kmax,
            sam_score_threshold,
            per_frame_paths if cache_robot_masks else None,
        )
        if workspace_masks is not None:
            masks_all &= workspace_masks
        save_robot_mask_demo_overlays(
            rgb_sam,
            masks_all,
            per_frame_paths,
            per_frame_robot_mask_video_path(
                h5_path, camera, robot_description, sam_kmax, sam_score_threshold, workspace_tag
            ),
        )
    else:
        masks_all = compute_propagated_robot_masks(
            rgb_sam,
            h5_path,
            camera,
            robot_description,
            sam_kmax,
            sam_score_threshold,
            cache_robot_masks,
            visualize_robot_masks,
            workspace_tag,
        )
        if workspace_masks is not None:
            masks_all &= workspace_masks
    assert masks_all.shape == (n_frames, height, width), f"masks_all {masks_all.shape} != {(n_frames, height, width)}"
    return masks_all


def load_extrinsics_yaml(path: pathlib.Path, camera: str) -> np.ndarray:
    """4x4 robot-base T camera from an extrinsics YAML written by the estimate_camera_extrinsics scripts."""
    assert path.is_file(), f"Extrinsics YAML not found: {path}"
    data = yaml.safe_load(path.read_text())
    assert data["camera"] == camera, f"{path} is for camera {data['camera']!r}, not {camera!r}"
    assert data["parent_frame"] == "robot_base", f"{path}: parent_frame must be robot_base, got {data['parent_frame']}"
    T = np.asarray(data["matrix"], dtype=np.float64)
    assert T.shape == (4, 4), f"{path}: matrix must be 4x4, got {T.shape}"
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]), f"{path}: matrix last row must be [0, 0, 0, 1], got {T[3]}"
    return T


def workspace_mask_tag(workspace_id: str, prior_extrinsics_path: pathlib.Path) -> str:
    """Cache-name tag for masks segmented on RGB filtered with ``padded_workspace_bounds(workspace_id)`` and that prior
    extrinsics YAML. Hashes of the padded bounds and of the YAML's content are included so that editing either does not
    reuse stale masks. The padding is also in the parent folder name (see ``robot_mask_cache_dir``)."""
    bounds_digest = hashlib.sha1(repr(padded_workspace_bounds(workspace_id)).encode()).hexdigest()[:8]
    prior_digest = hashlib.sha1(prior_extrinsics_path.read_bytes()).hexdigest()[:8]
    return f"workspace={workspace_id}-{bounds_digest}__prior={prior_extrinsics_path.stem}-{prior_digest}"


def padded_workspace_bounds(workspace_id: str) -> WorkspaceBounds:
    """The ``workspace_id`` bounds padded by ``WORKSPACE_PADDING_M`` on every side but the bottom, which is padded by
    ``WORKSPACE_BOTTOM_PADDING_M``."""
    bounds = get_workspace_bounds(workspace_id)
    return dataclasses.replace(bounds.padded(WORKSPACE_PADDING_M), min_z_m=bounds.min_z_m - WORKSPACE_BOTTOM_PADDING_M)


def in_workspace_pixels(
    depth_m: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, bounds: WorkspaceBounds
) -> np.ndarray:
    """``(H, W)`` bool: False where a pixel's depth puts it outside ``bounds`` (robot-base frame) seen from
    ``T_world_cam``, and where the pixel has no valid depth."""
    assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
    assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
    assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
    height, width = depth_m.shape
    vs, us = np.mgrid[0:height, 0:width]
    valid = np.isfinite(depth_m) & (depth_m >= MIN_DEPTH_M)
    z = depth_m[valid].astype(np.float64)
    pts_cam = np.stack([(us[valid] - K[0, 2]) * z / K[0, 0], (vs[valid] - K[1, 2]) * z / K[1, 1], z], axis=-1)
    keep = np.zeros((height, width), dtype=bool)
    keep[valid] = bounds.contains(pts_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3])
    return keep


def compute_workspace_masks(
    workspace_id: str | None,
    prior_extrinsics_path: pathlib.Path | None,
    h5_path: pathlib.Path,
    camera: str,
    rgb_all: np.ndarray,
    depth_raw_all: np.ndarray,
    color_intrinsics: CameraIntrinsics,
    depth_reprojection: tuple[CameraIntrinsics, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray | None, str | None]:
    """``(workspace_masks, workspace_tag)`` for ``compute_robot_masks``: ``in_workspace_pixels`` of every frame against
    ``padded_workspace_bounds(workspace_id)``, seen from the prior extrinsics. ``(None, None)`` when neither is given;
    giving only one is an error. Each frame's original RGB | depth | blacked-out RGB is written to
    ``workspace_filtered_rgb_dir``."""
    assert (workspace_id is None) == (
        prior_extrinsics_path is None
    ), "--workspace-id and --workspace-prior-extrinsics must be given together"
    if workspace_id is None:
        return None, None
    bounds = padded_workspace_bounds(workspace_id)
    T_world_cam = load_extrinsics_yaml(prior_extrinsics_path, camera)
    K = color_intrinsics.intrinsic_matrix.astype(np.float64)
    n_frames, height, width, _ = rgb_all.shape
    assert depth_raw_all.shape[0] == n_frames, f"depth frames {depth_raw_all.shape[0]} != rgb frames {n_frames}"
    workspace_tag = workspace_mask_tag(workspace_id, prior_extrinsics_path)
    out_dir = workspace_filtered_rgb_dir(h5_path, camera, workspace_tag)
    (out_dir / "rgb_depth").mkdir(parents=True, exist_ok=True)
    writer = H264VideoWriter(out_dir / "rgb_depth.mp4", MASK_VIDEO_FPS, 3 * width, height)
    t0 = time.perf_counter()
    workspace_masks = np.zeros((n_frames, height, width), dtype=bool)
    for t in tqdm(range(n_frames), desc="Workspace filter"):
        depth_m = frame_depth_m(depth_raw_all[t], color_intrinsics, depth_reprojection)
        assert depth_m.shape == (height, width), f"frame {t}: depth {depth_m.shape} != rgb {(height, width)}"
        workspace_masks[t] = in_workspace_pixels(depth_m, K, T_world_cam, bounds)
        image_bgr = cv2.cvtColor(rgb_all[t], cv2.COLOR_RGB2BGR)
        panel = np.concatenate(
            [image_bgr, depth_colormap(depth_m), np.where(workspace_masks[t][..., None], image_bgr, 0)], axis=1
        ).astype(np.uint8)
        path = out_dir / "rgb_depth" / f"{t:05d}.png"
        assert cv2.imwrite(str(path), panel), f"Failed to write {path}"
        writer.write(panel)
    writer.close()
    print(f"[info] Saved original RGB | depth | blacked-out RGB frames and video to {colored(str(out_dir), 'blue')}")
    log_elapsed(
        f"Workspace filter '{workspace_id}' (+{WORKSPACE_PADDING_M} m, bottom +{WORKSPACE_BOTTOM_PADDING_M} m) from {prior_extrinsics_path}: "
        f"{1.0 - workspace_masks.mean():.0%} of pixels outside",
        t0,
    )
    return workspace_masks, workspace_tag


def frames_with_robot_mask(masks_all: np.ndarray, per_frame_robot_masks: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(has_mask, no_mask)`` frame indices.

    Per-frame SAM 3 leaves the mask empty where it finds no robot, and can return a tiny mask (<=
    ``MIN_ROBOT_MASK_PIXELS`` pixels) on a bad segmentation; both go in ``no_mask`` and callers skip them like
    --ignore-timesteps. Propagated masks must be non-empty and larger than ``MIN_ROBOT_MASK_PIXELS``, since a tiny
    propagated mask means tracking was lost for every later frame too.
    """
    assert masks_all.ndim == 3 and masks_all.dtype == bool, f"masks_all must be (T, H, W) bool, got {masks_all.shape}"
    n_frames = masks_all.shape[0]
    mask_pixels = masks_all.reshape(n_frames, -1).sum(axis=1)
    empty = np.flatnonzero(mask_pixels == 0)
    tiny = np.flatnonzero((mask_pixels > 0) & (mask_pixels <= MIN_ROBOT_MASK_PIXELS))
    assert per_frame_robot_masks or empty.size == 0, f"Propagated robot mask is empty at frames {empty}"
    assert per_frame_robot_masks or tiny.size == 0, (
        f"Propagated robot mask has <= {MIN_ROBOT_MASK_PIXELS} pixels at {tiny.size}/{n_frames} frames (lost "
        f"tracking). First frames: {[(int(t), int(mask_pixels[t])) for t in tiny[:10]]}"
    )
    print(f"[info] Ignoring {empty.size} frames where SAM 3 found no robot: {index_ranges_text(empty)}")
    print(
        f"[info] Ignoring {tiny.size} frames where the SAM 3 robot mask has <= {MIN_ROBOT_MASK_PIXELS} pixels: "
        f"{index_ranges_text(tiny)}"
    )
    no_mask = np.union1d(empty, tiny)
    has_mask = np.flatnonzero(mask_pixels > MIN_ROBOT_MASK_PIXELS)
    return has_mask, no_mask


def depth_reprojection_for(
    camera: str,
    camera_model_id: str,
    depth_intrinsics_source: str,
    depth_hw: tuple[int, int],
) -> tuple[CameraIntrinsics, np.ndarray, np.ndarray] | None:
    """``frame_depth_m``'s ``depth_reprojection`` argument: None when the h5 depth is already color-aligned
    (``depth_intrinsics_source == "rgb"``), else the native depth intrinsics and depth-to-color extrinsics."""
    assert depth_intrinsics_source in (
        "rgb",
        "depth",
    ), f"depth_intrinsics_source must be 'rgb' or 'depth', got {depth_intrinsics_source!r}"
    print(f"[info] Depth channel uses {depth_intrinsics_source} intrinsics")
    if depth_intrinsics_source == "rgb":
        return None
    depth_intrinsics = get_depth_intrinsics(camera_model_id)
    assert depth_hw == (depth_intrinsics.height, depth_intrinsics.width), (
        f"{camera} depth is {depth_hw[0]}x{depth_hw[1]} but {camera_model_id} depth intrinsics are "
        f"{depth_intrinsics.height}x{depth_intrinsics.width}"
    )
    R_dc, t_dc = get_depth_to_color_extrinsics(camera_model_id)
    return depth_intrinsics, R_dc, t_dc


def build_real_pcd(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    max_point_distance_m: float,
    radius_outlier_nb_points: int,
    radius_outlier_radius_m: float,
    n_samples: int,
    n_random_downsample_initial: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Measured robot cloud in the OpenCV camera frame from an (already eroded) robot ``mask``.

    Unprojects the masked depth, keeps points within ``max_point_distance_m`` of the camera center, drops radius
    outliers, and subsamples to ``n_samples``. Returns the cloud (possibly empty) and the point counts after
    unprojection, the distance filter, and the radius filter.
    """
    if mask_valid_depth_fraction(mask, depth_m, max_point_distance_m) == 0.0:
        return np.zeros((0, 3), dtype=np.float32), (0, 0, 0)
    pts = masked_depth_to_points(depth_m, mask, K, max_depth_m=max_point_distance_m)
    n_depth = int(pts.shape[0])
    pts = pts[np.linalg.norm(pts, axis=1) <= max_point_distance_m]
    n_unproj = int(pts.shape[0])
    if n_unproj > 0:
        pts = remove_radius_outliers(pts, radius_outlier_nb_points, radius_outlier_radius_m)
    n_inliers = int(pts.shape[0])
    pts = subsample_pcd(pts, n_samples, n_random_downsample_initial, device, rng)
    return pts, (n_depth, n_unproj, n_inliers)


class SapienRobot:
    """Sapien scene with the Jrl2 URDF fixed at the origin. Subclasses add the cameras they render from."""

    def __init__(self, urdf_path: pathlib.Path) -> None:
        sapien_urdf = urdf_for_sapien(urdf_path)
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
    sims: list[Any] | None = None
    best_cost: float = np.inf
    t0: float = dataclasses.field(default_factory=time.perf_counter)


class ExtrinsicsVisualizerBase:
    """Viser view in the robot-base frame: robot model, measured robot clouds, camera frustum at the shown pose,
    CMA-ES population axes, and seed-search / CMA-ES cost plots.

    Subclasses draw their method's simulated observation for a timestep in ``_draw_timestep`` and may add GUI elements
    in ``_add_method_gui``. They must set the attributes those hooks use before calling ``super().__init__``.

    ``renderer`` only poses the displayed robot meshes. Do not share it with ``cost_fn``: the timestep dropdown sets its
    qpos from the Viser thread.
    """

    def __init__(
        self,
        K: np.ndarray,
        height: int,
        width: int,
        rgb_frames: np.ndarray,
        pcd_reals: list[np.ndarray],
        cma_maxiter: int,
        fps_idx: np.ndarray,
        renderer: SapienRobot,
        joint_angles: np.ndarray,
        cost_fn: CostFn,
        cost_unit: str,
    ) -> None:
        import viser
        from viser import uplot

        fps_idx = np.asarray(fps_idx, dtype=np.int64).reshape(-1)
        n_timesteps = fps_idx.shape[0]
        assert n_timesteps >= 1, "fps_idx must be non-empty"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0, f"height and width must be > 0, got {height}x{width}"
        assert rgb_frames.shape == (
            n_timesteps,
            height,
            width,
            3,
        ), f"rgb_frames must be ({n_timesteps}, {height}, {width}, 3), got {rgb_frames.shape}"
        assert len(pcd_reals) == n_timesteps, f"pcd_reals {len(pcd_reals)} != n timesteps {n_timesteps}"
        for i, pts in enumerate(pcd_reals):
            assert pts.ndim == 2 and pts.shape[1] == 3, f"pcd_reals[{i}] must be (N, 3), got {pts.shape}"
        assert cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {cma_maxiter}"
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        assert joint_angles.shape == (
            n_timesteps,
            renderer.dof,
        ), f"joint_angles {joint_angles.shape} != ({n_timesteps}, {renderer.dof})"
        fy = float(K[1, 1])
        assert fy > 0, f"K[1,1] (fy) must be > 0, got {fy}"

        self._n_timesteps = n_timesteps
        self._rgb_frames = rgb_frames
        self._pcd_reals = pcd_reals
        self._fps_idx = fps_idx
        self._renderer = renderer
        self._joint_angles = joint_angles
        self._cost_fn = cost_fn
        self._cost_unit = cost_unit
        self._T = look_at_with_roll(np.array([0.80, 0.00, 0.50], dtype=np.float64), SEED_LOOKAT_TARGET, np.pi, SEED_UP)
        self._T_init = self._T.copy()
        self._maxiter = cma_maxiter
        self._xs = np.arange(1, cma_maxiter + 1, dtype=np.float64)
        self._ys_best = np.full(cma_maxiter, np.nan, dtype=np.float64)
        self._ys_pop = np.full(cma_maxiter, np.nan, dtype=np.float64)
        self._Ts: list[np.ndarray | None] = [None] * cma_maxiter
        self._sims_hist: list[list[Any] | None] = [None] * cma_maxiter
        self._pop_Ts_hist: list[list[np.ndarray] | None] = [None] * cma_maxiter
        self._best_costs: list[float | None] = [None] * cma_maxiter
        self._n_recorded = 0
        self._live_sims: list[Any] | None = None
        self._live_pop_Ts: list[np.ndarray] | None = None
        self._pop_axes: list = []
        self._manual_seed: _ManualSeedSelection | None = None
        self._manual_seed_status: Any | None = None
        self._timestep = 0
        self._iteration = 0

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
            axes_length=BEST_AXES_LENGTH,
            axes_radius=BEST_AXES_RADIUS,
            origin_radius=BEST_ORIGIN_RADIUS,
            wxyz=wxyz0,
            position=pos0,
        )
        self._load_robot_meshes()
        self._cost_md = self._server.gui.add_markdown("best cost: —")
        self._pose_md = self._server.gui.add_markdown("camera pose: —")
        self._add_method_gui()

        # Sampled frames are sparse and in FPS order, so pick by demo frame index (sorted) rather than sample position
        sample_by_frame = {str(int(frame)): i for i, frame in enumerate(fps_idx)}
        assert len(sample_by_frame) == n_timesteps, f"fps_idx has duplicate frames: {fps_idx.tolist()}"
        timestep_dropdown = self._server.gui.add_dropdown(
            "timestep index", sorted(sample_by_frame, key=int), initial_value=str(int(fps_idx[0]))
        )

        @timestep_dropdown.on_update
        def _on_timestep(_) -> None:
            self._timestep = sample_by_frame[timestep_dropdown.value]
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

        cost_axis_label = f"cost ({cost_unit})" if cost_unit else "cost"
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
                uplot.Axis(label=cost_axis_label),
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
                uplot.Axis(label=cost_axis_label),
            ),
            aspect=1.8,
            height=220,
        )
        self._apply()
        print(f"[info] Viser extrinsics view at http://{self._server.get_host()}:{self._server.get_port()}")

    def _add_method_gui(self) -> None:
        """Add method-specific GUI elements above the timestep controls. Called once from ``__init__``."""

    def _draw_timestep(self, t: int, sim: Any | None) -> None:
        """Draw sampled timestep ``t`` at pose ``self._T`` (including the frustum image). ``sim`` is that timestep's
        simulated observation, or None before any pose has been evaluated."""
        raise NotImplementedError

    def _cost_text(self, cost: float) -> str:
        return f"`{cost:.6f}` {self._cost_unit}".rstrip()

    def select_seed_from_gui(
        self,
        translation_step_m: float,
        rotation_step_deg: float,
    ) -> tuple[np.ndarray, float, list[Any]]:
        """Block until the user adjusts and selects a CMA-ES seed in the Viser GUI.

        GUI callbacks only move the pose; this (calling) thread evaluates it, so the cost function is only used from
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
        assert state.cost is not None and state.sims is not None and state.T_evaluated is not None
        self._manual_seed_status.content = f"**Selected seed cost:** {self._cost_text(state.cost)}"
        return state.T_evaluated.copy(), float(state.cost), [sim.copy() for sim in state.sims]

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
            state.T_world_cam = step_pose_world(
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
        cost, sims = self._cost_fn(T_world_cam)
        state.T_evaluated = T_world_cam.copy()
        state.cost = float(cost)
        state.sims = sims
        state.best_cost = min(state.best_cost, state.cost)
        assert np.isfinite(state.best_cost), f"best_cost must be finite, got {state.best_cost}"
        self.record_seed_best(time.perf_counter() - state.t0, state.best_cost)
        self.set_best(sims, state.cost, T_world_cam)
        self._manual_seed_status.content = (
            f"**Current cost:** {self._cost_text(state.cost)}  \n"
            f"**Best cost:** {self._cost_text(state.best_cost)}  \n"
            f"{pose_translation_rotation_text(T_world_cam)}"
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

    def set_best(self, sims: list[Any], cost: float, T_world_cam: np.ndarray) -> None:
        """Live-update the scene when a new global best is found, if viewing the latest iteration."""
        assert len(sims) == self._n_timesteps, f"sims {len(sims)} != n timesteps {self._n_timesteps}"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        if self._n_recorded > 0 and self._iteration != self._n_recorded:
            return
        self._live_sims = [sim.copy() for sim in sims]
        self._apply(T=T_world_cam.copy(), sims=self._live_sims, cost=float(cost))

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
        """Append one seed-search sample: wall time (s) vs best cost so far."""
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
        sims: list[Any],
        pop_Ts: list[np.ndarray],
    ) -> None:
        assert 1 <= generation <= self._maxiter, f"generation {generation} not in [1, {self._maxiter}]"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        assert len(sims) == self._n_timesteps, f"sims {len(sims)} != n timesteps {self._n_timesteps}"
        assert len(pop_Ts) >= 1, "population must contain at least one pose"
        for i, T_pop in enumerate(pop_Ts):
            assert T_pop.shape == (4, 4), f"pop_Ts[{i}] must be 4x4, got {T_pop.shape}"
        i = generation - 1
        self._ys_best[i] = float(best_cost)
        self._ys_pop[i] = float(pop_min)
        self._Ts[i] = T_world_cam.copy()
        self._sims_hist[i] = [sim.copy() for sim in sims]
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
        sims: list[Any] | None = None,
        cost: float | None = None,
    ) -> None:
        t = self._timestep
        snap_i = self._snapshot_index()
        pop_Ts: list[np.ndarray] | None = None
        if T is None and self._iteration == 0 and self._manual_seed is not None and self._manual_seed.cost is not None:
            # The GUI seed is the CMA-ES start, so iteration 0 shows it (and keeps it when the timestep changes)
            T = self._manual_seed.T_evaluated.copy()
            sims = self._manual_seed.sims
            cost = self._manual_seed.cost
            pop_Ts = []
        elif T is None and self._iteration == 0:
            T = self._T_init
            sims = None
            cost = None
            pop_Ts = []
        elif T is None and snap_i is not None:
            T = self._Ts[snap_i]
            sims = self._sims_hist[snap_i]
            cost = self._best_costs[snap_i]
            pop_Ts = self._pop_Ts_hist[snap_i]
            assert T is not None, f"Missing T snapshot at iteration {snap_i + 1}"
            assert sims is not None, f"Missing sim snapshot at iteration {snap_i + 1}"
            assert cost is not None, f"Missing cost snapshot at iteration {snap_i + 1}"
            assert pop_Ts is not None, f"Missing population snapshot at iteration {snap_i + 1}"
        elif T is None and sims is None:
            sims = self._live_sims
            pop_Ts = self._live_pop_Ts
        if T is not None:
            self._T = T
        if cost is not None:
            self._cost_md.content = f"**iter {self._iteration} best cost:** {self._cost_text(cost)}"
        elif self._iteration == 0:
            self._cost_md.content = "**iter 0 (init)**"
        self._pose_md.content = (
            f"**timestep index={int(self._fps_idx[t])}**\n\n{pose_translation_rotation_text(self._T)}"
        )
        pcd_real = self._pcd_reals[t]
        self._real_handle.visible = pcd_real.shape[0] > 0
        if pcd_real.shape[0] > 0:
            self._real_handle.points = np.ascontiguousarray(transform_points(self._T, pcd_real))
        wxyz, pos = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum.wxyz = wxyz
        self._frustum.position = pos
        self._cam_axes.wxyz = wxyz
        self._cam_axes.position = pos
        if pop_Ts is not None:
            self._set_pop_axes(pop_Ts, hide_extra=True)
        self._sync_robot()
        self._draw_timestep(t, None if sims is None else sims[t])

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
                    axes_length=POP_AXES_LENGTH,
                    axes_radius=POP_AXES_RADIUS,
                    origin_radius=POP_ORIGIN_RADIUS,
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


def select_seed_pose(
    seed_Ts: list[np.ndarray],
    cost_fn: CostFn,
    vis: ExtrinsicsVisualizerBase | None,
) -> tuple[np.ndarray, float, list[Any]]:
    """Evaluate look-at seed poses and return the lowest-cost (T, cost, sims)."""
    assert len(seed_Ts) >= 1, "seed_Ts must be non-empty"
    best_cost = np.inf
    best_T: np.ndarray | None = None
    best_sims: list[Any] | None = None
    xyz_Ts: list[np.ndarray] = []
    t0_seed = time.perf_counter()
    pbar = tqdm(seed_Ts, desc="seed poses")
    for T in pbar:
        cost, sims = cost_fn(T)
        xyz = T[:3, 3]
        new_xyz = len(xyz_Ts) == 0 or not np.allclose(xyz_Ts[-1][:3, 3], xyz)
        if new_xyz:
            xyz_Ts.append(T.copy())
            if vis is not None:
                vis.set_population(xyz_Ts)
        if cost < best_cost:
            best_cost = cost
            best_T = T.copy()
            best_sims = sims
            if vis is not None:
                vis.set_best(sims, best_cost, best_T)
        if vis is not None and np.isfinite(best_cost):
            vis.record_seed_best(time.perf_counter() - t0_seed, best_cost)
        pbar.set_postfix(best=f"{best_cost:.6f}")
    assert best_T is not None and best_sims is not None, "Seed search produced no pose"
    return best_T, best_cost, best_sims


def optimize_extrinsics(
    cost_fn: CostFn,
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
    vis: ExtrinsicsVisualizerBase | None,
) -> tuple[np.ndarray, float, list[Any]]:
    """CMA-ES over 6D camera pose. Returns (best 4x4 T_world_cam, cost, sims at that pose)."""
    import cma

    assert n_seed_azimuth >= 1, f"n_seed_azimuth must be >= 1, got {n_seed_azimuth}"
    assert n_seed_polar >= 1, f"n_seed_polar must be >= 1, got {n_seed_polar}"
    assert n_seed_rolls >= 1, f"n_seed_rolls must be >= 1, got {n_seed_rolls}"
    assert n_seed_radii >= 1, f"n_seed_radii must be >= 1, got {n_seed_radii}"
    assert cma_sigma_pos > 0, f"cma_sigma_pos must be > 0, got {cma_sigma_pos}"
    assert cma_sigma_rot > 0, f"cma_sigma_rot must be > 0, got {cma_sigma_rot}"
    assert cma_early_stop_atol > 0, f"cma_early_stop_atol must be > 0, got {cma_early_stop_atol}"
    assert cma_early_stop_patience >= 1, f"cma_early_stop_patience must be >= 1, got {cma_early_stop_patience}"
    validate_seed_mode(seed_automatically, seed_from_gui, seed_pose)
    if seed_automatically:
        seed_Ts = generate_seed_poses(n_seed_azimuth, n_seed_polar, n_seed_rolls, n_seed_radii)
        radii = np.linspace(SEED_RADIUS_MIN_M, SEED_RADIUS_MAX_M, n_seed_radii, dtype=np.float64)
        print(
            f"[info] Seed search: n_azimuth={n_seed_azimuth} n_polar={n_seed_polar} n_rolls={n_seed_rolls} "
            f"n_radii={n_seed_radii} radii_m={radii.tolist()} n_total={len(seed_Ts)} "
            f"lookat={SEED_LOOKAT_TARGET.tolist()}"
        )
        t0_seed = time.perf_counter()
        best_T, best_cost, best_sims = select_seed_pose(seed_Ts, cost_fn, vis)
        log_elapsed(f"Seed search finished, best={best_cost:.6f}", t0_seed)
    elif seed_from_gui:
        assert vis is not None, "GUI seed selection requires an ExtrinsicsVisualizerBase"
        best_T, best_cost, best_sims = vis.select_seed_from_gui(gui_translation_step_m, gui_rotation_step_deg)
        print(f"[info] Manual seed selected: cost={best_cost:.6f} T_world_cam=\n{best_T}")
    else:
        assert seed_pose is not None, "seed_pose is required when not using --seed-automatically or --seed-from-gui"
        best_T = pose_from_xyz_wxyz(seed_pose)
        t0_seed = time.perf_counter()
        best_cost, best_sims = cost_fn(best_T)
        if vis is not None:
            vis.set_best(best_sims, best_cost, best_T)
            vis.record_seed_best(time.perf_counter() - t0_seed, best_cost)
        q_wxyz = mat2quat(best_T[:3, :3])
        print(
            f"[info] Seed pose xyz={best_T[:3, 3].tolist()} wxyz={q_wxyz.tolist()} "
            f"cost={best_cost:.6f} T_world_cam=\n{best_T}"
        )

    x0 = T_to_vec(best_T)
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
            T = vec_to_T(x)
            t0 = time.perf_counter()
            cost, sims = cost_fn(T)
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
                best_sims = sims
                if vis is not None:
                    t0 = time.perf_counter()
                    vis.set_best(sims, best_cost, best_T)
                    t_vis += time.perf_counter() - t0
            pbar.set_postfix(best=f"{best_cost:.6f}")
        t0 = time.perf_counter()
        es.tell(xs, costs)
        t_cma += time.perf_counter() - t0
        pop_min = float(min(costs))
        assert best_T is not None and best_sims is not None, "CMA-ES produced no pose"
        if vis is not None:
            t0 = time.perf_counter()
            vis.record_generation(generation, best_cost, pop_min, best_T, best_sims, pop_Ts)
            t_vis += time.perf_counter() - t0
        dt_gen = time.perf_counter() - t0_gen
        t_other = dt_gen - (t_cost + t_vis + t_cma)
        print(
            f"[info] gen {generation} ({dt_gen:.3f}s): best={best_cost:.6f}  gen_min={pop_min:.6f}  "
            f"gen_mean={float(np.mean(costs)):.6f}"
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
                    f"(gen_min={pop_min:.6f} spread={gen_min_spread:.6f} best={best_cost:.6f})"
                )
                break

    assert best_T is not None and best_sims is not None, "CMA-ES produced no pose"
    log_elapsed(f"CMA-ES finished ({generation} gens)", t0_opt)
    return best_T, best_cost, best_sims
