import dataclasses
import pathlib
import time

import cv2
import h5py
import numpy as np
import tyro
from tqdm import tqdm

from r2st.core import SAM3Predictor, bbox_xyxy_from_mask, binary_mask_to_sam_mask_input
from r2st.utils import ImageUtils, validate_merged_camera_group

"""
# Example usage:
uv run python examples/generate_masks_across_trajectory.py \
    --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
    --camera cam_1 \
    --object-description "robot arm"
"""

_MASK_VIDEO_FPS = 15.0


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    object_description: str
    """Language description of the object to segment on frame 0, e.g. 'robot arm'."""

    sam_kmax: int = 5
    """Union at most this many highest-confidence SAM 3 masks into the frame-0 seed."""

    sam_score_threshold: float = 0.3
    """Keep a top-k SAM 3 mask in the frame-0 union only if its confidence is strictly above this."""

    max_frames: int | None = None
    """If set, only process the first N frames (useful for smoke tests)."""

    output_dir: pathlib.Path | None = None
    """Directory for masks, overlays, and video. Defaults to <h5_dir>/<h5_stem>/."""


def _log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


def _load_rgb(h5_path: pathlib.Path, camera: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        validate_merged_camera_group(f, h5_path, camera)
        rgb = f[f"obs/sensor_data/{camera}"]["rgb"][:]
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"Expected NxHxWx3 rgb, got {rgb.shape}"
    assert rgb.shape[0] >= 1, f"{h5_path}: camera '{camera}' has no frames"
    return rgb


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
    assert n_keep >= 1, f"No SAM 3 masks in top {n} with score > {score_threshold} (scores={scores[:n].tolist()})"
    union = np.any(masks[:n][keep], axis=0)
    print(
        f"[info] Unioned {n_keep}/{n} masks with score > {score_threshold} "
        f"(kept ranks={np.flatnonzero(keep).tolist()}, scores={scores[:n][keep].tolist()})"
    )
    return union


def _propagate_masks(rgb: np.ndarray, predictor: SAM3Predictor, seed_mask: np.ndarray) -> np.ndarray:
    """Propagate ``seed_mask`` (frame 0) through ``rgb`` with SAM 3 mask_input + bbox."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert n_frames >= 1, f"rgb must contain at least 1 frame, got {rgb.shape}"
    assert seed_mask.shape == (height, width), f"seed_mask {seed_mask.shape} != image {(height, width)}"
    assert seed_mask.dtype == bool, f"seed_mask dtype must be bool, got {seed_mask.dtype}"
    assert seed_mask.any(), "seed_mask is empty"
    masks = np.zeros((n_frames, height, width), dtype=bool)
    masks[0] = seed_mask
    mask_input = binary_mask_to_sam_mask_input(seed_mask)
    prev_mask = seed_mask
    for t in tqdm(range(1, n_frames), desc="Propagate masks"):
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        box = bbox_xyxy_from_mask(prev_mask)
        mask, _score, low_res = predictor.propagate_from_mask(image_bgr, mask_input, box)
        assert mask.shape == (height, width), f"frame {t}: mask {mask.shape} != {(height, width)}"
        masks[t] = mask
        mask_input = low_res
        prev_mask = mask
    return masks


def _save_outputs(
    rgb: np.ndarray,
    masks: np.ndarray,
    output_dir: pathlib.Path,
    camera: str,
    object_slug: str,
) -> pathlib.Path:
    """Write per-frame bool npy + demo overlay, and an mp4 of the overlays. Returns the video path."""
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"rgb must be (T, H, W, 3), got {rgb.shape}"
    n_frames, height, width, _ = rgb.shape
    assert masks.shape == (n_frames, height, width), f"masks {masks.shape} != {(n_frames, height, width)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"mask-propagated__{camera}__{object_slug}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _MASK_VIDEO_FPS,
        (width, height),
    )
    assert writer.isOpened(), f"Failed to open video writer at {video_path}"
    for t in range(n_frames):
        prefix = f"mask-propagated__{camera}__{object_slug}__idx={t}"
        np.save(output_dir / f"{prefix}.npy", masks[t])
        image_bgr = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        overlay_path = output_dir / f"{prefix}.demo.png"
        demo = ImageUtils.demo_overlay(image_bgr, masks[t])
        assert overlay_path.name.endswith(".demo.png"), f"demo overlay path must end with .demo.png, got {overlay_path}"
        assert cv2.imwrite(str(overlay_path), demo), f"Failed to write {overlay_path}"
        writer.write(demo)
    writer.release()
    assert video_path.is_file() and video_path.stat().st_size > 0, f"Failed to write {video_path}"
    print(f"[info] Saved {n_frames} masks, demo overlays, and video to {video_path}")
    return video_path


def main(args: Args) -> None:
    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert len(args.camera) > 0, "camera must not be empty"
    assert len(args.object_description) > 0, "object_description must not be empty"
    assert args.sam_kmax >= 1, f"sam_kmax must be >= 1, got {args.sam_kmax}"
    assert (
        0.0 <= args.sam_score_threshold <= 1.0
    ), f"sam_score_threshold must be in [0, 1], got {args.sam_score_threshold}"
    assert args.max_frames is None or args.max_frames >= 1, f"max_frames must be >= 1, got {args.max_frames}"

    print(f"[info] Loading RGB from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb = _load_rgb(args.h5_path, args.camera)
    if args.max_frames is not None:
        rgb = rgb[: args.max_frames]
    n_frames = int(rgb.shape[0])
    _log_elapsed(f"Loaded {n_frames} frames, resolution {rgb.shape[1:3]}", t0)

    predictor = SAM3Predictor()
    image_bgr0 = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR)
    print(f"[info] Segmenting '{args.object_description}' on frame 0 ...")
    t0 = time.perf_counter()
    ranked_masks, scores, phrases = ImageUtils.get_sam_masks_ranked(predictor, image_bgr0, args.object_description)
    _log_elapsed(f"Segmented '{args.object_description}' on frame 0 ({ranked_masks.shape[0]} masks)", t0)
    for rank in range(min(args.sam_kmax, ranked_masks.shape[0])):
        print(
            f"[info]   rank={rank} score={scores[rank]:.4f} phrase={phrases[rank]!r} "
            f"pixels={int(ranked_masks[rank].sum())}"
        )
    seed_mask = _union_top_sam_masks(ranked_masks, scores, args.sam_kmax, args.sam_score_threshold)

    if n_frames == 1:
        masks = seed_mask[None, ...]
    else:
        print(f"[info] Propagating frame-0 mask through {n_frames} frames ...")
        t0 = time.perf_counter()
        masks = _propagate_masks(rgb, predictor, seed_mask)
        _log_elapsed(f"Propagated mask through {n_frames} frames", t0)

    object_slug = args.object_description.replace(" ", "_")
    output_dir = args.output_dir if args.output_dir is not None else args.h5_path.parent / args.h5_path.stem
    video_path = _save_outputs(rgb, masks, output_dir, args.camera, object_slug)
    print(f"Object: {args.object_description}")
    print(f"Frames: {n_frames}")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
