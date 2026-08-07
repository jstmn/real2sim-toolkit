import dataclasses
import pathlib

import cv2
import h5py
import numpy as np
import torch
import tyro

from r2st.core import GroundedSAMPredictor
from r2st.utils import ImageUtils

"""
# Example usage:
uv run python examples/track_object.py \
    --h5-path data/0802_mustard/demonstration_0/merged_sensor_data.h5 \
    --camera cam_1 \
    --object-description "mustard bottle"
"""


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see examples/merge_camera_streams.py)."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    object_description: str
    """Language description of the object to segment and mesh, e.g. 'mustard bottle'."""

    output_dir: pathlib.Path = pathlib.Path("data/meshyai")
    """Directory to save generated mesh assets."""


_EXPECTED_DATASETS = ("rgb", "depth", "timestamp_ms", "rgb_timestamp_ms")


def _validate_camera_group(f: h5py.File, h5_path: pathlib.Path, camera: str) -> None:
    """Check that `f` matches the merged sensor-data format from examples/merge_camera_streams.py:
    obs/sensor_data/{camera}/[rgb, depth, timestamp_ms, rgb_timestamp_ms]."""
    assert "obs/sensor_data" in f, f"{h5_path}: missing 'obs/sensor_data' group (not a merged sensor-data h5?)"
    sensor_data = f["obs/sensor_data"]
    group_path = f"obs/sensor_data/{camera}"
    assert group_path in f, f"Camera '{camera}' not found in {h5_path}. Available: {sorted(sensor_data.keys())}"
    group = f[group_path]
    for name in _EXPECTED_DATASETS:
        assert name in group, f"{h5_path}:{group_path} missing dataset '{name}'"

    rgb, depth, timestamp_ms, rgb_timestamp_ms = (group[name] for name in _EXPECTED_DATASETS)
    assert rgb.ndim == 4 and rgb.shape[3] == 3, f"{h5_path}:{group_path}/rgb must be NxHxWx3, got {rgb.shape}"
    assert rgb.dtype == np.uint8, f"{h5_path}:{group_path}/rgb must be uint8, got {rgb.dtype}"
    assert depth.ndim == 3, f"{h5_path}:{group_path}/depth must be NxHxW, got {depth.shape}"
    assert timestamp_ms.ndim == 1, f"{h5_path}:{group_path}/timestamp_ms must be 1D, got {timestamp_ms.shape}"
    assert (
        rgb_timestamp_ms.ndim == 1
    ), f"{h5_path}:{group_path}/rgb_timestamp_ms must be 1D, got {rgb_timestamp_ms.shape}"

    num_frames = timestamp_ms.shape[0]
    assert num_frames > 0, f"{h5_path}:{group_path} has no frames"
    assert (
        rgb.shape[0] == num_frames
    ), f"{h5_path}:{group_path}: rgb has {rgb.shape[0]} frames but timestamp_ms has {num_frames}"
    assert (
        depth.shape[0] == num_frames
    ), f"{h5_path}:{group_path}: depth has {depth.shape[0]} frames but timestamp_ms has {num_frames}"
    assert rgb_timestamp_ms.shape[0] == num_frames, (
        f"{h5_path}:{group_path}: rgb_timestamp_ms has {rgb_timestamp_ms.shape[0]} frames but timestamp_ms has "
        f"{num_frames}"
    )
    assert (
        depth.shape[1:] == rgb.shape[1:3]
    ), f"{h5_path}:{group_path}: depth resolution {depth.shape[1:]} != rgb resolution {rgb.shape[1:3]}"


def _load_first_frame(h5_path: pathlib.Path, camera: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        _validate_camera_group(f, h5_path, camera)
        return f[f"obs/sensor_data/{camera}/rgb"][0]


def _get_sam_mask(image_bgr: np.ndarray, object_description: str) -> np.ndarray:
    predictor = GroundedSAMPredictor()
    assert predictor._sam_predictor is not None, "GroundedSAM predictor not loaded"
    assert predictor._bert_model is not None, "GroundedSAM bert model not loaded"
    masks = predictor.get_sam_mask(image_bgr, object_description)
    assert isinstance(masks, torch.Tensor), f"Expected torch.Tensor masks, got {type(masks)}"
    mask = masks[0, 0].cpu().numpy()
    mask = mask.astype(bool)
    assert mask.shape[:2] == image_bgr.shape[:2], f"Mask shape {mask.shape[:2]} != image {image_bgr.shape[:2]}"
    return mask


def _generate_mesh_with_meshy(image_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    from r2st.meshy import MeshyAPI
    api = MeshyAPI()
    result_path = pathlib.Path(api.image_to_3d(image_path=image_path, output_dir=output_dir, enable_pbr=True))
    assert result_path.is_file(), f"MeshyAPI did not create result at {result_path}"
    return result_path


def main(args: Args) -> None:
    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert len(args.object_description) > 0, "object_description must not be empty"

    print(f"[info] Loading first frame from {args.h5_path} ({args.camera}) ...")
    rgb = _load_first_frame(args.h5_path, args.camera)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"Expected HxWx3 rgb frame, got {rgb.shape}"
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    print(f"[info] Frame shape: {image_bgr.shape}")

    print(f"[info] Segmenting '{args.object_description}' with GroundedSAM ...")
    mask = _get_sam_mask(image_bgr, args.object_description)
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    assert mask.sum() > 0, f"Segmentation mask is empty for description '{args.object_description}'"
    print(f"[info] Mask pixels: {int(mask.sum())} / {mask.size}")

    object_slug = args.object_description.replace(" ", "_")
    asset_dir = args.output_dir / f"{args.camera}__{object_slug}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    masked = image_bgr.copy()
    masked[np.logical_not(mask)] = 0
    masked_cropped = ImageUtils.crop_to_mask(masked, mask)
    masked_cropped_path = asset_dir / f"{object_slug}__masked_cropped.png"
    cv2.imwrite(str(masked_cropped_path), masked_cropped)
    print(
        f"[info] Saved masked cropped image to {masked_cropped_path} "
        f"({masked_cropped.shape[1]}x{masked_cropped.shape[0]})"
    )

    print("[info] Generating mesh with Meshy ...")
    glb_path = asset_dir / f"{object_slug}_glb.glb"
    meshy_result = _generate_mesh_with_meshy(masked_cropped_path, asset_dir)
    assert meshy_result.exists(), f"Mesh file not created at {meshy_result}"
    if meshy_result != glb_path:
        import shutil

        shutil.copy2(str(meshy_result), str(glb_path))
    assert glb_path.exists(), f"Mesh file not created at {glb_path}"
    print("Mesh generated successfully!")
    print(f"Object: {args.object_description}")
    print(f"Mask: {mask.shape}, {int(mask.sum())} foreground pixels")
    print(f"Mesh: {glb_path} ({glb_path.stat().st_size} bytes)")

    # TODO: Call FoundationPose (via the gRPC server we're about to write) to track
    # `args.object_description`'s 6D pose across the rest of `args.h5_path`'s `camera` trajectory,
    # using `glb_path` as the tracked mesh and the first frame's `mask` to register the initial pose.


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
