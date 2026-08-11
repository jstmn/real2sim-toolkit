import dataclasses
import pathlib
import pickle

import cv2
import h5py
import numpy as np
import tyro

"""
# Example usage:

uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_breadloaf/demonstration_0
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_breadloaf/demonstration_1
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_cube/demonstration_0
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_cube/demonstration_1
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_cube/demonstration_2
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_cube/demonstration_3
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_multi/demonstration_0
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_multi/demonstration_1
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_multi/demonstration_2
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_mustard/demonstration_0
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_mustard/demonstration_1
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_mustard/demonstration_2
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_mustard/demonstration_3
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_mustard/demonstration_4
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_oxiclean/demonstration_0
uv run python examples/merge_camera_streams.py --demo-dir  data/0802/0802_oxiclean/demonstration_1

"""


@dataclasses.dataclass
class Args:
    demo_dir: pathlib.Path
    """Directory containing cam_N_depth.h5 and cam_N_rgb_video.avi/.metadata files."""

    output: pathlib.Path | None = None
    """Output merged h5 path. Defaults to <demo_dir>/merged_sensor_data.h5."""


def _discover_cameras(demo_dir: pathlib.Path) -> list[str]:
    depth_files = sorted(demo_dir.glob("cam_*_depth.h5"))
    assert len(depth_files) > 0, f"No cam_*_depth.h5 files found in {demo_dir}"
    return [f.name[: -len("_depth.h5")] for f in depth_files]


def _load_depth_stream(depth_h5_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    assert depth_h5_path.is_file(), f"Depth file not found: {depth_h5_path}"
    with h5py.File(depth_h5_path, "r") as f:
        assert "timestamps" in f, f"{depth_h5_path}: no 'timestamps' dataset (recording is missing metadata)"
        depth_images = f["depth_images"][:]
        timestamps_ms = np.asarray(f["timestamps"][:], dtype=np.float64)
    assert depth_images.shape[0] == timestamps_ms.shape[0], (
        f"{depth_h5_path}: depth_images has {depth_images.shape[0]} frames but timestamps has "
        f"{timestamps_ms.shape[0]}"
    )
    assert np.all(np.diff(timestamps_ms) > 0), f"{depth_h5_path}: timestamps are not strictly increasing"
    return depth_images, timestamps_ms


def _load_rgb_stream(video_path: pathlib.Path, metadata_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    assert video_path.is_file(), f"RGB video not found: {video_path}"
    assert metadata_path.is_file(), f"RGB metadata not found: {metadata_path}"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    timestamps_ms = np.asarray(metadata["timestamps"], dtype=np.float64)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Failed to open video {video_path}"
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) > 0, f"No frames decoded from {video_path}"
    rgb_images = np.stack(frames, axis=0)
    assert rgb_images.shape[0] >= timestamps_ms.shape[0], (
        f"{video_path}: decoded {rgb_images.shape[0]} frames but metadata has {timestamps_ms.shape[0]} timestamps "
        "(fewer frames than timestamps -- can't be explained by a trailing untimestamped flush)"
    )
    if rgb_images.shape[0] > timestamps_ms.shape[0]:
        extra = rgb_images.shape[0] - timestamps_ms.shape[0]
        print(
            f"[warning] {video_path}: {extra} trailing video frame(s) have no logged timestamp "
            f"(decoded {rgb_images.shape[0]} vs {timestamps_ms.shape[0]} timestamps); dropping them"
        )
        rgb_images = rgb_images[: timestamps_ms.shape[0]]
    return rgb_images, timestamps_ms


def _nearest_indices(query_ts: np.ndarray, sorted_ref_ts: np.ndarray) -> np.ndarray:
    """For each timestamp in query_ts, the index into sorted_ref_ts of the closest timestamp."""
    idx = np.searchsorted(sorted_ref_ts, query_ts)
    idx = np.clip(idx, 1, len(sorted_ref_ts) - 1)
    left, right = sorted_ref_ts[idx - 1], sorted_ref_ts[idx]
    return np.where(query_ts - left <= right - query_ts, idx - 1, idx)


def _merge_camera(demo_dir: pathlib.Path, camera_name: str) -> dict[str, np.ndarray]:
    """Depth defines the merged timeline (it's the lower-rate stream); each depth frame is paired
    with its nearest-timestamp RGB frame. RGB and depth share a capture clock (see investigation:
    ~99% of depth timestamps exactly match an RGB timestamp), so this is a lossless pairing modulo
    occasional dropped RGB frames."""
    depth_images, depth_ts = _load_depth_stream(demo_dir / f"{camera_name}_depth.h5")
    rgb_images, rgb_ts = _load_rgb_stream(
        demo_dir / f"{camera_name}_rgb_video.avi", demo_dir / f"{camera_name}_rgb_video.metadata"
    )

    rgb_order = np.argsort(rgb_ts, kind="stable")
    rgb_ts_sorted = rgb_ts[rgb_order]
    nearest = _nearest_indices(depth_ts, rgb_ts_sorted)
    rgb_matched = rgb_images[rgb_order[nearest]]
    rgb_ts_matched = rgb_ts_sorted[nearest]

    delta_ms = np.abs(depth_ts - rgb_ts_matched)
    print(
        f"[info] {camera_name}: matched {len(depth_ts)} depth frames to rgb frames "
        f"(timestamp delta ms: mean={delta_ms.mean():.3f}, max={delta_ms.max():.3f})"
    )
    return {
        "rgb": rgb_matched,
        "depth": depth_images,
        "timestamp_ms": depth_ts,
        "rgb_timestamp_ms": rgb_ts_matched,
    }


def main(args: Args) -> None:
    assert args.demo_dir.is_dir(), f"Demo directory not found: {args.demo_dir}"
    output = args.output if args.output is not None else args.demo_dir / "merged_sensor_data.h5"
    tmp_output = output.with_name(output.name + ".tmp")

    camera_names = _discover_cameras(args.demo_dir)
    print(f"[info] Found cameras: {camera_names}")

    with h5py.File(tmp_output, "w") as out:
        for camera_name in camera_names:
            streams = _merge_camera(args.demo_dir, camera_name)
            group = out.create_group(f"obs/sensor_data/{camera_name}")
            group.create_dataset("rgb", data=streams["rgb"], compression="gzip", compression_opts=4)
            group.create_dataset("depth", data=streams["depth"], compression="gzip", compression_opts=4)
            group.create_dataset("timestamp_ms", data=streams["timestamp_ms"])
            group.create_dataset("rgb_timestamp_ms", data=streams["rgb_timestamp_ms"])
            print(
                f"[info] Wrote obs/sensor_data/{camera_name}: "
                f"rgb{streams['rgb'].shape}, depth{streams['depth'].shape}"
            )

    # Written under a .tmp name and renamed only on success, so a mid-run failure never leaves a
    # partial (e.g. missing a camera group) file at the final output path.
    tmp_output.rename(output)
    print(f"Merged sensor data written to {output}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
