import h5py
import numpy as np
import pytest

from r2st.utils import (
    rgb_depth_dataset_paths,
    validate_noisy_depth_camera_group,
    validate_stereo_depth_camera_group,
)

N_FRAMES, HEIGHT, WIDTH = 3, 4, 6


def _write_stereo_camera(f: h5py.File, camera: str, right: bool = True, right_width: int = WIDTH, depth: bool = True):
    group = f.create_group(f"obs/sensor_data/{camera}")
    group.create_dataset("timestamp_ms", data=np.arange(N_FRAMES, dtype=np.float64))
    eyes = (("left", WIDTH), ("right", right_width)) if right else (("left", WIDTH),)
    for eye, width in eyes:
        group.create_dataset(f"{eye}/rgb", data=np.zeros((N_FRAMES, HEIGHT, width, 3), dtype=np.uint8))
        group.create_dataset(f"{eye}/rgb_timestamp_ms", data=np.arange(N_FRAMES, dtype=np.float64))
    if depth:
        group.create_dataset("left/depth_noisy", data=np.zeros((N_FRAMES, HEIGHT, WIDTH), dtype=np.uint16))


def _write_single_image_camera(f: h5py.File, camera: str):
    group = f.create_group(f"obs/sensor_data/{camera}")
    group.create_dataset("rgb", data=np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8))
    group.create_dataset("depth", data=np.zeros((N_FRAMES, HEIGHT, WIDTH), dtype=np.uint16))
    group.create_dataset("timestamp_ms", data=np.arange(N_FRAMES, dtype=np.float64))
    group.create_dataset("rgb_timestamp_ms", data=np.arange(N_FRAMES, dtype=np.float64))


def test_stereo_layout_passes_both_validators(tmp_path):
    path = tmp_path / "ok.h5"
    with h5py.File(path, "w") as f:
        _write_stereo_camera(f, "zed")
        validate_noisy_depth_camera_group(f, path, "zed")
        validate_stereo_depth_camera_group(f, path, "zed")


def test_noisy_depth_does_not_need_right(tmp_path):
    path = tmp_path / "left_only.h5"
    with h5py.File(path, "w") as f:
        _write_stereo_camera(f, "zed", right=False)
        validate_noisy_depth_camera_group(f, path, "zed")
        with pytest.raises(AssertionError, match="right/rgb"):
            validate_stereo_depth_camera_group(f, path, "zed")


def test_stereo_depth_does_not_need_depth(tmp_path):
    path = tmp_path / "no_depth.h5"
    with h5py.File(path, "w") as f:
        _write_stereo_camera(f, "zed", depth=False)
        validate_stereo_depth_camera_group(f, path, "zed")
        with pytest.raises(AssertionError, match="left/depth_noisy"):
            validate_noisy_depth_camera_group(f, path, "zed")


def test_stereo_depth_rejects_mismatched_pair(tmp_path):
    path = tmp_path / "mismatch.h5"
    with h5py.File(path, "w") as f:
        _write_stereo_camera(f, "zed", right_width=WIDTH + 2)
        with pytest.raises(AssertionError, match="not a stereo pair"):
            validate_stereo_depth_camera_group(f, path, "zed")


def test_rgb_depth_dataset_paths_reads_both_layouts(tmp_path):
    path = tmp_path / "both.h5"
    with h5py.File(path, "w") as f:
        _write_stereo_camera(f, "zed")
        _write_single_image_camera(f, "cam_1")
        assert rgb_depth_dataset_paths(f, path, "zed") == (
            "obs/sensor_data/zed/left/rgb",
            "obs/sensor_data/zed/left/depth_noisy",
        )
        assert rgb_depth_dataset_paths(f, path, "cam_1") == ("obs/sensor_data/cam_1/rgb", "obs/sensor_data/cam_1/depth")
