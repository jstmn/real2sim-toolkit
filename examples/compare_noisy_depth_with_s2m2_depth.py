import dataclasses
import pathlib
import time

import h5py
import numpy as np
import tyro

from r2st.constants import get_color_intrinsics, get_stereo_baseline_m
from r2st.geometry import depth_mm_to_meters, depth_rgb_to_pointcloud
from r2st.stereo_depth import S2M2StereoDepth
from r2st.utils import (
    validate_noisy_depth_camera_group,
    validate_stereo_depth_camera_group,
)

"""
Compare each camera's own depth (`left/depth_noisy`, e.g. from the ZED SDK) with S2M2 stereo depth computed from
`left/rgb` and `right/rgb`. Both are unprojected to point clouds in the camera's optical frame and shown in viser, with
a timestep slider and per-camera toggles for each cloud.

# Example usage:
    uv run python examples/compare_noisy_depth_with_s2m2_depth.py \
        --h5-path data/demonstrations/Oct26_2026_datacollect/PickCube__10-26-2026__18-55-12/traj_0__VGA.h5 \
        --cameras zed zed2i zedm \
        --camera-model-ids zed_vga zed2i_vga zedm_vga
"""

# Solid colors for the "source" color mode
_NOISY_COLOR = (255, 140, 0)
_S2M2_COLOR = (40, 140, 255)
_COLOR_MODES = ("rgb", "source")


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a stereo-layout h5 (`obs/sensor_data/<cam>/left/[rgb, depth_noisy]`, `.../right/rgb`)."""

    cameras: list[str]
    """Camera names in the h5, e.g. `zed zed2i zedm`."""

    camera_model_ids: list[str]
    """Camera model id for each of --cameras, in the same order, e.g. `zed_vga zed2i_vga zedm_vga`."""

    s2m2_model_type: str = "L"
    """S2M2 model size: S, M, L or XL."""

    s2m2_refine_iter: int = 3
    """S2M2 local refinement iterations."""

    stride: int = 2
    """Unproject every `stride`-th pixel in each direction."""

    max_frames: int | None = None
    """If set, only process the first N frames."""

    point_size: float = 0.002
    """Viser point size in meters."""


@dataclasses.dataclass
class CameraDepths:
    name: str
    K: np.ndarray
    left_rgb: np.ndarray
    """`(T, H, W, 3)` uint8"""
    noisy_depth_m: np.ndarray
    """`(T, H, W)` float32, 0 where the camera has no depth"""
    s2m2_depth_m: np.ndarray
    """`(T, H, W)` float32, 0 where S2M2 disparity is not positive"""


def _log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


def _load_camera_depths(
    h5_path: pathlib.Path, camera: str, camera_model_id: str, stereo: S2M2StereoDepth, max_frames: int | None
) -> CameraDepths:
    """Read `camera`'s left/right RGB and noisy depth, and run S2M2 on every frame."""
    with h5py.File(h5_path, "r") as f:
        validate_noisy_depth_camera_group(f, h5_path, camera)
        validate_stereo_depth_camera_group(f, h5_path, camera)
        group = f[f"obs/sensor_data/{camera}"]
        frames = slice(None) if max_frames is None else slice(0, max_frames)
        left_rgb = group["left/rgb"][frames]
        right_rgb = group["right/rgb"][frames]
        noisy_depth_m = depth_mm_to_meters(group["left/depth_noisy"][frames])

    intrinsics = get_color_intrinsics(camera_model_id)
    H, W = left_rgb.shape[1:3]
    assert (H, W) == (
        intrinsics.height,
        intrinsics.width,
    ), f"{camera} left/rgb is {H}x{W} but {camera_model_id} intrinsics are {intrinsics.height}x{intrinsics.width}"
    baseline_m = get_stereo_baseline_m(camera_model_id)
    print(f"[info] {camera} ({camera_model_id}): fx={intrinsics.fx:.3f} baseline={baseline_m * 1000:.2f} mm")

    s2m2_depth_m = stereo.depth_m_sequence(left_rgb, right_rgb, intrinsics.fx, baseline_m, desc=f"S2M2 {camera}")
    return CameraDepths(
        name=camera,
        K=intrinsics.intrinsic_matrix,
        left_rgb=left_rgb,
        noisy_depth_m=noisy_depth_m,
        s2m2_depth_m=s2m2_depth_m,
    )


class DepthComparisonViewer:
    """Viser scene with a noisy and an S2M2 point cloud per camera, a timestep slider and per-cloud toggles. Every
    camera's clouds are in its own optical frame, so cameras overlap at the origin; only the first is shown at start."""

    def __init__(self, cameras: list[CameraDepths], stride: int, point_size: float) -> None:
        import viser

        assert len(cameras) > 0, "cameras must not be empty"
        self._num_frames = len(cameras[0].left_rgb)
        for cam in cameras:
            assert len(cam.left_rgb) == self._num_frames, f"{cam.name} has {len(cam.left_rgb)} frames"
        self._cameras = cameras
        self._stride = stride

        self._server = viser.ViserServer()
        self._server.scene.set_up_direction("-y")
        self._server.scene.world_axes.visible = True
        self._server.scene.world_axes.scale = 0.15
        self._server.initial_camera.up = (0.0, -1.0, 0.0)
        self._server.initial_camera.look_at = (0.0, 0.0, 1.0)
        self._server.initial_camera.position = (0.0, -0.4, -0.6)

        self._slider = self._server.gui.add_slider("timestep", min=0, max=self._num_frames - 1, step=1, initial_value=0)
        self._color_mode = self._server.gui.add_dropdown(
            "color by", _COLOR_MODES, initial_value="rgb", hint="source: noisy = orange, S2M2 = blue"
        )
        # (camera, source) -> (checkbox, point cloud handle)
        self._clouds = {}
        for i, cam in enumerate(cameras):
            with self._server.gui.add_folder(cam.name):
                for source, label in (("noisy", "noisy depth"), ("s2m2", "S2M2 depth")):
                    checkbox = self._server.gui.add_checkbox(label, initial_value=i == 0)
                    handle = self._server.scene.add_point_cloud(
                        f"/{cam.name}/{source}",
                        points=np.zeros((1, 3), dtype=np.float32),
                        colors=np.zeros((1, 3), dtype=np.uint8),
                        point_size=point_size,
                        point_shape="circle",
                    )
                    checkbox.on_update(lambda _: self._refresh())
                    self._clouds[(cam.name, source)] = (checkbox, handle)
        self._slider.on_update(lambda _: self._refresh())
        self._color_mode.on_update(lambda _: self._refresh())
        self._refresh()
        print(f"[info] Viser depth comparison at http://{self._server.get_host()}:{self._server.get_port()}")

    def _refresh(self) -> None:
        """Recompute the visible clouds for the current timestep and color mode."""
        t = int(self._slider.value)
        by_source = self._color_mode.value == "source"
        for cam in self._cameras:
            for source, depth_m, solid in (
                ("noisy", cam.noisy_depth_m[t], _NOISY_COLOR),
                ("s2m2", cam.s2m2_depth_m[t], _S2M2_COLOR),
            ):
                checkbox, handle = self._clouds[(cam.name, source)]
                handle.visible = checkbox.value
                if not checkbox.value:
                    continue
                pts, colors = depth_rgb_to_pointcloud(depth_m, cam.left_rgb[t], cam.K, stride=self._stride)
                if by_source:
                    colors = np.broadcast_to(np.array(solid, dtype=np.uint8), colors.shape)
                handle.points = pts
                handle.colors = colors

    def wait(self) -> None:
        self._server.sleep_forever()


def main(args: Args) -> None:
    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert len(args.cameras) > 0, "cameras must not be empty"
    assert len(args.cameras) == len(
        args.camera_model_ids
    ), f"--cameras {args.cameras} and --camera-model-ids {args.camera_model_ids} must have the same length"
    assert len(set(args.cameras)) == len(args.cameras), f"--cameras has duplicates: {args.cameras}"
    assert args.stride >= 1, f"stride must be >= 1, got {args.stride}"
    assert args.max_frames is None or args.max_frames >= 1, f"max_frames must be >= 1, got {args.max_frames}"
    assert args.point_size > 0, f"point_size must be > 0, got {args.point_size}"

    t0 = time.perf_counter()
    stereo = S2M2StereoDepth(
        args.s2m2_model_type, refine_iter=args.s2m2_refine_iter, allow_negative=False, device="cuda"
    )
    _log_elapsed(f"Loaded S2M2 {args.s2m2_model_type}", t0)

    cameras = []
    for camera, camera_model_id in zip(args.cameras, args.camera_model_ids, strict=True):
        t0 = time.perf_counter()
        cameras.append(_load_camera_depths(args.h5_path, camera, camera_model_id, stereo, args.max_frames))
        _log_elapsed(f"{camera}: {len(cameras[-1].left_rgb)} frames", t0)

    viewer = DepthComparisonViewer(cameras, args.stride, args.point_size)
    viewer.wait()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
