"""Measured camera calibration, keyed by camera_model_id (e.g. 'd435'), and measured workspace bounds, keyed by
workspace_id (e.g. 'vention-franka-table').

Values are hardware readouts, not guessed defaults.
"""

from __future__ import annotations

import numpy as np

from r2st.types import CameraIntrinsics, VerticalCylinder, WorkspaceBounds

MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 2.0


def cm_to_m(cm: float) -> float:
    return cm / 100.0


# Radius of the region excluded around each external camera's vertical mount
CAMERA_MOUNT_RADIUS_M = cm_to_m(20)

# Robot-base frame
WORKSPACE_BOUNDS: dict[str, WorkspaceBounds] = {
    # The bounds of the vention table. This is larger than the robot's reachable workspace. See
    # https://docs.google.com/drawings/d/1Mvzcg4S3asflx9s8VEFmOry6SM-qBRHE3Jz-pSo5bjY/edit?usp=sharing for details
    "vention-franka-table": WorkspaceBounds(
        min_x_m=cm_to_m(-21),
        max_x_m=cm_to_m(99),
        min_y_m=cm_to_m(-38.25),
        max_y_m=cm_to_m(42.75),
        min_z_m=cm_to_m(-1.0),
        max_z_m=cm_to_m(75.0),
        # Vertical mounts for the external cameras
        excluded=(
            VerticalCylinder(
                x_m=cm_to_m(104), y_m=cm_to_m(37.75), radius_m=CAMERA_MOUNT_RADIUS_M, min_z_m=cm_to_m(-2), max_z_m=2.0
            ),
            VerticalCylinder(
                x_m=cm_to_m(104), y_m=cm_to_m(-32.85), radius_m=CAMERA_MOUNT_RADIUS_M, min_z_m=cm_to_m(-2), max_z_m=2.0
            ),
        ),
    ),
}


def get_workspace_bounds(workspace_id: str) -> WorkspaceBounds:
    assert (
        workspace_id in WORKSPACE_BOUNDS
    ), f"Unknown workspace_id {workspace_id!r}. Available: {sorted(WORKSPACE_BOUNDS)}"
    return WORKSPACE_BOUNDS[workspace_id]


DEPTH_INTRINSICS: dict[str, CameraIntrinsics] = {
    "d435": CameraIntrinsics(
        width=640,
        height=480,
        fx=382.4931640625,
        fy=382.4931640625,
        cx=318.35821533203125,
        cy=239.32760620117188,
    ),
    # With align_depth.enable:=true the depth channel is resampled onto the COLOR grid, so
    # the aligned depth's camera_info is identical to the colour one -- that is what these
    # demos carry in d435_camera_info_depth.json. Nothing reads this entry while
    # --depth-intrinsics-source is `rgb` (the reprojection is skipped entirely), but the
    # key must exist because of the four-dict consistency assert below.
    "d435_720p": CameraIntrinsics(
        width=1280,
        height=720,
        fx=912.07177734375,
        fy=911.8675537109375,
        cx=642.824951171875,
        cy=355.07000732421875,
    ),
    # ZED 2i (serial 33386243), left camera at HD720. The ZED SDK's depth is computed for the rectified left image,
    # so the depth channel already lies on the colour grid: same K, identity depth-to-colour transform.
    "zed2i_hd720": CameraIntrinsics(
        width=1280,
        height=720,
        fx=522.732971,
        fy=522.732971,
        cx=640.685669,
        cy=365.680359,
    ),
    # ZED / ZED 2i / ZED Mini at VGA. Depth is registered to the rectified left image, so these equal the colour K
    # (the <cam>_camera_info_depth.json files panda_utils saves are identical to the colour ones).
    "zed_vga": CameraIntrinsics(
        width=672,
        height=376,
        fx=337.5408630371094,
        fy=337.5408630371094,
        cx=331.0454406738281,
        cy=178.57284545898438,
    ),
    "zed2i_vga": CameraIntrinsics(
        width=672,
        height=376,
        fx=261.0467834472656,
        fy=261.0467834472656,
        cx=335.6777648925781,
        cy=190.3715362548828,
    ),
    "zedm_vga": CameraIntrinsics(
        width=672,
        height=376,
        fx=364.70025634765625,
        fy=364.70025634765625,
        cx=338.702880859375,
        cy=187.4025115966797,
    ),
}

COLOR_INTRINSICS: dict[str, CameraIntrinsics] = {
    "d435": CameraIntrinsics(
        width=640,
        height=480,
        fx=608.0478515625,
        fy=607.9116821289062,
        cx=321.8832702636719,
        cy=236.7133331298828,
    ),
    # Same physical D435 (serial 841512071544) run at 1280x720, which is what the
    # 30-demo Square session was recorded at. Read straight off
    # /camera/d435/color/camera_info and saved into each demo's
    # d435_camera_info_color.json -- not rescaled from the 640x480 row, because the
    # sensor's crop/binning does not make that exact.
    "d435_720p": CameraIntrinsics(
        width=1280,
        height=720,
        fx=912.07177734375,
        fy=911.8675537109375,
        cx=642.824951171875,
        cy=355.07000732421875,
    ),
    # ZED 2i (serial 33386243) left rectified image at HD720, factory calibration (self-calibration disabled), as
    # reported by the ZED SDK's calibration_parameters.left_cam.
    "zed2i_hd720": CameraIntrinsics(
        width=1280,
        height=720,
        fx=522.732971,
        fy=522.732971,
        cx=640.685669,
        cy=365.680359,
    ),
    # Left rectified image at VGA, from the ROS camera_info that panda_utils' log_demonstration.py saves as
    # <cam>_camera_info_color.json next to each trajectory (rectified, so all distortion coefficients are 0).
    "zed_vga": CameraIntrinsics(
        width=672,
        height=376,
        fx=337.5408630371094,
        fy=337.5408630371094,
        cx=331.0454406738281,
        cy=178.57284545898438,
    ),
    "zed2i_vga": CameraIntrinsics(
        width=672,
        height=376,
        fx=261.0467834472656,
        fy=261.0467834472656,
        cx=335.6777648925781,
        cy=190.3715362548828,
    ),
    "zedm_vga": CameraIntrinsics(
        width=672,
        height=376,
        fx=364.70025634765625,
        fy=364.70025634765625,
        cx=338.702880859375,
        cy=187.4025115966797,
    ),
}

DEPTH_TO_COLOR_ROTATION: dict[str, np.ndarray] = {
    "d435": np.array(
        [[0.999749, 0.021574, 0.00599926], [-0.021599, 0.999758, 0.0041374], [-0.00590855, -0.00426594, 0.999973]],
        dtype=np.float64,
    ),
    # Physical transform between the depth and colour sensors: a property of the module,
    # not of the streaming resolution, so it is shared with the 640x480 entry.
    "d435_720p": np.array(
        [[0.999749, 0.021574, 0.00599926], [-0.021599, 0.999758, 0.0041374], [-0.00590855, -0.00426594, 0.999973]],
        dtype=np.float64,
    ),
    "zed2i_hd720": np.eye(3, dtype=np.float64),
    "zed_vga": np.eye(3, dtype=np.float64),
    "zed2i_vga": np.eye(3, dtype=np.float64),
    "zedm_vga": np.eye(3, dtype=np.float64),
}

DEPTH_TO_COLOR_TRANSLATION: dict[str, np.ndarray] = {
    "d435": np.array([0.0146080, -0.00004137, 0.0008026], dtype=np.float64),
    "d435_720p": np.array([0.0146080, -0.00004137, 0.0008026], dtype=np.float64),
    "zed2i_hd720": np.zeros(3, dtype=np.float64),
    "zed_vga": np.zeros(3, dtype=np.float64),
    "zed2i_vga": np.zeros(3, dtype=np.float64),
    "zedm_vga": np.zeros(3, dtype=np.float64),
}

assert (
    DEPTH_INTRINSICS.keys() == COLOR_INTRINSICS.keys()
), f"DEPTH_INTRINSICS keys {sorted(DEPTH_INTRINSICS)} != COLOR_INTRINSICS keys {sorted(COLOR_INTRINSICS)}"
assert (
    DEPTH_INTRINSICS.keys() == DEPTH_TO_COLOR_ROTATION.keys()
), f"DEPTH_INTRINSICS keys {sorted(DEPTH_INTRINSICS)} != DEPTH_TO_COLOR_ROTATION keys {sorted(DEPTH_TO_COLOR_ROTATION)}"
assert DEPTH_INTRINSICS.keys() == DEPTH_TO_COLOR_TRANSLATION.keys(), (
    f"DEPTH_INTRINSICS keys {sorted(DEPTH_INTRINSICS)} != "
    f"DEPTH_TO_COLOR_TRANSLATION keys {sorted(DEPTH_TO_COLOR_TRANSLATION)}"
)


def _require_camera_model(camera_model_id: str) -> None:
    assert (
        camera_model_id in DEPTH_INTRINSICS
    ), f"Unknown camera_model_id {camera_model_id!r}. Available: {sorted(DEPTH_INTRINSICS)}"


def get_depth_intrinsics(camera_model_id: str) -> CameraIntrinsics:
    _require_camera_model(camera_model_id)
    return DEPTH_INTRINSICS[camera_model_id]


def get_color_intrinsics(camera_model_id: str) -> CameraIntrinsics:
    _require_camera_model(camera_model_id)
    return COLOR_INTRINSICS[camera_model_id]


def get_depth_to_color_extrinsics(camera_model_id: str) -> tuple[np.ndarray, np.ndarray]:
    _require_camera_model(camera_model_id)
    return DEPTH_TO_COLOR_ROTATION[camera_model_id], DEPTH_TO_COLOR_TRANSLATION[camera_model_id]
