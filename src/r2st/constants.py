"""Measured camera calibration, keyed by camera_model_id (e.g. 'd435').

Values are hardware readouts, not guessed defaults.
"""

from __future__ import annotations

import numpy as np

from r2st.types import CameraIntrinsics

MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 2.0

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
}

DEPTH_TO_COLOR_TRANSLATION: dict[str, np.ndarray] = {
    "d435": np.array([0.0146080, -0.00004137, 0.0008026], dtype=np.float64),
    "d435_720p": np.array([0.0146080, -0.00004137, 0.0008026], dtype=np.float64),
    "zed2i_hd720": np.zeros(3, dtype=np.float64),
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
