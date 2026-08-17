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
}

DEPTH_TO_COLOR_ROTATION: dict[str, np.ndarray] = {
    "d435": np.array(
        [[0.999749, 0.021574, 0.00599926], [-0.021599, 0.999758, 0.0041374], [-0.00590855, -0.00426594, 0.999973]],
        dtype=np.float64,
    ),
}

DEPTH_TO_COLOR_TRANSLATION: dict[str, np.ndarray] = {
    "d435": np.array([0.0146080, -0.00004137, 0.0008026], dtype=np.float64),
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
