"""Measured RealSense D435 color/depth intrinsics for the lab cameras.

Values are hardware readouts (same numbers as in the upstream MPCM constants), not
guessed defaults.
"""

from __future__ import annotations

from r2st.types import CameraIntrinsics

REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480

REALSENSE_DEPTH_INTRINSICS: dict[str, CameraIntrinsics] = {
    "d435": CameraIntrinsics(
        width=REALSENSE_WIDTH,
        height=REALSENSE_HEIGHT,
        fx=382.4931640625,
        fy=382.4931640625,
        cx=318.35821533203125,
        cy=239.32760620117188,
    ),
}

REALSENSE_COLOR_INTRINSICS: dict[str, CameraIntrinsics] = {
    "d435": CameraIntrinsics(
        width=REALSENSE_WIDTH,
        height=REALSENSE_HEIGHT,
        fx=608.0478515625,
        fy=607.9116821289062,
        cx=321.8832702636719,
        cy=236.7133331298828,
    ),
}


def get_depth_intrinsics(realsense_id: str) -> CameraIntrinsics:
    assert (
        realsense_id in REALSENSE_DEPTH_INTRINSICS
    ), f"Unknown realsense_id {realsense_id!r}. Available: {sorted(REALSENSE_DEPTH_INTRINSICS)}"
    return REALSENSE_DEPTH_INTRINSICS[realsense_id]


def get_color_intrinsics(realsense_id: str) -> CameraIntrinsics:
    assert (
        realsense_id in REALSENSE_COLOR_INTRINSICS
    ), f"Unknown realsense_id {realsense_id!r}. Available: {sorted(REALSENSE_COLOR_INTRINSICS)}"
    return REALSENSE_COLOR_INTRINSICS[realsense_id]
