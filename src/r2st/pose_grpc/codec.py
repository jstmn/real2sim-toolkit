"""Shared numpy <-> `pose_tracking_pb2` message conversion helpers, used by both the gRPC
server (r2st.pose_grpc.server) and client (r2st.pose_grpc.client)."""

from __future__ import annotations

import numpy as np

from r2st.pose_grpc import pose_tracking_pb2


def encode_image(array: np.ndarray) -> pose_tracking_pb2.Image:
    assert array.ndim in (2, 3), f"Image array must be HxW or HxWxC, got {array.shape}"
    height, width = array.shape[0], array.shape[1]
    channels = array.shape[2] if array.ndim == 3 else 1
    return pose_tracking_pb2.Image(
        data=np.ascontiguousarray(array).tobytes(),
        height=height,
        width=width,
        channels=channels,
    )


def decode_image(image: pose_tracking_pb2.Image, dtype: type) -> np.ndarray:
    assert image.height > 0 and image.width > 0, f"Invalid image dims: {image.height}x{image.width}"
    assert image.channels > 0, f"Invalid channel count: {image.channels}"
    expected_bytes = image.height * image.width * image.channels * np.dtype(dtype).itemsize
    assert len(image.data) == expected_bytes, (
        f"Image payload is {len(image.data)} bytes, expected {expected_bytes} for "
        f"{image.height}x{image.width}x{image.channels} {np.dtype(dtype)}"
    )
    array = np.frombuffer(image.data, dtype=dtype).reshape(image.height, image.width, image.channels)
    if image.channels == 1:
        array = array[..., 0]
    return array


def encode_k_matrix(K: np.ndarray) -> list[float]:
    assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
    return [float(v) for v in np.ascontiguousarray(K, dtype=np.float32).flatten()]


def decode_k_matrix(k_matrix) -> np.ndarray:
    assert len(k_matrix) == 9, f"k_matrix must have 9 elements, got {len(k_matrix)}"
    return np.array(k_matrix, dtype=np.float64).reshape(3, 3)


def encode_pose(pose_cam: np.ndarray) -> list[float]:
    assert pose_cam.shape == (4, 4), f"pose_cam must be 4x4, got {pose_cam.shape}"
    return [float(v) for v in np.ascontiguousarray(pose_cam, dtype=np.float32).flatten()]


def decode_pose(pose_cam) -> np.ndarray:
    assert len(pose_cam) == 16, f"pose_cam must have 16 elements, got {len(pose_cam)}"
    return np.array(pose_cam, dtype=np.float64).reshape(4, 4)
