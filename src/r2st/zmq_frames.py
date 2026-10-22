"""Read live RGB-D frames that panda_utils' `scripts/zed_frame_server.py` publishes over ZMQ.

The cameras are read through ROS 2 Humble, whose rclpy only exists for Python 3.10, so the frame server runs in that
environment and this toolkit subscribes to it. Wire format "zed_frame_v1", one single-part ZMQ message per frame:
    [uint32 little-endian header length][header, UTF-8 JSON][rgb bytes][depth bytes]
header keys: format, camera, stamp_ns (host wall clock ns since the Unix epoch, shared by rgb and depth), height,
width, K (colour camera matrix, 9 floats, row-major), rgb_dtype ("uint8", (H, W, 3) RGB), depth_dtype ("uint16",
(H, W) millimetres on the colour grid, 0 = no return). The format is defined in zed_frame_server.py; keep them in sync.
"""

from __future__ import annotations

import dataclasses
import json
import struct

import numpy as np
import zmq

FRAME_FORMAT = "zed_frame_v1"
_HEADER_KEYS = {"format", "camera", "stamp_ns", "height", "width", "K", "rgb_dtype", "depth_dtype"}


@dataclasses.dataclass(frozen=True)
class Frame:
    camera: str
    stamp_ns: int
    rgb: np.ndarray
    """(H, W, 3) uint8 RGB."""
    depth_mm: np.ndarray
    """(H, W) uint16 millimetres, registered to `rgb`."""
    K: np.ndarray
    """(3, 3) colour camera matrix the frame server read from camera_info."""


def decode_frame(message: bytes) -> Frame:
    assert len(message) >= 4, f"frame message is {len(message)} bytes, too short for the header length"
    (header_len,) = struct.unpack_from("<I", message, 0)
    header = json.loads(message[4 : 4 + header_len].decode("utf-8"))
    assert set(header) == _HEADER_KEYS, f"frame header keys {sorted(header)} != {sorted(_HEADER_KEYS)}"
    assert header["format"] == FRAME_FORMAT, f"frame format {header['format']!r}, expected {FRAME_FORMAT!r}"
    assert header["rgb_dtype"] == "uint8" and header["depth_dtype"] == "uint16", f"dtypes in {header}"
    H, W = header["height"], header["width"]
    rgb_len, depth_len = H * W * 3, H * W * 2
    body = memoryview(message)[4 + header_len :]
    assert len(body) == rgb_len + depth_len, f"frame body is {len(body)} bytes, expected {rgb_len + depth_len}"
    rgb = np.frombuffer(body[:rgb_len], dtype=np.uint8).reshape(H, W, 3)
    depth_mm = np.frombuffer(body[rgb_len:], dtype=np.uint16).reshape(H, W)
    K = np.asarray(header["K"], dtype=np.float64).reshape(3, 3)
    return Frame(camera=header["camera"], stamp_ns=int(header["stamp_ns"]), rgb=rgb, depth_mm=depth_mm, K=K)


def subscribe_newest_frames(context: zmq.Context, address: str) -> zmq.Socket:
    """SUB socket that holds only the newest frame (ZMQ_CONFLATE), so every recv returns the latest one queued."""
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(address)
    return socket
