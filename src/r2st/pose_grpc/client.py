"""gRPC client for the FoundationPose pose-tracking server (r2st.pose_grpc.server).

The server runs inside the FoundationPose Docker container; this client runs on the host (e.g.
in the `uv` venv) and mirrors the `r2st.pose_tracking.FoundationPoseTracker` API (`register`/
`track`) so callers don't need to care that FoundationPose itself runs remotely.
"""

from __future__ import annotations

import grpc
import numpy as np

from r2st.pose_grpc import codec, pose_tracking_pb2, pose_tracking_pb2_grpc


class FoundationPoseClient:
    def __init__(self, address: str = "localhost:50051"):
        self._channel = grpc.insecure_channel(address)
        self._stub = pose_tracking_pb2_grpc.PoseTrackingStub(self._channel)

    def register(
        self,
        mesh_path: str,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
        use_2d_tracker: bool = False,
        use_kalman_filter: bool = False,
        kalman_measurement_noise_scale: float = 0.05,
    ) -> np.ndarray:
        """Initialize the server's tracked object from its first frame; returns the 4x4 pose."""
        assert len(mesh_path) > 0, "mesh_path must not be empty"
        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
        assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
        assert depth_m.shape == color_rgb.shape[:2], f"depth shape {depth_m.shape} != color {color_rgb.shape[:2]}"
        assert mask.shape == color_rgb.shape[:2], f"mask shape {mask.shape} != color {color_rgb.shape[:2]}"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert iteration >= 1, f"iteration must be >= 1, got {iteration}"
        assert use_2d_tracker or not use_kalman_filter, (
            "use_kalman_filter requires use_2d_tracker=True: the filter fuses the 2D tracker's "
            "image-plane measurement with FoundationPose's own pose estimate each frame."
        )

        request = pose_tracking_pb2.RegisterRequest(
            mesh_path=str(mesh_path),
            color_rgb=codec.encode_image(color_rgb.astype(np.uint8)),
            depth_m=codec.encode_image(depth_m.astype(np.float32)),
            mask=codec.encode_image(mask.astype(np.uint8)),
            k_matrix=codec.encode_k_matrix(K),
            iteration=iteration,
            use_2d_tracker=use_2d_tracker,
            use_kalman_filter=use_kalman_filter,
            kalman_measurement_noise_scale=kalman_measurement_noise_scale,
        )
        response = self._stub.Register(request)
        return codec.decode_pose(response.pose_cam)

    def track(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
    ) -> np.ndarray:
        """Refine the pose of the server's currently-registered object in a new frame."""
        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
        assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
        assert depth_m.shape == color_rgb.shape[:2], f"depth shape {depth_m.shape} != color {color_rgb.shape[:2]}"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert iteration >= 1, f"iteration must be >= 1, got {iteration}"

        request = pose_tracking_pb2.TrackRequest(
            color_rgb=codec.encode_image(color_rgb.astype(np.uint8)),
            depth_m=codec.encode_image(depth_m.astype(np.float32)),
            k_matrix=codec.encode_k_matrix(K),
            iteration=iteration,
        )
        response = self._stub.Track(request)
        return codec.decode_pose(response.pose_cam)

    def close(self) -> None:
        self._channel.close()
