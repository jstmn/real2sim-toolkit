"""gRPC server exposing `r2st.pose_tracking.FoundationPoseTracker`.

Must run *inside* the FoundationPose Docker container (see src/r2st/FoundationPose/docker),
which has the CUDA/PyTorch/nvdiffrast stack FoundationPose needs. The rest of the toolkit runs
outside the container (e.g. in the `uv` venv) and talks to this server over gRPC via
`r2st.pose_grpc.client.FoundationPoseClient`.

The server holds a single active tracked object at a time: `Register` (re)initializes it from
its first frame, and subsequent `Track` calls refine that pose frame-to-frame until the next
`Register` call replaces it -- this mirrors `FoundationPoseTracker` itself, just remoted over
gRPC. Model weights (scorer/refiner) and the CUDA rasterizer context are loaded once at server
startup and reused across every `Register` call, so switching to a new mesh/video doesn't pay
that cost again.

Usage (inside the container, with this repo's `src/` on PYTHONPATH):
    PYTHONPATH=src python -m r2st.pose_grpc.server
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent import futures

import grpc
import numpy as np
import tyro

from r2st.pose_grpc import codec, pose_tracking_pb2, pose_tracking_pb2_grpc
from r2st.pose_tracking import FoundationPoseTracker, _load_foundationpose_deps


@dataclasses.dataclass
class Args:
    port: int = 50051
    """Port to listen on."""

    debug_dir: str = "debug_fp"
    """Directory (inside the container) for FoundationPose debug visualizations."""

    max_workers: int = 4
    """Max gRPC worker threads."""


class PoseTrackingServicer(pose_tracking_pb2_grpc.PoseTrackingServicer):
    def __init__(self, debug_dir: str):
        self._debug_dir = debug_dir
        dr, _, PoseRefinePredictor, ScorePredictor, _, _ = _load_foundationpose_deps()
        print("[pose_server] Loading FoundationPose scorer/refiner/rasterizer (once) ...")
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()
        print("[pose_server] Ready.")

        self._lock = threading.Lock()
        self._tracker: FoundationPoseTracker | None = None

    def Register(self, request, context):
        assert request.mesh_path, "mesh_path must not be empty"
        assert request.iteration >= 1, f"iteration must be >= 1, got {request.iteration}"
        color_rgb = codec.decode_image(request.color_rgb, np.uint8)
        depth_m = codec.decode_image(request.depth_m, np.float32)
        mask = codec.decode_image(request.mask, np.uint8).astype(bool)
        K = codec.decode_k_matrix(request.k_matrix)

        with self._lock:
            tracker = FoundationPoseTracker(
                mesh_file=request.mesh_path,
                debug_dir=self._debug_dir,
                use_2d_tracker=request.use_2d_tracker,
                use_kalman_filter=request.use_kalman_filter,
                kalman_measurement_noise_scale=request.kalman_measurement_noise_scale,
                scorer=self._scorer,
                refiner=self._refiner,
                glctx=self._glctx,
            )
            pose_cam = tracker.register(color_rgb, depth_m, mask, K, iteration=request.iteration)
            self._tracker = tracker
        print(f"[pose_server] Registered mesh '{request.mesh_path}'")
        return pose_tracking_pb2.PoseResponse(pose_cam=codec.encode_pose(pose_cam))

    def Track(self, request, context):
        assert request.iteration >= 1, f"iteration must be >= 1, got {request.iteration}"
        color_rgb = codec.decode_image(request.color_rgb, np.uint8)
        depth_m = codec.decode_image(request.depth_m, np.float32)
        K = codec.decode_k_matrix(request.k_matrix)

        with self._lock:
            assert self._tracker is not None, "Must call Register before Track"
            pose_cam = self._tracker.track(color_rgb, depth_m, K, iteration=request.iteration)
        return pose_tracking_pb2.PoseResponse(pose_cam=codec.encode_pose(pose_cam))


def main(args: Args) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    pose_tracking_pb2_grpc.add_PoseTrackingServicer_to_server(PoseTrackingServicer(args.debug_dir), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"[pose_server] Listening on port {args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
