import dataclasses
import json
import pathlib
import time

import cv2
import numpy as np
import tyro
import zmq

from r2st.constants import get_color_intrinsics
from r2st.core import SAM3Predictor
from r2st.geometry import (
    depth_mm_to_meters,
    project_axes_to_image,
    transform_pose_cam_to_world,
)
from r2st.pose_grpc.client import FoundationPoseClient
from r2st.utils import ImageUtils
from r2st.zmq_frames import Frame, decode_frame, subscribe_newest_frames

"""
Live version of examples/track_object.py: track one object in the frames panda_utils' zed_frame_server.py publishes,
and publish its robot-base pose over ZMQ for a controller to read.

The object is (re)registered on request: a client sends {"cmd": "register"} on --command-address (ZMQ REQ/REP). The
tracker waits for a frame captured after the request arrived, segments --object-description in it with SAM 3,
registers --mesh-path with FoundationPose (gRPC server in the container), replies, then calls FoundationPose's Track on
every newer frame until the next register request. Each new episode registers again, because the object starts
somewhere new.

Reply to a register request, JSON:
    {"registration_id": int, "stamp_ns": int, "mask_pixels": int, "T_base_obj": 16 floats, row-major}
Pose messages on --pose-address (ZMQ PUB), JSON, one per tracked frame, format "object_pose_v1":
    {"format", "object", "camera", "registration_id", "stamp_ns", "T_base_obj", "T_cam_obj"}
`stamp_ns` is the camera frame's capture time (host wall clock, ns since the Unix epoch). T_base_obj is the mesh frame in
the robot base frame, from --extrinsics-path (examples/estimate_camera_extrinsics.py). The mesh is used as given: it is
not regenerated or rescaled.

Each run writes to a new --debug-dir/<date>__<time>/ folder: per registration, SAM 3's mask images and the registered
pose drawn on the frame, and latest_overlay.png, rewritten every few seconds with the newest tracked pose.

Example usage (FoundationPose gRPC server must already be running in the container, and zed_frame_server.py must be
publishing --frame-address):
    uv run python examples/track_object_live.py \
        --frame-address tcp://127.0.0.1:5600 \
        --camera zed2i \
        --camera-model-id zed2i_vga \
        --object-description "red cube" \
        --mesh-path data/meshyai/zed2i__red_cube/red_cube_glb_scaled.glb \
        --extrinsics-path data/Oct20_2026_datacollect/extrinsics_zed2i.yaml \
        --pose-address tcp://127.0.0.1:5610 \
        --command-address tcp://127.0.0.1:5611 \
        --debug-dir data/track_object_live
"""

POSE_FORMAT = "object_pose_v1"
STATUS_PERIOD_S = 5.0
POLL_TIMEOUT_MS = 1000
# Tolerance, in pixels, between the frame server's camera_info K and the --camera-model-id intrinsics the extrinsics
# were calibrated with. The ZED recalibrates at every launch, which moves K by hundredths of a pixel; a different
# resolution moves it by tens to hundreds. Tracking uses the --camera-model-id K either way.
K_ATOL_PX = 1.0


@dataclasses.dataclass
class Args:
    frame_address: str
    """ZMQ address zed_frame_server.py publishes this camera on, e.g. tcp://127.0.0.1:5600."""

    camera: str
    """Camera id; must match the frames and the extrinsics YAML, e.g. 'zed2i'."""

    camera_model_id: str
    """Camera model id for calibration lookup, e.g. 'zed2i_vga'. Must match the extrinsics YAML."""

    object_description: str
    """Language description of the object for SAM 3, e.g. 'red cube'."""

    mesh_path: pathlib.Path
    """Scaled mesh to register, e.g. the *_glb_scaled.glb examples/track_object.py writes."""

    extrinsics_path: pathlib.Path
    """YAML from examples/estimate_camera_extrinsics.py (robot-base T camera)."""

    pose_address: str
    """ZMQ address to publish poses on (PUB, bind), e.g. tcp://127.0.0.1:5610."""

    command_address: str
    """ZMQ address to take register requests on (REP, bind), e.g. tcp://127.0.0.1:5611."""

    debug_dir: pathlib.Path
    """Directory for registration masks and pose overlays; each run gets a new <date>__<time> folder in it."""

    server_address: str = "localhost:50051"
    """gRPC address of the FoundationPose pose-tracking server."""

    est_refine_iter: int = 5
    """FoundationPose register refine iterations."""

    track_refine_iter: int = 5
    """FoundationPose track refine iterations."""

    use_2d_tracker: bool = True
    """Re-anchor FoundationPose (x, y) each frame with Cutie (FoundationPose++)."""

    use_kalman_filter: bool = True
    """Fuse Cutie (x, y) with FoundationPose via a 6-DoF Kalman filter. Requires use_2d_tracker."""


def _load_camera_extrinsics_yaml(path: pathlib.Path, camera: str, camera_model_id: str) -> np.ndarray:
    """Load robot-base T camera (4x4) from an Example 4 extrinsics YAML."""
    import yaml

    assert path.is_file(), f"Extrinsics YAML not found: {path}"
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"Extrinsics YAML must be a mapping, got {type(payload)}"
    assert payload["camera"] == camera, f"Extrinsics YAML camera {payload['camera']!r} != --camera {camera!r}"
    assert (
        payload["camera_model_id"] == camera_model_id
    ), f"Extrinsics YAML camera_model_id {payload['camera_model_id']!r} != --camera-model-id {camera_model_id!r}"
    T = np.asarray(payload["matrix"], dtype=np.float64)
    assert T.shape == (4, 4), f"Extrinsics matrix must be 4x4, got {T.shape}"
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]), f"Bad homogeneous row: {T[3]}"
    return T


def _draw_pose_axes(color_rgb: np.ndarray, pose_cam: np.ndarray, K: np.ndarray, axis_len: float = 0.08) -> np.ndarray:
    """RGB image with the XYZ axes of `pose_cam` drawn on it."""
    pts = project_axes_to_image(pose_cam, K, axis_len=axis_len)
    img = color_rgb.copy()
    if pts is None:
        return img
    origin, x_pt, y_pt, z_pt = pts
    cv2.line(img, origin, x_pt, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, origin, y_pt, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, origin, z_pt, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def _write_overlay(path: pathlib.Path, frame: Frame, pose_cam: np.ndarray, K: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(_draw_pose_axes(frame.rgb, pose_cam, K), cv2.COLOR_RGB2BGR))


def _check_frame(frame: Frame, args: Args, intrinsics) -> None:
    """The frame is from --camera, at the resolution and intrinsics of --camera-model-id."""
    assert frame.camera == args.camera, f"frame is from camera {frame.camera!r}, expected {args.camera!r}"
    assert frame.rgb.shape[:2] == (intrinsics.height, intrinsics.width), (
        f"{args.camera} frame is {frame.rgb.shape[:2]} but {args.camera_model_id} intrinsics are "
        f"{intrinsics.height}x{intrinsics.width}; check the ZED grab_resolution"
    )
    K = intrinsics.intrinsic_matrix.astype(np.float64)
    assert np.allclose(frame.K, K, rtol=0.0, atol=K_ATOL_PX), (
        f"camera_info K from the frame server\n{frame.K}\n!= {args.camera_model_id} intrinsics\n{K}\n"
        f"The extrinsics were calibrated with the {args.camera_model_id} intrinsics."
    )


def _pose_message(args: Args, registration_id: int, stamp_ns: int, T_base: np.ndarray, T_cam: np.ndarray) -> bytes:
    return json.dumps(
        {
            "format": POSE_FORMAT,
            "object": args.object_description,
            "camera": args.camera,
            "registration_id": registration_id,
            "stamp_ns": stamp_ns,
            "T_base_obj": [float(v) for v in T_base.reshape(-1)],
            "T_cam_obj": [float(v) for v in T_cam.reshape(-1)],
        }
    ).encode("utf-8")


def main(args: Args) -> None:
    assert len(args.object_description) > 0, "object_description must not be empty"
    assert args.mesh_path.is_file(), f"Mesh not found: {args.mesh_path}"
    assert args.est_refine_iter >= 1, f"est_refine_iter must be >= 1, got {args.est_refine_iter}"
    assert args.track_refine_iter >= 1, f"track_refine_iter must be >= 1, got {args.track_refine_iter}"
    assert args.use_2d_tracker or not args.use_kalman_filter, (
        "use_kalman_filter requires use_2d_tracker: the filter fuses the 2D tracker's "
        "image-plane measurement with FoundationPose's pose estimate each frame."
    )
    T_world_cam = _load_camera_extrinsics_yaml(args.extrinsics_path, args.camera, args.camera_model_id)
    intrinsics = get_color_intrinsics(args.camera_model_id)
    K = intrinsics.intrinsic_matrix.astype(np.float64)
    # The server opens the mesh itself; run_container.sh mounts this repo at its host path, so the absolute host path
    # resolves inside the container.
    mesh_path_abs = str(args.mesh_path.resolve())
    object_slug = args.object_description.replace(" ", "_")
    run_dir = args.debug_dir / time.strftime("%Y-%m-%d__%H-%M-%S")
    run_dir.mkdir(parents=True)
    print(f"[info] Debug images in {run_dir}")

    print("[info] Loading SAM 3 ...")
    predictor = SAM3Predictor()
    print(f"[info] Connecting to FoundationPose server at {args.server_address} ...")
    client = FoundationPoseClient(args.server_address)

    context = zmq.Context()
    frames = subscribe_newest_frames(context, args.frame_address)
    poses = context.socket(zmq.PUB)
    poses.setsockopt(zmq.SNDHWM, 1)
    poses.bind(args.pose_address)
    commands = context.socket(zmq.REP)
    commands.bind(args.command_address)
    poller = zmq.Poller()
    poller.register(frames, zmq.POLLIN)
    poller.register(commands, zmq.POLLIN)
    print(f"[info] Frames from {args.frame_address}, poses on {args.pose_address}, commands on {args.command_address}")
    print("[info] Waiting for a register request ...")

    registration_id = 0
    last_stamp_ns = None
    last_frame, last_pose_cam = None, None
    n_tracked, t_status = 0, time.monotonic()
    while True:
        events = dict(poller.poll(POLL_TIMEOUT_MS))
        if commands in events:
            request = json.loads(commands.recv().decode("utf-8"))
            assert request == {"cmd": "register"}, f"unknown command {request!r}"
            t_request_ns = time.time_ns()
            frame = decode_frame(frames.recv())
            while frame.stamp_ns < t_request_ns:
                frame = decode_frame(frames.recv())
            _check_frame(frame, args, intrinsics)
            depth_m = depth_mm_to_meters(frame.depth_mm)
            image_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            mask = ImageUtils.get_sam_mask(predictor, image_bgr, args.object_description)
            assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
            assert mask.sum() > 0, f"Segmentation mask is empty for description '{args.object_description}'"
            registration_id += 1
            registration_dir = run_dir / f"registration_{registration_id:03d}"
            registration_dir.mkdir()
            ImageUtils.save_masked_debug(image_bgr, mask, registration_dir, object_slug)
            pose_cam = client.register(
                mesh_path=mesh_path_abs,
                color_rgb=frame.rgb,
                depth_m=depth_m,
                mask=mask,
                K=K,
                iteration=args.est_refine_iter,
                use_2d_tracker=args.use_2d_tracker,
                use_kalman_filter=args.use_kalman_filter,
            )
            T_base = transform_pose_cam_to_world(T_world_cam, pose_cam)
            _write_overlay(registration_dir / f"{object_slug}__registered_axes.png", frame, pose_cam, K)
            poses.send(_pose_message(args, registration_id, frame.stamp_ns, T_base, pose_cam))
            commands.send(
                json.dumps(
                    {
                        "registration_id": registration_id,
                        "stamp_ns": frame.stamp_ns,
                        "mask_pixels": int(mask.sum()),
                        "T_base_obj": [float(v) for v in T_base.reshape(-1)],
                    }
                ).encode("utf-8")
            )
            print(
                f"[info] Registration {registration_id}: mask={int(mask.sum())} px, "
                f"base position (cm) = {np.round(T_base[:3, 3] * 100, 2)}, debug images in {registration_dir}"
            )
            last_stamp_ns, last_frame, last_pose_cam = frame.stamp_ns, frame, pose_cam
            continue
        if frames in events and registration_id > 0:
            frame = decode_frame(frames.recv())
            if frame.stamp_ns <= last_stamp_ns:
                continue
            _check_frame(frame, args, intrinsics)
            pose_cam = client.track(
                color_rgb=frame.rgb,
                depth_m=depth_mm_to_meters(frame.depth_mm),
                K=K,
                iteration=args.track_refine_iter,
            )
            T_base = transform_pose_cam_to_world(T_world_cam, pose_cam)
            poses.send(_pose_message(args, registration_id, frame.stamp_ns, T_base, pose_cam))
            last_stamp_ns, last_frame, last_pose_cam = frame.stamp_ns, frame, pose_cam
            n_tracked += 1
        now = time.monotonic()
        if now - t_status >= STATUS_PERIOD_S:
            if last_frame is None:
                print("[info] Waiting for a register request ...")
            else:
                T_base = transform_pose_cam_to_world(T_world_cam, last_pose_cam)
                age_ms = (time.time_ns() - last_stamp_ns) / 1e6
                print(
                    f"[info] Registration {registration_id}: {n_tracked / (now - t_status):.1f} Hz, newest frame "
                    f"{age_ms:.0f} ms old, base position (cm) = {np.round(T_base[:3, 3] * 100, 2)}"
                )
                _write_overlay(run_dir / "latest_overlay.png", last_frame, last_pose_cam, K)
            n_tracked, t_status = 0, now


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
