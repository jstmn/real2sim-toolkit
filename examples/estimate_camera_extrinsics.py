"""
This script does the following:
1. Loads the demonstration data from the h5 file
2. Loads the robot from Jrl2
3. Runs an optimization procedure to estimate the camera extrinsics.
4. Saves the estimated camera extrinsics to a yaml file.

At a high level, the optimization procedure runs CMA-ES to estimate the SE(3) pose (in the robot's base frame) of the
specified camera. There are two pointclouds that we care about: pcd_real and pcd_sim. pcd_real is the measured
pointcloud from the camera (found by projecting the depth points with the camera's intrinsic matrix). pcd_sim is a
6-view render of the robot (cameras on +X/-X/+Y/-Y/+Z/-Z looking at the workspace), fused in the robot-base frame and
FPS-downsampled once per sampled joint configuration, then mapped into the candidate camera with T_world_cam. The cost
is the one-sided squared nearest-neighbor distance from pcd_real to pcd_sim.

Notes:
1. A preprocessing step is performed on the measured pointclouds to remove points that aren't part of the robot.
    Frame 0 is SAM 3 (union of top --sam-kmax masks). That union is then propagated through every later
    RGB frame with SAM 3 ``mask_input`` + the previous mask's bbox (text prompt only on frame 0).
    After unprojection, Open3D ``remove_radius_outlier`` drops points with fewer than
    ``--radius-outlier-nb-points`` neighbors in ``--radius-outlier-radius-m`` (default 10 in 10 cm).
2. If --visualize is set, a viser server is started. The server shows measured vs best-so-far
    simulated pointclouds in the robot-base frame, a camera frustum at the estimated pose, and a
    plot of lowest population cost vs CMA-ES iteration.
3. if --visualize-robot-masks is set, debug PNGs are written for the frame-0 SAM masks that enter the union.
4. If --cache-robot-masks is set (default), each propagated mask is written as soon as it is computed to
   `<h5_dir>/<h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy`.
    A later run loads a consecutive prefix of those files and resumes SAM from the last cached frame.
    Frame 0's SAM 3 union may also be read from the older
    `robot-mask__<camera>__idx=0__...npy` seed cache. After the last frame, a dimmed-mask video is written to
    `robot-mask-propagated__<camera>__kmax=<k>__score_threshold=<t>.mp4`.
    With --per-frame-robot-masks, masks are cached per frame as
    `robot-mask-per-frame__<camera>__prompt=<p>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy` and only uncached
    frames are segmented.


The pseudo-code is as follows:

inputs:
- rgbds: array of RGBD images
- all_joint_angles: array of joint angles from the demonstration. Same length as rgbds; frame i of qpos is frame i of RGB-D.
- N: the number of timesteps to sample from the demonstration for the cost function
- rgb_to_pcd(rgbd): unprojects color-aligned depth with color K, then keeps SAM robot pixels
- S: number of poses in the CMA-ES population
- pcd_sim_world[t]: 6-view rendered robot cloud at joint_angles[t], FPS-downsampled, in the robot-base frame
- camera_from_world(pose, pcd_world): maps world points into the OpenCV camera frame
- compute_chamfer_distance(pcd_real, pcd_sim): one-sided mean min squared distance from pcd_real to pcd_sim

# Find the N joint angles that are the furthest apart from each other
timesteps, joint_angles = furthest_point_sample(all_joint_angles, N, distance='circular')

pcd_reals = [rgb_to_pcd(rgbd) for rgbd in rgbds]

until convergence:
    sample poses pi, ..., pN from CMA-ES
    total_costs = [0] * S
    for each pose pi in population:
        total_cost = 0
        for joint_angle, pcd_real, pcd_sim_world in zip(joint_angles, pcd_reals, pcd_sim_worlds):
            pcd_sim = camera_from_world(pose, pcd_sim_world)
            cost = compute_chamfer_distance(pcd_real, pcd_sim)
            total_cost += cost
        total_costs[i] = total_cost
    update CMA-ES with total_costs as fitness values

output:
- estimated pose
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Literal

import cv2
import numpy as np
import tyro
import yaml
from jrl2.robots import NAME_TO_ROBOT, get_robot_by_name
from transforms3d.quaternions import mat2quat

from r2st.constants import MIN_DEPTH_M, get_color_intrinsics
from r2st.extrinsics import (
    SEED_LOOKAT_TARGET,
    SEED_MIN_LOOK_DIST_M,
    SEED_UP,
    ExtrinsicsVisualizerBase,
    SapienRobot,
    T_to_sapien_pose,
    build_real_pcd,
    compute_robot_masks,
    depth_reprojection_for,
    erode_mask,
    expand_demo_qpos_to_joint_names,
    frame_depth_m,
    frames_with_robot_mask,
    index_ranges_text,
    load_demo,
    log_elapsed,
    look_at_opencv,
    mask_valid_depth_fraction,
    optimize_extrinsics,
    transform_points,
    validate_seed_mode,
    world_points_to_camera,
)
from r2st.geometry import one_sided_squared_nn_distance
from r2st.utils import (
    ImageUtils,
    farthest_point_sample_naive,
    farthest_point_sample_pyg_lib,
)

"""
# See README.md for example usage
"""

_CAPTURE_AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_CAPTURE_DISTANCE_M = 1.0
_CAPTURE_FOV_DEG = 90.0
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 480
_CAPTURE_FAR_M = 5.0
_CAPTURE_CAMERA_NAME = "multiview_cam"
_CAPTURE_UP_WHEN_PARALLEL = np.array([1.0, 0.0, 0.0], dtype=np.float64)


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see data/grail_xarm_july_2/merge_camera_streams.py)."""

    robot_id: str
    """Jrl2 robot name, e.g. 'xarm7__gripper'. See the README `{robot}__{eef}` table."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    camera_model_id: str
    """Camera model id for calibration lookup, e.g. 'd435'. See the README camera-model table."""

    depth_intrinsics_source: Literal["rgb", "depth"]
    """Which camera-model K the h5 depth channel uses. `rgb`: color-aligned depth, unproject with color K. `depth`: native depth image, reproject into the color frame first."""

    output_path: pathlib.Path
    """YAML path to write the estimated 4x4 camera extrinsics (robot-base T camera)."""

    n_timesteps: int = 5
    """Number of furthest-apart joint configurations used in the chamfer cost."""

    ignore_timesteps: tuple[int, ...] = ()
    """Demo frame indices to never sample, e.g. `--ignore-timesteps 12 40 41`."""

    n_pcd_samples_real: int = 2048
    """Number of FPS points kept from each measured robot cloud."""

    n_pcd_samples_sim: int = 2048
    """Number of FPS points kept from each fused 6-view sim robot cloud."""

    n_random_downsample_initial: int = 16384
    """Random pre-FPS cap for each measured robot cloud. Must be >= n_pcd_samples_real."""

    cma_sigma_pos: float = 0.10
    """CMA-ES initial std for camera translation (meters)."""

    cma_sigma_rot: float = 1.0
    """CMA-ES initial std for camera rotation-vector (radians)."""

    cma_maxiter: int = 250
    """Maximum CMA-ES generations."""

    cma_early_stop_atol: float = 0.00001
    """Stop CMA-ES if this generation's min cost (`gen_min`) changes by less than this for `cma_early_stop_patience` gens."""

    cma_early_stop_patience: int = 5
    """Consecutive generations whose `gen_min` range is < `cma_early_stop_atol` before early stop."""

    seed_automatically: bool = False
    """If set, choose the CMA-ES seed with the spherical-grid search."""

    seed_from_gui: bool = False
    """If set, choose the CMA-ES seed interactively with Viser GUI controls."""

    seed_pose: tuple[float, float, float, float, float, float, float] | None = None
    """If set, seed CMA-ES with this robot-base camera pose: x y z qw qx qy qz."""

    gui_translation_step_m: float = 0.02
    """World-frame translation per GUI button press, in meters."""

    gui_rotation_step_deg: float = 5.0
    """World-frame rotation per GUI button press, in degrees."""

    n_seed_azimuth: int = 25
    """Number of azimuth steps (about world +Z) for seed camera positions on each sphere."""

    n_seed_polar: int = 25
    """Number of polar-angle steps (from world +Z) for seed camera positions on each sphere."""

    n_seed_radii: int = 3
    """Number of sphere radii, linspace from 1.0 m to 1.5 m inclusive."""

    n_seed_rolls: int = 10
    """Number of evenly spaced rolls about the look-at axis (toward (0, 0, 0.5)) per seed position."""

    cma_popsize: int = 25
    """CMA-ES population size."""

    robot_description: str = "robot arm"
    """SAM 3 text prompt used to mask the robot: on frame 0 only (later frames propagate that mask), or on every frame
    with --per-frame-robot-masks."""

    sam_kmax: int = 5
    """Union at most this many highest-confidence SAM masks into the frame-0 robot mask."""

    sam_score_threshold: float = 0.3
    """Keep a top-k SAM mask in the frame-0 union only if its confidence is strictly above this."""

    per_frame_robot_masks: bool = False
    """If set, segment the robot with the SAM 3 text prompt on every frame (union of the top --sam-kmax masks) instead
    of propagating frame 0's mask. For recordings of separate arm poses (not a continuous trajectory), where the arm
    jumps between frames and propagation stays on the static base."""

    cache_robot_masks: bool = True
    """If set, load/save robot masks as <h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy,
    or <h5_stem>/robot-mask-per-frame__<camera>__prompt=<p>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy with
    --per-frame-robot-masks."""

    mask_erode_px: int = 9
    """Erode the robot mask with an elliptical kernel of this size (pixels) before unprojecting. 0 skips erosion."""

    min_mask_valid_depth_frac: float = 0.5
    """Only sample frames where at least this fraction of the eroded robot mask has valid depth. When the arm is too
    close to the camera, most of its mask has no depth and the few valid pixels are wrong."""

    max_point_distance_m: float = 2.0
    """Drop measured robot points farther than this (meters, Euclidean) from the camera center."""

    radius_outlier_nb_points: int = 10
    """Keep a measured point only if this many points lie within radius_outlier_radius_m of it."""

    radius_outlier_radius_m: float = 0.10
    """Search radius (meters) for the measured-cloud radius-outlier filter."""

    visualize: bool = False
    """If set, start a viser server with robot-base pointclouds, camera frustum, and a cost plot."""

    visualize_robot_masks: bool = False
    """If set, write debug PNGs for the frame-0 SAM masks that enter the union (top --sam-kmax with score above --sam-score-threshold)."""


def _save_sampled_frame_inputs(
    h5_path: pathlib.Path,
    camera: str,
    sample_idx: int,
    frame_idx: int,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
) -> pathlib.Path:
    """Write the RGB, depth (meters, .npy + colormap .png), and eroded mask overlay that a sampled frame unprojects.

    Returns the output directory.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
    assert depth_m.shape == rgb.shape[:2], f"depth {depth_m.shape} != rgb {rgb.shape[:2]}"
    assert mask.shape == rgb.shape[:2], f"mask {mask.shape} != rgb {rgb.shape[:2]}"
    out_dir = h5_path.parent / h5_path.stem / "sampled_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{camera}__sample={sample_idx}__idx={frame_idx}"
    rgb_path = prefix.with_name(prefix.name + "__rgb.png")
    depth_npy_path = prefix.with_name(prefix.name + "__depth_m.npy")
    depth_png_path = prefix.with_name(prefix.name + "__depth.png")
    overlay_path = prefix.with_name(prefix.name + "__mask_overlay.png")
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(rgb_path), image_bgr), f"Failed to write {rgb_path}"
    np.save(depth_npy_path, depth_m)
    finite = np.isfinite(depth_m) & (depth_m > 0)
    assert finite.any(), f"frame {frame_idx}: depth has no finite positive values"
    depth_max = float(np.percentile(depth_m[finite], 99))
    depth_u8 = np.zeros(depth_m.shape, dtype=np.uint8)
    depth_u8[finite] = (255.0 * np.clip(depth_m[finite] / depth_max, 0.0, 1.0)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~finite] = 0
    assert cv2.imwrite(str(depth_png_path), depth_color), f"Failed to write {depth_png_path}"
    assert cv2.imwrite(str(overlay_path), ImageUtils.demo_overlay(image_bgr, mask)), f"Failed to write {overlay_path}"
    return out_dir


def _axis_aligned_capture_poses(lookat: np.ndarray, distance_m: float) -> list[np.ndarray]:
    """Six OpenCV T_world_cam poses on ±X/±Y/±Z looking at `lookat` from `distance_m`."""
    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)
    assert distance_m > 0.0, f"distance_m must be > 0, got {distance_m}"
    assert _CAPTURE_AXES.shape == (6, 3), f"_CAPTURE_AXES must be (6, 3), got {_CAPTURE_AXES.shape}"
    poses: list[np.ndarray] = []
    up_default = SEED_UP / np.linalg.norm(SEED_UP)
    for axis in _CAPTURE_AXES:
        axis_norm = float(np.linalg.norm(axis))
        assert axis_norm > 0.0, f"capture axis must be non-zero, got {axis}"
        axis_u = axis / axis_norm
        eye = lookat + float(distance_m) * axis_u
        look = lookat - eye
        look_n = float(np.linalg.norm(look))
        assert look_n >= SEED_MIN_LOOK_DIST_M, f"capture look-at distance {look_n} < {SEED_MIN_LOOK_DIST_M}"
        look_u = look / look_n
        up = up_default
        if abs(float(look_u @ up)) > (1.0 - 1e-6):
            up = _CAPTURE_UP_WHEN_PARALLEL
        poses.append(look_at_opencv(eye, lookat, up))
    assert len(poses) == 6, f"expected 6 capture poses, got {len(poses)}"
    return poses


class SimRobotRenderer(SapienRobot):
    """Sapien robot with one capture camera. 6-view depth renders are fused in the robot-base frame."""

    def __init__(self, urdf_path: pathlib.Path) -> None:
        super().__init__(urdf_path)
        fovy = float(np.deg2rad(_CAPTURE_FOV_DEG))
        assert fovy > 0.0, f"capture fovy must be > 0, got {fovy}"
        self._cam = self._scene.add_camera(
            _CAPTURE_CAMERA_NAME,
            _CAPTURE_WIDTH,
            _CAPTURE_HEIGHT,
            fovy=fovy,
            near=MIN_DEPTH_M,
            far=_CAPTURE_FAR_M,
        )
        self._capture_Ts = _axis_aligned_capture_poses(SEED_LOOKAT_TARGET, _CAPTURE_DISTANCE_M)
        print(
            f"[info] Multiview capture: 6 axis cameras at {_CAPTURE_DISTANCE_M:g} m, "
            f"{_CAPTURE_WIDTH}x{_CAPTURE_HEIGHT} fov={_CAPTURE_FOV_DEG:g} deg far={_CAPTURE_FAR_M:g} m"
        )

    def _render_camera_frame(self, T_world_cam: np.ndarray) -> np.ndarray:
        """Robot-only cloud in the OpenCV camera frame of `T_world_cam`, `(M, 3)` float32."""
        self._cam.local_pose = T_to_sapien_pose(T_world_cam)
        self._scene.update_render()
        self._cam.take_picture()
        position = self._cam.get_picture("Position")
        segmentation = self._cam.get_picture("Segmentation")
        assert position.ndim == 3 and position.shape[2] >= 3, f"Bad Position texture: {position.shape}"
        xyz_gl = position[:, :, :3].astype(np.float64)
        if segmentation.ndim == 3:
            seg = segmentation[:, :, 0]
        else:
            seg = segmentation
        robot_px = seg > 0
        xyz_cv = np.stack([xyz_gl[:, :, 0], -xyz_gl[:, :, 1], -xyz_gl[:, :, 2]], axis=-1)
        z = xyz_cv[:, :, 2]
        valid = robot_px & np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= _CAPTURE_FAR_M)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float32)
        return xyz_cv[valid].astype(np.float32)

    def render_multiview_world(self, qpos: np.ndarray, n_samples: int, device: str) -> np.ndarray:
        """Fuse 6 axis-aligned renders in the robot-base frame and FPS to `n_samples`."""
        assert n_samples >= 1, f"n_samples must be >= 1, got {n_samples}"
        assert isinstance(device, str) and len(device) > 0, f"device must be a non-empty str, got {device!r}"
        self.set_qpos(qpos)
        parts: list[np.ndarray] = []
        n_per_view: list[int] = []
        for T in self._capture_Ts:
            pts_cam = self._render_camera_frame(T)
            n_per_view.append(int(pts_cam.shape[0]))
            if pts_cam.shape[0] == 0:
                continue
            parts.append(transform_points(T, pts_cam))
        assert len(parts) >= 1, f"All 6 capture views were empty, counts={n_per_view}"
        p_world = np.concatenate(parts, axis=0)
        assert p_world.ndim == 2 and p_world.shape[1] == 3 and p_world.shape[0] >= 1
        n_fused = int(p_world.shape[0])
        if n_fused > n_samples:
            idx = farthest_point_sample_pyg_lib(p_world, n_samples, device=device)
            p_world = p_world[idx]
        print(f"[info]   views[+x,-x,+y,-y,+z,-z]={n_per_view} fused={n_fused} fps={p_world.shape[0]}")
        return p_world.astype(np.float32, copy=False)


class ChamferPoseCost:
    """Sum over the sampled timesteps of the one-sided squared NN distance from each measured cloud to the sim cloud
    seen from a candidate `T_world_cam`."""

    def __init__(self, pcd_reals: list[np.ndarray], pcd_sim_worlds: list[np.ndarray], device: str) -> None:
        assert len(pcd_reals) >= 1, "pcd_reals must be non-empty"
        assert len(pcd_reals) == len(
            pcd_sim_worlds
        ), f"real clouds {len(pcd_reals)} != sim worlds {len(pcd_sim_worlds)}"
        for i, p_world in enumerate(pcd_sim_worlds):
            assert (
                p_world.ndim == 2 and p_world.shape[1] == 3 and p_world.shape[0] >= 1
            ), f"pcd_sim_worlds[{i}] must be (N, 3) with N>=1, got {p_world.shape}"
        self._pcd_reals = pcd_reals
        self._pcd_sim_worlds = pcd_sim_worlds
        self._device = device

    def __call__(self, T_world_cam: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Return (cost, per-timestep sim clouds in the camera frame)."""
        total = 0.0
        pcd_sims: list[np.ndarray] = []
        for pcd_real, p_world in zip(self._pcd_reals, self._pcd_sim_worlds, strict=True):
            pcd_sim = world_points_to_camera(T_world_cam, p_world)
            pcd_sims.append(pcd_sim)
            total += float(one_sided_squared_nn_distance(pcd_real, pcd_sim, device=self._device))
        return total, pcd_sims


class ExtrinsicsVisualizer(ExtrinsicsVisualizerBase):
    """Adds the sim cloud at the shown pose (green) and the masked RGB frame on the camera frustum."""

    def __init__(
        self,
        demo_frames: np.ndarray,
        K: np.ndarray,
        height: int,
        width: int,
        rgb_frames: np.ndarray,
        pcd_reals: list[np.ndarray],
        cma_maxiter: int,
        fps_idx: np.ndarray,
        renderer: SimRobotRenderer,
        joint_angles: np.ndarray,
        cost_fn: ChamferPoseCost,
    ) -> None:
        assert demo_frames.shape == rgb_frames.shape, f"demo_frames {demo_frames.shape} != rgb {rgb_frames.shape}"
        self._demo_frames = demo_frames
        super().__init__(
            K, height, width, rgb_frames, pcd_reals, cma_maxiter, fps_idx, renderer, joint_angles, cost_fn, ""
        )

    def _add_method_gui(self) -> None:
        self._sim_handle = self._server.scene.add_point_cloud(
            "/pcd_sim",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=(40, 200, 120),
            point_size=0.003,
            point_shape="circle",
        )

    def _draw_timestep(self, t: int, sim: np.ndarray | None) -> None:
        self._sim_handle.visible = sim is not None and sim.shape[0] > 0
        if self._sim_handle.visible:
            self._sim_handle.points = np.ascontiguousarray(transform_points(self._T, sim))
        self._frustum.image = self._demo_frames[t]


def main(args: Args) -> None:
    import torch

    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert (
        args.robot_id.lower() in NAME_TO_ROBOT
    ), f"Unknown robot_id {args.robot_id!r}. Available: {sorted(NAME_TO_ROBOT)}"
    assert args.n_timesteps >= 1, f"n_timesteps must be >= 1, got {args.n_timesteps}"
    assert args.n_pcd_samples_real >= 1, f"n_pcd_samples_real must be >= 1, got {args.n_pcd_samples_real}"
    assert args.n_pcd_samples_sim >= 1, f"n_pcd_samples_sim must be >= 1, got {args.n_pcd_samples_sim}"
    assert args.n_random_downsample_initial >= args.n_pcd_samples_real, (
        f"n_random_downsample_initial ({args.n_random_downsample_initial}) must be >= "
        f"n_pcd_samples_real ({args.n_pcd_samples_real})"
    )
    assert args.cma_sigma_pos > 0, f"cma_sigma_pos must be > 0, got {args.cma_sigma_pos}"
    assert args.cma_sigma_rot > 0, f"cma_sigma_rot must be > 0, got {args.cma_sigma_rot}"
    assert args.cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {args.cma_maxiter}"
    assert args.cma_popsize is None or args.cma_popsize >= 2, f"cma_popsize must be >= 2, got {args.cma_popsize}"
    assert args.cma_early_stop_atol > 0, f"cma_early_stop_atol must be > 0, got {args.cma_early_stop_atol}"
    assert (
        args.cma_early_stop_patience >= 1
    ), f"cma_early_stop_patience must be >= 1, got {args.cma_early_stop_patience}"
    validate_seed_mode(args.seed_automatically, args.seed_from_gui, args.seed_pose)
    assert args.gui_translation_step_m > 0.0, f"gui_translation_step_m must be > 0, got {args.gui_translation_step_m}"
    assert args.gui_rotation_step_deg > 0.0, f"gui_rotation_step_deg must be > 0, got {args.gui_rotation_step_deg}"
    assert args.n_seed_azimuth >= 1, f"n_seed_azimuth must be >= 1, got {args.n_seed_azimuth}"
    assert args.n_seed_polar >= 1, f"n_seed_polar must be >= 1, got {args.n_seed_polar}"
    assert args.n_seed_rolls >= 1, f"n_seed_rolls must be >= 1, got {args.n_seed_rolls}"
    assert args.n_seed_radii >= 1, f"n_seed_radii must be >= 1, got {args.n_seed_radii}"
    assert len(args.robot_description) > 0, "robot_description must not be empty"
    assert args.sam_kmax >= 1, f"sam_kmax must be >= 1, got {args.sam_kmax}"
    assert (
        0.0 <= args.sam_score_threshold < 1.0
    ), f"sam_score_threshold must be in [0, 1), got {args.sam_score_threshold}"
    assert args.mask_erode_px == 0 or (
        args.mask_erode_px >= 1 and args.mask_erode_px % 2 == 1
    ), f"mask_erode_px must be 0 or a positive odd int, got {args.mask_erode_px}"
    assert (
        0.0 < args.min_mask_valid_depth_frac <= 1.0
    ), f"min_mask_valid_depth_frac must be in (0, 1], got {args.min_mask_valid_depth_frac}"
    assert (
        args.max_point_distance_m > MIN_DEPTH_M
    ), f"max_point_distance_m must be > {MIN_DEPTH_M}, got {args.max_point_distance_m}"
    assert (
        args.radius_outlier_nb_points >= 1
    ), f"radius_outlier_nb_points must be >= 1, got {args.radius_outlier_nb_points}"
    assert (
        args.radius_outlier_radius_m > 0.0
    ), f"radius_outlier_radius_m must be > 0, got {args.radius_outlier_radius_m}"
    chamfer_device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    color_intrinsics = get_color_intrinsics(args.camera_model_id)

    print(f"[info] Loading RGB-D + qpos from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb_all, depth_raw_all, qpos_all = load_demo(args.h5_path, args.camera)
    log_elapsed(f"Loaded rgb{rgb_all.shape} depth{depth_raw_all.shape} qpos{qpos_all.shape}", t0)

    n_frames = rgb_all.shape[0]
    assert args.n_timesteps <= n_frames, f"n_timesteps={args.n_timesteps} > n_frames={n_frames}"
    ignore_timesteps = np.asarray(args.ignore_timesteps, dtype=np.int64)
    assert (
        np.unique(ignore_timesteps).size == ignore_timesteps.size
    ), f"Duplicate ignore_timesteps: {args.ignore_timesteps}"
    assert np.all(
        (ignore_timesteps >= 0) & (ignore_timesteps < n_frames)
    ), f"ignore_timesteps {args.ignore_timesteps} must be in [0, {n_frames})"
    H, W = int(rgb_all.shape[1]), int(rgb_all.shape[2])
    assert (H, W) == (color_intrinsics.height, color_intrinsics.width), (
        f"{args.camera} rgb is {H}x{W} but {args.camera_model_id} color intrinsics are "
        f"{color_intrinsics.height}x{color_intrinsics.width}"
    )
    K = color_intrinsics.intrinsic_matrix.astype(np.float64)
    print(f"[info] Color K ({args.camera_model_id}):\n{K}")
    depth_reprojection = depth_reprojection_for(
        args.camera, args.camera_model_id, args.depth_intrinsics_source, depth_raw_all.shape[1:]
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    t0_masks = time.perf_counter()
    masks_all = compute_robot_masks(
        rgb_all,
        args.h5_path,
        args.camera,
        args.robot_description,
        args.sam_kmax,
        args.sam_score_threshold,
        args.per_frame_robot_masks,
        args.cache_robot_masks,
        args.visualize_robot_masks,
    )
    has_mask, no_mask = frames_with_robot_mask(masks_all, args.per_frame_robot_masks)
    log_elapsed(f"Built robot masks for {n_frames} frames", t0_masks)

    # When the arm is too close to the camera, most of its mask has no depth and the few valid pixels are wrong
    t0 = time.perf_counter()
    valid_depth_frac = np.full(n_frames, np.nan)
    for t in has_mask:
        valid_depth_frac[t] = mask_valid_depth_fraction(
            erode_mask(masks_all[t], args.mask_erode_px),
            frame_depth_m(depth_raw_all[t], color_intrinsics, depth_reprojection),
            args.max_point_distance_m,
        )
    candidates = has_mask[valid_depth_frac[has_mask] >= args.min_mask_valid_depth_frac]
    excluded = has_mask[valid_depth_frac[has_mask] < args.min_mask_valid_depth_frac]
    log_elapsed(
        f"Excluded {excluded.size}/{has_mask.size} masked frames with < {args.min_mask_valid_depth_frac:.0%} of the "
        f"eroded robot mask on valid depth: {index_ranges_text(excluded)}",
        t0,
    )
    candidates = np.setdiff1d(candidates, ignore_timesteps)
    print(
        f"[info] Ignoring {ignore_timesteps.size} frames from --ignore-timesteps: {index_ranges_text(np.sort(ignore_timesteps))}"
    )
    assert candidates.size >= args.n_timesteps, (
        f"Only {candidates.size} frames have a robot mask with >= {args.min_mask_valid_depth_frac:.0%} on valid depth "
        f"and are not in ignore_timesteps, need n_timesteps={args.n_timesteps}"
    )
    fps_local = farthest_point_sample_naive(
        qpos_all[candidates].astype(np.float64), args.n_timesteps, distance="circular"
    )
    fps_idx = candidates[np.asarray(fps_local, dtype=np.int64)]
    print(
        f"[info] FPS circular joint samples (over {candidates.size} usable frames) at RGB-D frames {fps_idx.tolist()}"
    )

    rgb_sel = rgb_all[fps_idx]
    qpos_sel = qpos_all[fps_idx]
    depth_m_sel = np.stack(
        [frame_depth_m(depth_raw_all[t], color_intrinsics, depth_reprojection) for t in fps_idx], axis=0
    )
    assert depth_m_sel.shape == (
        args.n_timesteps,
        H,
        W,
    ), f"depth_m_sel shape {depth_m_sel.shape} != ({args.n_timesteps}, {H}, {W})"
    masks_sel = masks_all[fps_idx]
    pcd_reals: list[np.ndarray] = []
    demo_frames: list[np.ndarray] = []
    for i, frame_idx in enumerate(fps_idx):
        frame_idx = int(frame_idx)
        mask = masks_sel[i]
        assert mask.any(), f"Empty robot mask at frame {frame_idx} for prompt {args.robot_description!r}"
        n_mask = int(mask.sum())
        mask = erode_mask(mask, args.mask_erode_px)
        assert valid_depth_frac[frame_idx] >= args.min_mask_valid_depth_frac, (
            f"frame {frame_idx}: {valid_depth_frac[frame_idx]:.2f} of the mask on valid depth "
            f"< {args.min_mask_valid_depth_frac}"
        )
        demo_frames.append(ImageUtils.demo_overlay(rgb_sel[i], mask))
        inputs_dir = _save_sampled_frame_inputs(
            args.h5_path, args.camera, i, frame_idx, rgb_sel[i], depth_m_sel[i], mask
        )
        pts, (n_depth, n_unproj, n_inliers) = build_real_pcd(
            depth_m_sel[i],
            mask,
            K,
            args.max_point_distance_m,
            args.radius_outlier_nb_points,
            args.radius_outlier_radius_m,
            args.n_pcd_samples_real,
            args.n_random_downsample_initial,
            chamfer_device,
            rng,
        )
        assert pts.shape[0] >= 1, (
            f"No robot points at frame {frame_idx}: {n_depth} unprojected -> {n_unproj} within "
            f"{args.max_point_distance_m} m -> {n_inliers} radius inliers"
        )
        pcd_reals.append(pts)
        print(
            f"[info] pcd_real[{i}] frame {frame_idx}: mask {n_mask} -> {int(mask.sum())} px eroded "
            f"({valid_depth_frac[frame_idx]:.0%} valid depth) | points {n_depth} -> {n_unproj} within "
            f"{args.max_point_distance_m} m -> {n_inliers} radius inliers -> {pts.shape[0]} sampled"
        )
    print(f"[info] Saved sampled-frame RGB / depth / mask images to {inputs_dir}")
    assert len(demo_frames) == args.n_timesteps, f"demo_frames {len(demo_frames)} != n_timesteps {args.n_timesteps}"
    demo_sel = np.stack(demo_frames, axis=0)

    print(f"[info] Loading Jrl2 robot '{args.robot_id}' ...")
    t0 = time.perf_counter()
    robot = get_robot_by_name(args.robot_id)
    urdf_path = pathlib.Path(robot._urdf_filepath)
    print(f"[info] URDF: {urdf_path}")
    renderer = SimRobotRenderer(urdf_path)
    qpos_sel = expand_demo_qpos_to_joint_names(qpos_sel, robot, renderer.joint_names)
    pcd_sim_worlds: list[np.ndarray] = []
    t0_sim = time.perf_counter()
    for i, q in enumerate(qpos_sel):
        p_world = renderer.render_multiview_world(q, args.n_pcd_samples_sim, chamfer_device)
        pcd_sim_worlds.append(p_world)
        print(f"[info] pcd_sim_world[{i}] n={p_world.shape[0]}")
    log_elapsed(f"Rendered 6-view sim clouds for {len(pcd_sim_worlds)} timesteps", t0_sim)
    log_elapsed("Loaded robot + Sapien renderer", t0)

    cost_fn = ChamferPoseCost(pcd_reals, pcd_sim_worlds, chamfer_device)
    vis = (
        ExtrinsicsVisualizer(
            demo_sel, K, H, W, rgb_sel, pcd_reals, args.cma_maxiter, fps_idx, renderer, qpos_sel, cost_fn
        )
        if args.visualize or args.seed_from_gui
        else None
    )
    best_T, best_cost, _ = optimize_extrinsics(
        cost_fn,
        args.seed_automatically,
        args.seed_from_gui,
        args.seed_pose,
        args.gui_translation_step_m,
        args.gui_rotation_step_deg,
        args.n_seed_azimuth,
        args.n_seed_polar,
        args.n_seed_rolls,
        args.n_seed_radii,
        args.cma_sigma_pos,
        args.cma_sigma_rot,
        args.cma_maxiter,
        args.cma_popsize,
        args.cma_early_stop_atol,
        args.cma_early_stop_patience,
        vis,
    )
    t, q_wxyz = best_T[:3, 3], mat2quat(best_T[:3, :3])
    payload = {
        "camera": args.camera,
        "robot_id": args.robot_id,
        "camera_model_id": args.camera_model_id,
        "depth_intrinsics_source": args.depth_intrinsics_source,
        "parent_frame": "robot_base",
        "child_frame": f"{args.camera}_optical",
        "h5_path": str(args.h5_path),
        "n_timesteps": args.n_timesteps,
        "ignore_timesteps": list(args.ignore_timesteps),
        "frames_without_robot_mask": no_mask.tolist(),
        "sampled_frame_indices": fps_idx.tolist(),
        "cost": float(best_cost),
        "translation": t.tolist(),
        "quaternion_wxyz": q_wxyz.tolist(),
        "matrix": best_T.tolist(),
    }
    args.output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"✅ Estimated camera extrinsics written to {args.output_path}")
    print(f"[info] cost={best_cost:.6f}")
    print(f"[info] translation={t}")
    print(f"[info] quaternion_wxyz={q_wxyz}")

    if args.visualize:
        assert vis is not None
        vis.wait()
    elif vis is not None:
        vis.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
