"""
This script does the following:
1. Loads the demonstration data from the h5 file
2. Loads the robot from Jrl2
3. Runs an optimization procedure to estimate the camera extrinsics.
4. Saves the estimated camera extrinsics to a yaml file.

At a high level, the optimization procedure runs CMA-ES to estimate the SE(3) pose (in the robot's base frame) of the
specified camera by comparing robot silhouettes. There are two masks that we care about: sam_mask and sim_mask.
sam_mask is the SAM 3 robot mask on the real RGB frame. sim_mask is the ground-truth segmentation of the simulated robot,
rendered by Sapien from the candidate camera pose with the real color intrinsics at that frame's joint configuration.

A SAM pixel that is not in sim_mask is an error. A sim pixel that is not in sam_mask is not necessarily an error,
because SAM 3 can return partial masks. The per-frame cost is therefore mostly one-sided:

    cost = mean_{p in sam_mask} dist(p, sim_mask) + sim_outside_sam_weight * mean_{p in sim_mask} dist(p, sam_mask)

where dist(p, mask) is the pixel distance from p to the nearest pixel of mask (0 inside it). The distance transform
(rather than a pixel count) keeps the cost informative when the silhouettes do not overlap. The second term is small
but must be > 0 in practice: with the first term alone, any pose whose sim_mask covers sam_mask is optimal, so CMA-ES
drifts toward cameras pressed against the robot whose sim_mask fills the image.

Notes:
1. A preprocessing step builds the robot masks.
    Frame 0 is SAM 3 (union of top --sam-kmax masks). That union is then propagated through every later
    RGB frame with SAM 3 ``mask_input`` + the previous mask's bbox (text prompt only on frame 0).
    The SAM mask of each sampled frame is eroded by --mask-erode-px so that SAM boundary bleed onto the background
    is not counted as an error.
2. If --visualize is set, a viser server is started. The server shows the robot model, a camera frustum at the
    estimated pose with the SAM vs sim mask overlay, and a plot of lowest population cost vs CMA-ES iteration.
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
5. After optimization, a SAM vs sim overlay at the estimated pose is written for every sampled frame to
    `<h5_dir>/<h5_stem>/sampled_inputs/`.


The pseudo-code is as follows:

inputs:
- rgbs: array of RGB images
- all_joint_angles: array of joint angles from the demonstration. Same length as rgbs; frame i of qpos is frame i of RGB.
- N: the number of timesteps to sample from the demonstration for the cost function
- S: number of poses in the CMA-ES population
- render_sim_mask(pose, joint_angle): Sapien robot segmentation from `pose` with the real color K
- silhouette_cost(sam_mask, sim_mask): the per-frame cost above

# Find the N joint angles that are the furthest apart from each other
timesteps, joint_angles = furthest_point_sample(all_joint_angles, N, distance='circular')

sam_masks = [erode(sam3(rgb)) for rgb in rgbs[timesteps]]

until convergence:
    sample poses p1, ..., pS from CMA-ES
    total_costs = [0] * S
    for each pose pi in population:
        total_cost = 0
        for joint_angle, sam_mask in zip(joint_angles, sam_masks):
            sim_mask = render_sim_mask(pi, joint_angle)
            total_cost += silhouette_cost(sam_mask, sim_mask)
        total_costs[i] = total_cost / N
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
    ExtrinsicsVisualizerBase,
    SapienRobot,
    T_to_sapien_pose,
    build_real_pcd,
    compute_robot_masks,
    compute_workspace_masks,
    depth_reprojection_for,
    erode_mask,
    expand_demo_qpos_to_joint_names,
    frame_depth_m,
    frames_with_robot_mask,
    index_ranges_text,
    load_demo,
    log_elapsed,
    optimize_extrinsics,
    validate_seed_mode,
)
from r2st.types import CameraIntrinsics
from r2st.utils import ImageUtils, farthest_point_sample_naive

"""
# See README.md for example usage
"""

_RENDER_FAR_M = 5.0
_RENDER_CAMERA_NAME = "mask_cam"
_OVERLAY_BACKGROUND_ALPHA = 0.8
_OVERLAY_MASK_ALPHA = 0.6
_OVERLAY_SAM_ONLY_RGB = np.array([255.0, 40.0, 40.0], dtype=np.float32)
_OVERLAY_SIM_ONLY_RGB = np.array([60.0, 120.0, 255.0], dtype=np.float32)
_OVERLAY_BOTH_RGB = np.array([40.0, 220.0, 90.0], dtype=np.float32)


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
    """Which camera-model K the h5 depth channel uses. `rgb`: color-aligned depth, unproject with color K. `depth`: native depth image, reproject into the color frame first. Depth is only used to draw the measured robot pointcloud in Viser."""

    output_path: pathlib.Path
    """YAML path to write the estimated 4x4 camera extrinsics (robot-base T camera)."""

    n_timesteps: int = 5
    """Number of furthest-apart joint configurations used in the silhouette cost."""

    sim_outside_sam_weight: float = 0.1
    """Weight of the sim-pixels-outside-the-SAM-mask term. Keep it small (SAM masks can be partial) but > 0: with 0,
    any pose whose sim mask covers the SAM mask is optimal, including cameras pressed against the robot."""

    cma_sigma_pos: float = 0.10
    """CMA-ES initial std for camera translation (meters)."""

    cma_sigma_rot: float = 1.0
    """CMA-ES initial std for camera rotation-vector (radians)."""

    cma_maxiter: int = 250
    """Maximum CMA-ES generations."""

    cma_early_stop_atol: float = 0.01
    """Stop CMA-ES if this generation's min cost (`gen_min`, pixels) changes by less than this for `cma_early_stop_patience` gens."""

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

    workspace_id: str | None = None
    """Workspace bounds id from r2st.constants.WORKSPACE_BOUNDS (robot-base frame), e.g. 'vention-franka-table'. With
    --workspace-prior-extrinsics, pixels whose depth lands outside these bounds (padded by 30 cm, 5 cm below) are blacked out
    before SAM 3, and the SAM masks are cut to the workspace. Pixels without depth are kept. Masks are cached under
    names that include the workspace id and the prior YAML."""

    workspace_prior_extrinsics: pathlib.Path | None = None
    """Approximate extrinsics YAML for --workspace-id, e.g. a previous run's --output-path for this camera."""

    cache_robot_masks: bool = True
    """If set, load/save robot masks as <h5_stem>/robot-mask-propagated__<camera>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy,
    or <h5_stem>/robot-mask-per-frame__<camera>__prompt=<p>__idx=<frame>__kmax=<k>__score_threshold=<t>.npy with
    --per-frame-robot-masks."""

    min_sam_mask_pixels: int = 2500
    """Only sample frames whose SAM robot mask (before erosion) has more than this many pixels (2500 = 50x50). Small
    silhouettes barely constrain the pose."""

    mask_erode_px: int = 9
    """Erode the SAM robot mask with an elliptical kernel of this size (pixels) before the cost, so SAM bleed onto the
    background is not counted as an error. 0 skips erosion."""

    max_point_distance_m: float = 2.0
    """Viser pointcloud: drop measured robot points farther than this (meters, Euclidean) from the camera center."""

    radius_outlier_nb_points: int = 10
    """Viser pointcloud: keep a measured point only if this many points lie within radius_outlier_radius_m of it."""

    radius_outlier_radius_m: float = 0.10
    """Viser pointcloud: search radius (meters) for the measured-cloud radius-outlier filter."""

    n_pcd_samples_real: int = 2048
    """Viser pointcloud: number of FPS points kept from each measured robot cloud."""

    n_random_downsample_initial: int = 16384
    """Viser pointcloud: random pre-FPS cap for each measured robot cloud. Must be >= n_pcd_samples_real."""

    visualize: bool = False
    """If set, start a viser server with the robot model, the measured robot pointcloud (from depth), camera frustum,
    SAM vs sim mask overlay, and a cost plot."""

    visualize_robot_masks: bool = False
    """If set, write debug PNGs for the frame-0 SAM masks that enter the union (top --sam-kmax with score above --sam-score-threshold)."""


def _mask_comparison_overlay(rgb: np.ndarray, sam_mask: np.ndarray, sim_mask: np.ndarray | None) -> np.ndarray:
    """Slightly dimmed RGB with SAM-only pixels (the error) in red, sim-only in blue, and both in green.
    ``sim_mask=None`` (no pose evaluated yet) shows the SAM mask alone (in green)."""
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
    assert sam_mask.shape == rgb.shape[:2], f"sam_mask {sam_mask.shape} != rgb {rgb.shape[:2]}"
    if sim_mask is None:
        sim_mask = sam_mask
    assert sim_mask.shape == rgb.shape[:2], f"sim_mask {sim_mask.shape} != rgb {rgb.shape[:2]}"
    rgb_f = rgb.astype(np.float32)
    out = _OVERLAY_BACKGROUND_ALPHA * rgb_f
    for region, color in (
        (sam_mask & ~sim_mask, _OVERLAY_SAM_ONLY_RGB),
        (~sam_mask & sim_mask, _OVERLAY_SIM_ONLY_RGB),
        (sam_mask & sim_mask, _OVERLAY_BOTH_RGB),
    ):
        out[region] = (1.0 - _OVERLAY_MASK_ALPHA) * rgb_f[region] + _OVERLAY_MASK_ALPHA * color
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _sampled_frame_prefix(h5_path: pathlib.Path, camera: str, sample_idx: int, frame_idx: int) -> pathlib.Path:
    return h5_path.parent / h5_path.stem / "sampled_inputs" / f"{camera}__sample={sample_idx}__idx={frame_idx}"


def _save_sampled_frame_inputs(
    h5_path: pathlib.Path,
    camera: str,
    sample_idx: int,
    frame_idx: int,
    rgb: np.ndarray,
    mask: np.ndarray,
) -> pathlib.Path:
    """Write the RGB and eroded SAM mask overlay that a sampled frame contributes to the cost.

    Returns the output directory.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
    assert mask.shape == rgb.shape[:2], f"mask {mask.shape} != rgb {rgb.shape[:2]}"
    prefix = _sampled_frame_prefix(h5_path, camera, sample_idx, frame_idx)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = prefix.with_name(prefix.name + "__rgb.png")
    overlay_path = prefix.with_name(prefix.name + "__mask_overlay.png")
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(rgb_path), image_bgr), f"Failed to write {rgb_path}"
    assert cv2.imwrite(str(overlay_path), ImageUtils.demo_overlay(image_bgr, mask)), f"Failed to write {overlay_path}"
    return prefix.parent


def _save_best_pose_overlays(
    h5_path: pathlib.Path,
    camera: str,
    fps_idx: np.ndarray,
    rgb_sel: np.ndarray,
    sam_masks: list[np.ndarray],
    sim_masks: list[np.ndarray],
) -> None:
    """Write the SAM vs sim mask overlay at the estimated pose next to each sampled frame's inputs."""
    assert len(sam_masks) == len(sim_masks) == fps_idx.shape[0] == rgb_sel.shape[0], (
        f"sam_masks {len(sam_masks)}, sim_masks {len(sim_masks)}, fps_idx {fps_idx.shape[0]}, "
        f"rgb_sel {rgb_sel.shape[0]} must match"
    )
    for i, frame_idx in enumerate(fps_idx):
        prefix = _sampled_frame_prefix(h5_path, camera, i, int(frame_idx))
        path = prefix.with_name(prefix.name + "__best_pose_sam_vs_sim.png")
        overlay = _mask_comparison_overlay(rgb_sel[i], sam_masks[i], sim_masks[i])
        assert cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)), f"Failed to write {path}"
    print(f"[info] Saved best-pose SAM (red: outside sim) vs sim (blue) overlays to {path.parent}")


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Per-pixel L2 distance (pixels) to the nearest pixel of non-empty bool `mask`; 0 inside it."""
    assert mask.ndim == 2 and mask.dtype == bool, f"mask must be a 2D bool array, got {mask.shape} {mask.dtype}"
    assert mask.any(), "Cannot compute a distance transform to an empty mask"
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


class SimRobotRenderer(SapienRobot):
    """Sapien robot with one camera with the real color intrinsics. Renders robot silhouettes."""

    def __init__(self, urdf_path: pathlib.Path, K: np.ndarray, height: int, width: int) -> None:
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0, f"height and width must be > 0, got {height}x{width}"
        super().__init__(urdf_path)
        # Segmentation channel 1 holds each entity's per_scene_id
        self._robot_seg_ids = np.array(
            sorted({link.entity.per_scene_id for link in self._robot.get_links()}), dtype=np.uint32
        )
        self._hw = (int(height), int(width))
        fx, fy, cx, cy, skew = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), float(K[0, 1])
        assert fx > 0.0 and fy > 0.0, f"K focal lengths must be > 0, got fx={fx} fy={fy}"
        self._cam = self._scene.add_camera(
            _RENDER_CAMERA_NAME,
            int(width),
            int(height),
            fovy=float(2.0 * np.arctan(height / (2.0 * fy))),
            near=MIN_DEPTH_M,
            far=_RENDER_FAR_M,
        )
        self._cam.set_perspective_parameters(MIN_DEPTH_M, _RENDER_FAR_M, fx, fy, cx, cy, skew)
        K_render = np.asarray(self._cam.get_intrinsic_matrix(), dtype=np.float64)
        assert np.allclose(K_render, K, atol=1e-3), f"Sapien camera K\n{K_render}\n!= color K\n{K}"

    def render_mask(self, T_world_cam: np.ndarray) -> np.ndarray:
        """`(H, W)` bool robot mask seen from OpenCV `T_world_cam` at the current qpos."""
        self._cam.local_pose = T_to_sapien_pose(T_world_cam)
        self._scene.update_render()
        self._cam.take_picture()
        segmentation = self._cam.get_picture("Segmentation")
        assert segmentation.shape[:2] == self._hw, f"Segmentation {segmentation.shape} != {self._hw}"
        return np.isin(segmentation[:, :, 1], self._robot_seg_ids)


class ExtrinsicsVisualizer(ExtrinsicsVisualizerBase):
    """Adds the SAM vs sim mask overlay at the shown pose, as a GUI image and on the camera frustum."""

    def __init__(
        self,
        sam_masks: list[np.ndarray],
        K: np.ndarray,
        height: int,
        width: int,
        rgb_frames: np.ndarray,
        pcd_reals: list[np.ndarray],
        cma_maxiter: int,
        fps_idx: np.ndarray,
        renderer: SimRobotRenderer,
        joint_angles: np.ndarray,
        cost_fn: MaskPoseCost,
    ) -> None:
        assert len(sam_masks) == rgb_frames.shape[0], f"sam_masks {len(sam_masks)} != rgb {rgb_frames.shape[0]}"
        assert cost_fn._renderer is not renderer, "Visualizer renderer must not be the cost renderer"
        self._sam_masks = sam_masks
        super().__init__(
            K, height, width, rgb_frames, pcd_reals, cma_maxiter, fps_idx, renderer, joint_angles, cost_fn, "px"
        )

    def _add_method_gui(self) -> None:
        self._server.gui.add_markdown("**red:** SAM outside sim (error) · **blue:** sim only · **green:** both")
        self._overlay_image = self._server.gui.add_image(
            _mask_comparison_overlay(self._rgb_frames[0], self._sam_masks[0], None),
            label="SAM vs sim mask",
            format="jpeg",
        )

    def _draw_timestep(self, t: int, sim: np.ndarray | None) -> None:
        overlay = _mask_comparison_overlay(self._rgb_frames[t], self._sam_masks[t], sim)
        self._overlay_image.image = overlay
        self._frustum.image = overlay


class MaskPoseCost:
    """Mean per-frame silhouette cost of a candidate `T_world_cam` over the sampled timesteps (see module docstring)."""

    def __init__(
        self,
        renderer: SimRobotRenderer,
        joint_angles: np.ndarray,
        sam_masks: list[np.ndarray],
        sim_outside_sam_weight: float,
    ) -> None:
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        assert len(sam_masks) >= 1, "sam_masks must be non-empty"
        assert joint_angles.shape == (
            len(sam_masks),
            renderer.dof,
        ), f"joint_angles {joint_angles.shape} != ({len(sam_masks)}, {renderer.dof})"
        assert sim_outside_sam_weight >= 0.0, f"sim_outside_sam_weight must be >= 0, got {sim_outside_sam_weight}"
        height, width = sam_masks[0].shape
        for i, mask in enumerate(sam_masks):
            assert mask.shape == (height, width), f"sam_masks[{i}] {mask.shape} != {(height, width)}"
            assert mask.dtype == bool and mask.any(), f"sam_masks[{i}] must be a non-empty bool mask"
        self._renderer = renderer
        self._joint_angles = joint_angles
        self._sam_masks = sam_masks
        self._sam_dists = [_distance_to_mask(mask) for mask in sam_masks]
        self._w = float(sim_outside_sam_weight)
        # Each term is at most the image diagonal, so an empty render costs more than any non-empty one
        self._empty_sim_cost = float(np.hypot(height, width)) * (1.0 + self._w)

    def __call__(self, T_world_cam: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Return (mean cost in pixels, per-timestep sim masks)."""
        total = 0.0
        sim_masks: list[np.ndarray] = []
        for q, sam_mask, sam_dist in zip(self._joint_angles, self._sam_masks, self._sam_dists, strict=True):
            self._renderer.set_qpos(q)
            sim_mask = self._renderer.render_mask(T_world_cam)
            sim_masks.append(sim_mask)
            if not sim_mask.any():
                total += self._empty_sim_cost
                continue
            total += float(_distance_to_mask(sim_mask)[sam_mask].mean())
            total += self._w * float(sam_dist[sim_mask].mean())
        return total / len(self._sam_masks), sim_masks


def _measured_robot_clouds(
    args: Args,
    depth_raw_all: np.ndarray,
    color_intrinsics: CameraIntrinsics,
    depth_reprojection: tuple[CameraIntrinsics, np.ndarray, np.ndarray] | None,
    fps_idx: np.ndarray,
    sam_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Measured robot cloud (camera frame) per sampled frame, built like the pointcloud script's, for Viser only. A
    frame without usable depth gets an empty cloud."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    K = color_intrinsics.intrinsic_matrix.astype(np.float64)
    pcd_reals: list[np.ndarray] = []
    for i, frame_idx in enumerate(fps_idx):
        pts, (n_depth, n_unproj, n_inliers) = build_real_pcd(
            frame_depth_m(depth_raw_all[int(frame_idx)], color_intrinsics, depth_reprojection),
            sam_masks[i],
            K,
            args.max_point_distance_m,
            args.radius_outlier_nb_points,
            args.radius_outlier_radius_m,
            args.n_pcd_samples_real,
            args.n_random_downsample_initial,
            device,
            rng,
        )
        pcd_reals.append(pts)
        print(
            f"[info] pcd_real[{i}] frame {int(frame_idx)}: points {n_depth} -> {n_unproj} within "
            f"{args.max_point_distance_m} m -> {n_inliers} radius inliers -> {pts.shape[0]} sampled"
        )
    return pcd_reals


def main(args: Args) -> None:
    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert (
        args.robot_id.lower() in NAME_TO_ROBOT
    ), f"Unknown robot_id {args.robot_id!r}. Available: {sorted(NAME_TO_ROBOT)}"
    assert args.n_timesteps >= 1, f"n_timesteps must be >= 1, got {args.n_timesteps}"
    assert args.sim_outside_sam_weight >= 0.0, f"sim_outside_sam_weight must be >= 0, got {args.sim_outside_sam_weight}"
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
    assert args.min_sam_mask_pixels >= 0, f"min_sam_mask_pixels must be >= 0, got {args.min_sam_mask_pixels}"
    assert (
        args.max_point_distance_m > MIN_DEPTH_M
    ), f"max_point_distance_m must be > {MIN_DEPTH_M}, got {args.max_point_distance_m}"
    assert (
        args.radius_outlier_nb_points >= 1
    ), f"radius_outlier_nb_points must be >= 1, got {args.radius_outlier_nb_points}"
    assert (
        args.radius_outlier_radius_m > 0.0
    ), f"radius_outlier_radius_m must be > 0, got {args.radius_outlier_radius_m}"
    assert args.n_pcd_samples_real >= 1, f"n_pcd_samples_real must be >= 1, got {args.n_pcd_samples_real}"
    assert args.n_random_downsample_initial >= args.n_pcd_samples_real, (
        f"n_random_downsample_initial ({args.n_random_downsample_initial}) must be >= "
        f"n_pcd_samples_real ({args.n_pcd_samples_real})"
    )

    color_intrinsics = get_color_intrinsics(args.camera_model_id)

    print(f"[info] Loading RGB-D + qpos from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb_all, depth_raw_all, qpos_all = load_demo(args.h5_path, args.camera)
    log_elapsed(f"Loaded rgb{rgb_all.shape} depth{depth_raw_all.shape} qpos{qpos_all.shape}", t0)

    n_frames = rgb_all.shape[0]
    assert args.n_timesteps <= n_frames, f"n_timesteps={args.n_timesteps} > n_frames={n_frames}"
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
    workspace_masks, workspace_tag = compute_workspace_masks(
        args.workspace_id,
        args.workspace_prior_extrinsics,
        args.camera,
        depth_raw_all,
        color_intrinsics,
        depth_reprojection,
    )
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
        workspace_masks,
        workspace_tag,
    )
    has_mask, no_mask = frames_with_robot_mask(masks_all, args.per_frame_robot_masks)
    log_elapsed(f"Built robot masks for {n_frames} frames", t0_masks)

    mask_pixels = masks_all[has_mask].reshape(has_mask.size, -1).sum(axis=1)
    candidates = has_mask[mask_pixels > args.min_sam_mask_pixels]
    small = has_mask[mask_pixels <= args.min_sam_mask_pixels]
    print(
        f"[info] Excluded {small.size}/{has_mask.size} masked frames with <= {args.min_sam_mask_pixels} SAM mask "
        f"pixels: {index_ranges_text(small)}"
    )
    assert candidates.size >= args.n_timesteps, (
        f"Only {candidates.size} frames have a SAM robot mask with > {args.min_sam_mask_pixels} pixels, "
        f"need n_timesteps={args.n_timesteps}"
    )
    fps_local = farthest_point_sample_naive(
        qpos_all[candidates].astype(np.float64), args.n_timesteps, distance="circular"
    )
    fps_idx = candidates[np.asarray(fps_local, dtype=np.int64)]
    print(f"[info] FPS circular joint samples (over {candidates.size} usable frames) at RGB frames {fps_idx.tolist()}")

    rgb_sel = rgb_all[fps_idx]
    qpos_sel = qpos_all[fps_idx]
    sam_masks: list[np.ndarray] = []
    for i, frame_idx in enumerate(fps_idx):
        frame_idx = int(frame_idx)
        mask = masks_all[frame_idx]
        n_mask = int(mask.sum())
        mask = erode_mask(mask, args.mask_erode_px)
        sam_masks.append(mask)
        inputs_dir = _save_sampled_frame_inputs(args.h5_path, args.camera, i, frame_idx, rgb_sel[i], mask)
        print(f"[info] sam_mask[{i}] frame {frame_idx}: {n_mask} -> {int(mask.sum())} px eroded")
    print(f"[info] Saved sampled-frame RGB / mask images to {inputs_dir}")

    print(f"[info] Loading Jrl2 robot '{args.robot_id}' ...")
    t0 = time.perf_counter()
    robot = get_robot_by_name(args.robot_id)
    urdf_path = pathlib.Path(robot._urdf_filepath)
    print(f"[info] URDF: {urdf_path}")
    renderer = SimRobotRenderer(urdf_path, K, H, W)
    qpos_sel = expand_demo_qpos_to_joint_names(qpos_sel, robot, renderer.joint_names)
    cost_fn = MaskPoseCost(renderer, qpos_sel, sam_masks, args.sim_outside_sam_weight)
    log_elapsed("Loaded robot + Sapien renderer", t0)

    vis = None
    if args.visualize or args.seed_from_gui:
        pcd_reals = _measured_robot_clouds(
            args, depth_raw_all, color_intrinsics, depth_reprojection, fps_idx, sam_masks
        )
        vis = ExtrinsicsVisualizer(
            sam_masks,
            K,
            H,
            W,
            rgb_sel,
            pcd_reals,
            args.cma_maxiter,
            fps_idx,
            SimRobotRenderer(urdf_path, K, H, W),
            qpos_sel,
            cost_fn,
        )
    best_T, best_cost, best_sim_masks = optimize_extrinsics(
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
    _save_best_pose_overlays(args.h5_path, args.camera, fps_idx, rgb_sel, sam_masks, best_sim_masks)
    t, q_wxyz = best_T[:3, 3], mat2quat(best_T[:3, :3])
    payload = {
        "camera": args.camera,
        "robot_id": args.robot_id,
        "camera_model_id": args.camera_model_id,
        "parent_frame": "robot_base",
        "child_frame": f"{args.camera}_optical",
        "h5_path": str(args.h5_path),
        "n_timesteps": args.n_timesteps,
        "workspace_id": args.workspace_id,
        "workspace_prior_extrinsics": (
            None if args.workspace_prior_extrinsics is None else str(args.workspace_prior_extrinsics)
        ),
        "frames_without_robot_mask": no_mask.tolist(),
        "sampled_frame_indices": fps_idx.tolist(),
        "cost": float(best_cost),
        "sim_outside_sam_weight": float(args.sim_outside_sam_weight),
        "translation": t.tolist(),
        "quaternion_wxyz": q_wxyz.tolist(),
        "matrix": best_T.tolist(),
    }
    args.output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"✅ Estimated camera extrinsics written to {args.output_path}")
    print(f"[info] cost={best_cost:.4f} px")
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
