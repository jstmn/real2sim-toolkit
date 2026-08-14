"""
This script does the following:
1. Loads the demonstration data from the h5 file
2. Loads the robot from Jrl2
3. Runs an optimization procedure to estimate the camera extrinsics.
4. Saves the estimated camera extrinsics to a yaml file.

At a high level, the optimization procedure runs CMA-ES to estimate the SE(3) pose (in the robot's base frame) of the
specified camera. There are two pointclouds that we care about: pcd_real and pcd_sim. pcd_real is the measured
pointcloud from the camera (found by projecting the depth points with the camera's intrinsic matrix). pcd_sim is a
rendered pointcloud from ManiSkill. To get the rendering, a simulation is run. The robot is set to the measured joint
angles. The camera is then moved to the specified camera pose, and lastly the pointcloud is rendered. The cost function
is the chamfer distance between pcd_real and pcd_sim.

Notes:
1. A preproccessing step is performed on the measured pointclouds to remove points that aren't part of the robot. To
    do so, the mask of the robot is generated at each timestep and used to mask the pointcloud. See
    ImageUtils.get_sam_mask for details.
2. If --visualize is set, a viser server is started. The server shows measured vs best-so-far
    simulated pointclouds in the robot-base frame, a camera frustum at the estimated pose, and a
    plot of lowest population cost vs CMA-ES iteration.
3. if --visualize-robot-masks is set, ImageUtils.save_masked_debug writes debug PNGs next to the input h5 file.
4. If --cache-robot-masks is set (default), masks are loaded from / saved to
   `<h5_dir>/<h5_stem>/robot-mask__<camera>__idx=<frame>.npy`.



The pseudo-code is as follows:

inputs:
- rgbds: array of RGBD images
- all_joint_angles: array of joint angles from the demonstration. Same length as rgbds; frame i of qpos is frame i of RGB-D.
- N: the number of timesteps to sample from the demonstration for the cost function
- rgb_to_pcd(rgbd): converts the RGBD image to a robot only pointcloud
- S: number of poses in the CMA-ES population
- render_sim_pointcloud(pose, joint_angle): renders the pointcloud from the simulation at the given pose and joint angle
- compute_chamfer_distance(pcd_real, pcd_sim): computes the chamfer distance between the two pointclouds

# Find the N joint angles that are the furthest apart from each other
timesteps, joint_angles = furthest_point_sample(all_joint_angles, N, distance='circular')

pcd_reals = [rgb_to_pcd(rgbd) for rgbd in rgbds]

until convergence:
    sample poses pi, ..., pN from CMA-ES
    total_costs = [0] * S
    for each pose pi in population:
        total_cost = 0
        for joint_angle in joint_angles:
            pcd_sim = render_sim_pointcloud(pose, joint_angle)
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

import cv2
import h5py
import numpy as np
import sapien
import tyro
import yaml
from jrl2.robots import NAME_TO_ROBOT, get_robot_by_name
from tqdm import tqdm
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat

from r2st.constants import (
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    get_color_intrinsics,
    get_depth_intrinsics,
    get_depth_to_color_extrinsics,
)
from r2st.core import GroundedSAMPredictor
from r2st.geometry import (
    reproject_depth_to_color_frame,
    camera_extrinsic_to_maniskill_pose,
    chamfer_distance,
    depth_mm_to_meters,
    masked_depth_to_points,
    scale_intrinsics,
)
from r2st.utils import (
    ImageUtils,
    MeshUtils,
    farthest_point_sample_naive,
    validate_merged_camera_group,
)

"""
# Example usage:
uv run python examples/estimate_camera_extrinsics.py \
    --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
    --robot-id xarm7 --camera cam_1 --camera-model-id d435 \
    --output-path data/demonstrations/0802/extrinsics.yaml \
    --visualize --visualize-robot-masks
"""

_EMPTY_PCD_COST = 1.0e3
_SIM_CAMERA_NAME = "extrinsics_cam"
_INIT_EYE = np.array([0.80, 0.00, 0.50], dtype=np.float64)
_INIT_TARGET = np.array([0.00, 0.00, 0.20], dtype=np.float64)


@dataclasses.dataclass
class Args:
    h5_path: pathlib.Path
    """Path to a merged sensor-data h5 file (see examples/merge_camera_streams.py)."""

    robot_id: str
    """Jrl2 robot id, e.g. 'xarm7'."""

    camera: str
    """Camera name to read from within the h5 file, e.g. 'cam_1'."""

    camera_model_id: str
    """Camera model id for calibration lookup, e.g. 'd435'. See the README camera-model table."""

    output_path: pathlib.Path
    """YAML path to write the estimated 4x4 camera extrinsics (robot-base T camera)."""

    n_timesteps: int = 8
    """Number of furthest-apart joint configurations used in the chamfer cost."""

    n_pcd_samples: int = 1024
    """Number of FPS points kept from each (real or sim) robot cloud before chamfer."""

    cma_sigma: float = 0.05
    """CMA-ES initial step size on the 6D pose vector (meters / radians)."""

    cma_maxiter: int = 100
    """Maximum CMA-ES generations."""

    cma_popsize: int = 25
    """CMA-ES population size. If unset, the cma library default is used."""

    robot_description: str = "robot arm"
    """GroundedSAM text prompt used to mask the robot in each RGB frame."""

    cache_robot_masks: bool = True
    """If set, load/save robot masks as <h5_stem>/robot-mask__<camera>__idx=<frame>.npy next to the h5 file."""

    visualize: bool = False
    """If set, start a viser server with robot-base pointclouds, camera frustum, and a cost plot."""

    visualize_robot_masks: bool = False
    """If set, write ImageUtils.save_masked_debug PNGs next to the input h5 file."""


def _log_elapsed(label: str, t0: float) -> None:
    print(f"[info] {label} ({time.perf_counter() - t0:.1f}s)")


def _robot_mask_cache_path(h5_path: pathlib.Path, camera: str, frame_idx: int) -> pathlib.Path:
    return h5_path.parent / h5_path.stem / f"robot-mask__{camera}__idx={int(frame_idx)}.npy"


def _load_demo(h5_path: pathlib.Path, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rgb, depth, and qpos, aligned by frame index (same N)."""
    with h5py.File(h5_path, "r") as f:
        validate_merged_camera_group(f, h5_path, camera)
        assert (
            "obs/qpos" in f
        ), f"{h5_path}: missing 'obs/qpos' (re-merge the demo with examples/merge_camera_streams.py)"
        group = f[f"obs/sensor_data/{camera}"]
        rgb = group["rgb"][:]
        depth = group["depth"][:]
        qpos = np.asarray(f["obs/qpos"][:])
    assert qpos.ndim == 2, f"{h5_path}: obs/qpos must be NxD, got {qpos.shape}"
    assert qpos.shape[0] > 0, f"{h5_path}: obs/qpos is empty"
    assert qpos.shape[0] == rgb.shape[0], (
        f"{h5_path}: obs/qpos has {qpos.shape[0]} frames, {camera} rgb has {rgb.shape[0]} "
        "(qpos and RGB-D must share the same index axis)"
    )
    return rgb, depth, qpos


def _look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """4x4 T_world_cam in OpenCV / RealSense optical convention (+Z forward, +Y down)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = np.asarray(up, dtype=np.float64).reshape(3)
    z_fwd = target - eye
    z_norm = np.linalg.norm(z_fwd)
    assert z_norm > 1e-8, f"eye and target are coincident: eye={eye} target={target}"
    z_fwd = z_fwd / z_norm
    x_right = np.cross(up, z_fwd)
    x_norm = np.linalg.norm(x_right)
    assert x_norm > 1e-8, f"up is parallel to look direction: up={up} z={z_fwd}"
    x_right = x_right / x_norm
    y_down = np.cross(z_fwd, x_right)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_right
    T[:3, 1] = y_down
    T[:3, 2] = z_fwd
    T[:3, 3] = eye
    return T


def _T_to_vec(T: np.ndarray) -> np.ndarray:
    """SE(3) 4x4 → 6D [tx, ty, tz, rx, ry, rz] with rotation-vector orientation."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    axis, angle = mat2axangle(T[:3, :3])
    rotvec = np.asarray(axis, dtype=np.float64) * float(angle)
    return np.concatenate([T[:3, 3].astype(np.float64), rotvec], axis=0)


def _vec_to_T(x: np.ndarray) -> np.ndarray:
    """6D [tx, ty, tz, rx, ry, rz] → SE(3) 4x4."""
    x = np.asarray(x, dtype=np.float64).reshape(6)
    rotvec = x[3:]
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        R = np.eye(3, dtype=np.float64)
    else:
        R = axangle2mat(rotvec / angle, angle)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = x[:3]
    return T


def _T_to_sapien_pose(T_world_cam: np.ndarray) -> sapien.Pose:
    """RealSense-optical T_world_cam → Sapien / ManiSkill camera pose."""
    t, R_ms = camera_extrinsic_to_maniskill_pose(T_world_cam)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_ms
    mat[:3, 3] = t
    return sapien.Pose(mat[:3, 3], mat2quat(mat[:3, :3]))


def _urdf_for_sapien(urdf_path: pathlib.Path) -> pathlib.Path:
    """Sapien cannot resolve `file://` mesh hrefs that xacrodoc emits. Strip the prefix."""
    text = urdf_path.read_text()
    rewritten = text.replace('filename="file://', 'filename="')
    if rewritten == text:
        return urdf_path
    out = urdf_path.with_name(urdf_path.stem + "__sapien.urdf")
    out.write_text(rewritten)
    return out


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE(3) `T` to `(N, 3)` points. Empty clouds stay empty."""
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    assert pts.ndim == 2 and pts.shape[1] == 3, f"pts must be (N, 3), got {pts.shape}"
    if pts.shape[0] == 0:
        return pts
    return (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def _subsample_pcd(points: np.ndarray, n_samples: int) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    if points.shape[0] <= n_samples:
        return points.astype(np.float32, copy=False)
    idx = farthest_point_sample_naive(points.astype(np.float64), n_samples, distance="euclidean")
    return points[idx].astype(np.float32, copy=False)


class SimRobotRenderer:
    """Sapien scene with the Jrl2 URDF. Camera pose uses ManiSkill (Sapien) convention."""

    def __init__(self, urdf_path: pathlib.Path, K: np.ndarray, height: int, width: int, joint_names: list[str]) -> None:
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0
        sapien_urdf = _urdf_for_sapien(urdf_path)
        assert sapien_urdf.is_file(), f"Sapien URDF not found: {sapien_urdf}"

        self._scene = sapien.Scene()
        self._scene.set_ambient_light([0.45, 0.45, 0.45])
        self._scene.add_directional_light([0.0, 1.0, -1.0], [1.0, 1.0, 1.0])
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(sapien_urdf))
        assert robot is not None, f"Failed to load URDF {sapien_urdf}"
        self._robot = robot
        self._robot.set_root_pose(sapien.Pose([0.0, 0.0, 0.0]))

        sim_joint_names = [j.name for j in self._robot.get_active_joints()]
        assert sim_joint_names == list(
            joint_names
        ), f"Jrl2 actuated joints {list(joint_names)} != Sapien active joints {sim_joint_names}"
        self._dof = int(self._robot.dof)
        assert self._dof == len(joint_names), f"dof {self._dof} != n joints {len(joint_names)}"

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        fovy = 2.0 * np.arctan2(height * 0.5, fy)
        self._cam = self._scene.add_camera(
            _SIM_CAMERA_NAME, width, height, fovy=float(fovy), near=MIN_DEPTH_M, far=MAX_DEPTH_M
        )
        self._cam.set_focal_lengths(fx, fy)
        self._cam.set_principal_point(cx, cy)

    def render_pointcloud(self, T_world_cam: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Robot-only cloud in OpenCV camera frame, `(M, 3)` float32. Empty if nothing is visible."""
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        assert qpos.shape == (self._dof,), f"qpos must be ({self._dof},), got {qpos.shape}"
        self._robot.set_qpos(qpos)
        self._cam.local_pose = _T_to_sapien_pose(T_world_cam)
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
        valid = robot_px & np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float32)
        return xyz_cv[valid].astype(np.float32)


class ExtrinsicsVisualizer:
    """Viser overlay in the robot-base frame: clouds, camera frustum, CMA-ES cost plot."""

    def __init__(
        self,
        pcd_reals: list[np.ndarray],
        K: np.ndarray,
        height: int,
        width: int,
        rgb_frames: np.ndarray,
        cma_maxiter: int,
    ) -> None:
        import viser
        import viser.uplot as uplot

        assert len(pcd_reals) >= 1
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert height > 0 and width > 0
        assert rgb_frames.ndim == 4 and rgb_frames.shape[0] == len(pcd_reals), (
            f"rgb_frames must be ({len(pcd_reals)}, H, W, 3), got {rgb_frames.shape}"
        )
        assert cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {cma_maxiter}"
        fy = float(K[1, 1])
        assert fy > 0, f"K[1,1] (fy) must be > 0, got {fy}"

        self._pcd_reals = pcd_reals
        self._rgb_frames = rgb_frames
        self._T = _look_at_opencv(_INIT_EYE, _INIT_TARGET)
        self._maxiter = cma_maxiter
        self._xs = np.arange(1, cma_maxiter + 1, dtype=np.float64)
        self._ys_best = np.full(cma_maxiter, np.nan, dtype=np.float64)
        self._ys_pop = np.full(cma_maxiter, np.nan, dtype=np.float64)
        self._Ts: list[np.ndarray | None] = [None] * cma_maxiter
        self._pcd_sims_hist: list[list[np.ndarray] | None] = [None] * cma_maxiter
        self._best_costs: list[float | None] = [None] * cma_maxiter
        self._n_recorded = 0
        self._timestep = 0
        self._iteration = 1

        self._server = viser.ViserServer()
        self._server.scene.world_axes.visible = True
        self._server.scene.world_axes.scale = 0.15
        MeshUtils.add_xy_grid(self._server)
        self._server.initial_camera.position = (1.2, 1.2, 0.8)
        self._server.initial_camera.look_at = (0.0, 0.0, 0.3)
        self._server.initial_camera.up = (0.0, 0.0, 1.0)

        self._real_handle = self._server.scene.add_point_cloud(
            "/pcd_real",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=(220, 60, 60),
            point_size=0.003,
            point_shape="circle",
        )
        self._sim_handle = self._server.scene.add_point_cloud(
            "/pcd_sim",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=(40, 200, 120),
            point_size=0.003,
            point_shape="circle",
        )
        wxyz0, pos0 = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum = self._server.scene.add_camera_frustum(
            "/camera",
            fov=float(2.0 * np.arctan(height / (2.0 * fy))),
            aspect=width / height,
            scale=0.15,
            line_width=1.5,
            color=(40, 120, 255),
            image=rgb_frames[0],
            format="jpeg",
            wxyz=wxyz0,
            position=pos0,
        )
        self._cam_axes = self._server.scene.add_frame(
            "/camera_axes",
            axes_length=0.08,
            axes_radius=0.004,
            origin_radius=0.008,
            wxyz=wxyz0,
            position=pos0,
        )
        self._cost_md = self._server.gui.add_markdown("best cost: —")
        self._pose_md = self._server.gui.add_markdown("camera pose: —")
        timestep_slider = self._server.gui.add_slider(
            "timestep", min=0, max=max(len(pcd_reals) - 1, 0), step=1, initial_value=0
        )

        @timestep_slider.on_update
        def _on_timestep(_) -> None:
            self._timestep = int(timestep_slider.value)
            self._apply()

        self._iter_slider = self._server.gui.add_slider(
            "iteration", min=1, max=cma_maxiter, step=1, initial_value=1
        )

        @self._iter_slider.on_update
        def _on_iteration(_) -> None:
            self._iteration = int(self._iter_slider.value)
            self._apply()

        self._cost_plot = self._server.gui.add_uplot(
            data=(self._xs, self._ys_best, self._ys_pop),
            series=(
                uplot.Series(show=False),
                uplot.Series(label="best found", stroke="#e67e22", width=2),
                uplot.Series(label="pop min", stroke="#3498db", width=1),
            ),
            title="CMA-ES cost vs iteration",
            scales={"x": {"time": False}},
            axes=(
                uplot.Axis(label="iteration"),
                uplot.Axis(label="cost"),
            ),
            aspect=1.8,
            height=220,
        )
        self._apply()
        print(f"[info] Viser extrinsics view at http://{self._server.get_host()}:{self._server.get_port()}")

    def set_best(self, pcd_sims: list[np.ndarray], cost: float, T_world_cam: np.ndarray) -> None:
        """Live-update the scene when a new global best is found, if viewing the latest iteration."""
        assert len(pcd_sims) == len(
            self._pcd_reals
        ), f"sim clouds {len(pcd_sims)} != real clouds {len(self._pcd_reals)}"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        if self._n_recorded > 0 and self._iteration != self._n_recorded:
            return
        self._apply(T=T_world_cam.copy(), pcd_sims=pcd_sims, cost=float(cost))

    def record_generation(
        self,
        generation: int,
        best_cost: float,
        pop_min: float,
        T_world_cam: np.ndarray,
        pcd_sims: list[np.ndarray],
    ) -> None:
        assert 1 <= generation <= self._maxiter, f"generation {generation} not in [1, {self._maxiter}]"
        assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
        assert len(pcd_sims) == len(
            self._pcd_reals
        ), f"sim clouds {len(pcd_sims)} != real clouds {len(self._pcd_reals)}"
        i = generation - 1
        self._ys_best[i] = float(best_cost)
        self._ys_pop[i] = float(pop_min)
        self._Ts[i] = T_world_cam.copy()
        self._pcd_sims_hist[i] = pcd_sims
        self._best_costs[i] = float(best_cost)
        self._cost_plot.data = (self._xs, self._ys_best, self._ys_pop)
        follow = self._n_recorded == 0 or self._iteration == self._n_recorded
        self._n_recorded = generation
        if follow:
            self._iteration = generation
            self._iter_slider.value = generation
            self._apply()

    def _snapshot_index(self) -> int | None:
        if self._n_recorded == 0:
            return None
        return min(int(self._iteration), self._n_recorded) - 1

    def _apply(
        self,
        T: np.ndarray | None = None,
        pcd_sims: list[np.ndarray] | None = None,
        cost: float | None = None,
    ) -> None:
        t = self._timestep
        snap_i = self._snapshot_index()
        if T is None and snap_i is not None:
            T = self._Ts[snap_i]
            pcd_sims = self._pcd_sims_hist[snap_i]
            cost = self._best_costs[snap_i]
            assert T is not None, f"Missing T snapshot at iteration {snap_i + 1}"
            assert pcd_sims is not None, f"Missing sim-cloud snapshot at iteration {snap_i + 1}"
            assert cost is not None, f"Missing cost snapshot at iteration {snap_i + 1}"
        if T is not None:
            self._T = T
        if cost is not None:
            self._cost_md.content = f"**iter {self._iteration} best cost:** `{cost:.6f}`"
            cam_t = self._T[:3, 3]
            self._pose_md.content = f"**camera t (robot base):** `[{cam_t[0]:.4f}, {cam_t[1]:.4f}, {cam_t[2]:.4f}]`"
        self._real_handle.points = _transform_points(self._T, self._pcd_reals[t])
        if pcd_sims is not None and pcd_sims[t].shape[0] > 0:
            self._sim_handle.points = _transform_points(self._T, pcd_sims[t])
        wxyz, pos = MeshUtils._pose_mat_to_wxyz_position(self._T)
        self._frustum.wxyz = wxyz
        self._frustum.position = pos
        self._frustum.image = self._rgb_frames[t]
        self._cam_axes.wxyz = wxyz
        self._cam_axes.position = pos

    def wait(self) -> None:
        self._server.sleep_forever()


def _pose_cost(
    T_world_cam: np.ndarray,
    renderer: SimRobotRenderer,
    joint_angles: np.ndarray,
    pcd_reals: list[np.ndarray],
    n_pcd_samples: int,
    chamfer_device: str,
) -> tuple[float, list[np.ndarray], float, float]:
    total = 0.0
    pcd_sims: list[np.ndarray] = []
    t_render = 0.0
    t_chamfer = 0.0
    for t, q in enumerate(joint_angles):
        t0 = time.perf_counter()
        pcd_sim = renderer.render_pointcloud(T_world_cam, q)
        t_render += time.perf_counter() - t0
        if pcd_sim.shape[0] == 0:
            pcd_sims.append(pcd_sim)
            total += _EMPTY_PCD_COST
            continue
        pcd_sim = _subsample_pcd(pcd_sim, n_pcd_samples)
        pcd_sims.append(pcd_sim)
        t0 = time.perf_counter()
        total += float(chamfer_distance(pcd_reals[t], pcd_sim, device=chamfer_device))
        t_chamfer += time.perf_counter() - t0
    return total, pcd_sims, t_render, t_chamfer


def _optimize_extrinsics(
    renderer: SimRobotRenderer,
    joint_angles: np.ndarray,
    pcd_reals: list[np.ndarray],
    n_pcd_samples: int,
    chamfer_device: str,
    cma_sigma: float,
    cma_maxiter: int,
    cma_popsize: int | None,
    vis: ExtrinsicsVisualizer | None,
) -> tuple[np.ndarray, float]:
    """CMA-ES over 6D camera pose. Returns (best 4x4 T_world_cam, cost)."""
    import cma

    x0 = _T_to_vec(_look_at_opencv(_INIT_EYE, _INIT_TARGET))
    cma_opts: dict = {"maxiter": cma_maxiter, "verbose": -1, "CMA_stds": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]}
    if cma_popsize is not None:
        cma_opts["popsize"] = cma_popsize
    es = cma.CMAEvolutionStrategy(x0, cma_sigma, cma_opts)
    print(f"[info] CMA-ES x0={x0} sigma={cma_sigma} popsize={es.popsize} maxiter={cma_maxiter}")

    best_cost = np.inf
    best_T: np.ndarray | None = None
    best_pcd_sims: list[np.ndarray] | None = None
    generation = 0
    t0_opt = time.perf_counter()
    while not es.stop():
        generation += 1
        t0_gen = time.perf_counter()
        xs = es.ask()
        costs = []
        t_render = 0.0
        t_chamfer = 0.0
        t_vis = 0.0
        for x in tqdm(xs, desc=f"cma gen {generation}", leave=False):
            T = _vec_to_T(x)
            cost, pcd_sims, dt_render, dt_chamfer = _pose_cost(
                T, renderer, joint_angles, pcd_reals, n_pcd_samples, chamfer_device
            )
            t_render += dt_render
            t_chamfer += dt_chamfer
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_T = T.copy()
                best_pcd_sims = pcd_sims
                if vis is not None:
                    t0_vis = time.perf_counter()
                    vis.set_best(pcd_sims, best_cost, best_T)
                    t_vis += time.perf_counter() - t0_vis
        es.tell(xs, costs)
        pop_min = float(min(costs))
        assert best_T is not None and best_pcd_sims is not None, "CMA-ES generation produced no pose"
        if vis is not None:
            vis.record_generation(generation, best_cost, pop_min, best_T, best_pcd_sims)
        dt_gen = time.perf_counter() - t0_gen
        print(
            f"[info] gen {generation} ({dt_gen:.1f}s): best={best_cost:.6f}  gen_min={pop_min:.6f}  "
            f"gen_mean={float(np.mean(costs)):.6f}  "
            f"render={t_render:.1f}s chamfer={t_chamfer:.1f}s vis={t_vis:.1f}s"
        )

    assert best_T is not None, "CMA-ES produced no pose"
    _log_elapsed(f"CMA-ES finished ({generation} gens)", t0_opt)
    return best_T, best_cost


def main(args: Args) -> None:
    import torch

    assert args.h5_path.is_file(), f"H5 file not found: {args.h5_path}"
    assert (
        args.robot_id.lower() in NAME_TO_ROBOT
    ), f"Unknown robot_id {args.robot_id!r}. Available: {sorted(NAME_TO_ROBOT)}"
    assert args.n_timesteps >= 1, f"n_timesteps must be >= 1, got {args.n_timesteps}"
    assert args.n_pcd_samples >= 1, f"n_pcd_samples must be >= 1, got {args.n_pcd_samples}"
    assert args.cma_sigma > 0, f"cma_sigma must be > 0, got {args.cma_sigma}"
    assert args.cma_maxiter >= 1, f"cma_maxiter must be >= 1, got {args.cma_maxiter}"
    assert args.cma_popsize is None or args.cma_popsize >= 2, f"cma_popsize must be >= 2, got {args.cma_popsize}"
    assert len(args.robot_description) > 0, "robot_description must not be empty"
    chamfer_device = "cuda" if torch.cuda.is_available() else "cpu"

    depth_intrinsics = get_depth_intrinsics(args.camera_model_id)
    color_intrinsics = get_color_intrinsics(args.camera_model_id)

    print(f"[info] Loading RGB-D + qpos from {args.h5_path} ({args.camera}) ...")
    t0 = time.perf_counter()
    rgb_all, depth_raw_all, qpos_all = _load_demo(args.h5_path, args.camera)
    _log_elapsed(f"Loaded rgb{rgb_all.shape} depth{depth_raw_all.shape} qpos{qpos_all.shape}", t0)

    n_frames = rgb_all.shape[0]
    assert args.n_timesteps <= n_frames, f"n_timesteps={args.n_timesteps} > n_frames={n_frames}"
    fps_idx = farthest_point_sample_naive(qpos_all.astype(np.float64), args.n_timesteps, distance="circular")
    fps_idx = np.asarray(fps_idx, dtype=np.int64)
    print(f"[info] FPS circular joint samples at RGB-D frames {fps_idx.tolist()}")

    rgb_sel = rgb_all[fps_idx]
    depth_raw_sel = depth_raw_all[fps_idx]
    qpos_sel = qpos_all[fps_idx]
    H, W = int(rgb_sel.shape[1]), int(rgb_sel.shape[2])
    K = scale_intrinsics(
        color_intrinsics.intrinsic_matrix,
        (color_intrinsics.height, color_intrinsics.width),
        (H, W),
    ).astype(np.float64)
    print(f"[info] Color K ({args.camera_model_id}):\n{K}")

    print(f"[info] Reprojecting depth into the color frame ({args.n_timesteps} sampled frames) ...")
    t0 = time.perf_counter()
    R_dc, t_dc = get_depth_to_color_extrinsics(args.camera_model_id)
    depth_m_sel = np.stack(
        [
            reproject_depth_to_color_frame(
                depth_mm_to_meters(depth_raw_sel[i]),
                depth_intrinsics,
                color_intrinsics,
                R_dc,
                t_dc,
            )
            for i in range(args.n_timesteps)
        ],
        axis=0,
    )
    _log_elapsed("Reprojected depth into the color frame", t0)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor: GroundedSAMPredictor | None = None
    pcd_reals: list[np.ndarray] = []
    t0_masks = time.perf_counter()
    for i, frame_idx in enumerate(fps_idx):
        image_bgr = cv2.cvtColor(rgb_sel[i], cv2.COLOR_RGB2BGR)
        cache_path = _robot_mask_cache_path(args.h5_path, args.camera, int(frame_idx))
        if args.cache_robot_masks and cache_path.is_file():
            mask = np.load(cache_path)
            assert isinstance(mask, np.ndarray), f"{cache_path}: expected ndarray, got {type(mask)}"
            assert mask.shape == image_bgr.shape[:2], (
                f"{cache_path}: mask shape {mask.shape} != image {image_bgr.shape[:2]}"
            )
            assert mask.dtype == bool, f"{cache_path}: mask dtype must be bool, got {mask.dtype}"
            print(f"[info] Loaded robot mask from {cache_path}")
        else:
            if predictor is None:
                print("[info] Loading GroundedSAM on cpu ...")
                t0 = time.perf_counter()
                predictor = GroundedSAMPredictor(device="cpu")
                _log_elapsed("GroundedSAM loaded", t0)
            print(f"[info] Segmenting '{args.robot_description}' on frame {int(frame_idx)} ...")
            t0 = time.perf_counter()
            mask = ImageUtils.get_sam_mask(predictor, image_bgr, args.robot_description)
            _log_elapsed(f"Segmented '{args.robot_description}' on frame {int(frame_idx)}", t0)
            if args.cache_robot_masks:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, mask)
                print(f"[info] Saved robot mask to {cache_path}")
        assert mask.any(), f"Empty robot mask at frame {int(frame_idx)} for prompt {args.robot_description!r}"
        print(f"[info] Mask pixels: {int(mask.sum())} / {mask.size}")
        if args.visualize_robot_masks:
            ImageUtils.save_masked_debug(
                image_bgr,
                mask,
                args.h5_path.parent,
                f"{args.camera}__robot_mask__{int(frame_idx)}",
            )
        pts = masked_depth_to_points(depth_m_sel[i], mask, K)
        pts = _subsample_pcd(pts, args.n_pcd_samples)
        assert pts.shape[0] >= 1, f"No robot points at frame {int(frame_idx)}"
        pcd_reals.append(pts)
        print(f"[info] pcd_real[{i}] n={pts.shape[0]}")
    _log_elapsed(f"Segmented {args.n_timesteps} frames on cpu", t0_masks)

    print(f"[info] Loading Jrl2 robot '{args.robot_id}' ...")
    t0 = time.perf_counter()
    robot = get_robot_by_name(args.robot_id)
    assert qpos_sel.shape[1] == robot.num_actuators, (
        f"obs/qpos dim {qpos_sel.shape[1]} != Jrl2 {args.robot_id} actuators {robot.num_actuators} "
        f"({robot.actuated_joint_names})"
    )
    urdf_path = pathlib.Path(robot._urdf_filepath)
    print(f"[info] URDF: {urdf_path}")
    renderer = SimRobotRenderer(urdf_path, K, H, W, robot.actuated_joint_names)
    _log_elapsed("Loaded robot + Sapien renderer", t0)

    vis = (
        ExtrinsicsVisualizer(pcd_reals, K, H, W, rgb_sel, args.cma_maxiter) if args.visualize else None
    )
    best_T, best_cost = _optimize_extrinsics(
        renderer,
        qpos_sel,
        pcd_reals,
        args.n_pcd_samples,
        chamfer_device,
        args.cma_sigma,
        args.cma_maxiter,
        args.cma_popsize,
        vis,
    )
    t, q_wxyz = best_T[:3, 3], mat2quat(best_T[:3, :3])
    payload = {
        "camera": args.camera,
        "robot_id": args.robot_id,
        "parent_frame": "robot_base",
        "child_frame": f"{args.camera}_optical",
        "h5_path": str(args.h5_path),
        "n_timesteps": args.n_timesteps,
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

    if vis is not None:
        vis.wait()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
