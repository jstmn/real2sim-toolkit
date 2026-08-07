"""FoundationPose-based object pose estimation and tracking.

Expects a FoundationPose checkout at ``src/r2st/FoundationPose`` (same layout as
the GroundingDINO vendor tree). Lazy-imports nvdiffrast and FoundationPose so
the rest of the package can load without those heavy deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

FOUNDATIONPOSE_DIR = Path(__file__).resolve().parent / "FoundationPose"


def _ensure_foundationpose_on_path() -> None:
    assert FOUNDATIONPOSE_DIR.is_dir(), (
        f"FoundationPose not found at {FOUNDATIONPOSE_DIR}. " "Vendor or symlink the FoundationPose repo there."
    )
    path_str = str(FOUNDATIONPOSE_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_foundationpose_deps():
    _ensure_foundationpose_on_path()
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import draw_posed_3d_box, draw_xyz_axis

    return dr, FoundationPose, PoseRefinePredictor, ScorePredictor, draw_posed_3d_box, draw_xyz_axis


def _pose_mat_to_6d(pose) -> np.ndarray:
    """Convert a possibly-batched 4x4 pose (torch tensor or ndarray) to [tx,ty,tz,rx,ry,rz]."""
    import torch
    from scipy.spatial.transform import Rotation

    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    pose = np.asarray(pose)
    if pose.ndim == 3:
        pose = pose[0]
    assert pose.shape == (4, 4), f"pose must be 4x4 (optionally batched), got {pose.shape}"
    xyz = pose[:3, 3]
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
    return np.r_[xyz, euler]


def _pose_6d_to_mat(pose_6d: np.ndarray) -> np.ndarray:
    """Inverse of `_pose_mat_to_6d`."""
    from scipy.spatial.transform import Rotation

    assert pose_6d.shape == (6,), f"pose_6d must be a 6-vector, got {pose_6d.shape}"
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", pose_6d[3:], degrees=False).as_matrix()
    mat[:3, 3] = pose_6d[:3]
    return mat


def _pose_xy_at_image_point(pose_cam, K: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Camera-frame (tx, ty) such that, at pose_cam's current depth, it projects to image point (x, y)."""
    pose_2d = pose_cam[0] if pose_cam.ndim == 3 else pose_cam
    tz = float(pose_2d[2, 3])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return (x - cx) * tz / fx, (y - cy) * tz / fy


def _reanchor_pose_xy(pose_cam, tx: float, ty: float):
    """Return a copy of pose_cam (torch tensor, optionally batched) with its translation's
    (x, y) components set to (tx, ty)."""
    out = pose_cam.clone()
    if pose_cam.ndim == 3:
        out[:, 0, 3] = tx
        out[:, 1, 3] = ty
    else:
        out[0, 3] = tx
        out[1, 3] = ty
    return out


def draw_pose_overlay(
    color_rgb: np.ndarray,
    pose_cam: np.ndarray,
    mesh_path: str | Path,
    K: np.ndarray,
    axis_len: float = 0.1,
) -> np.ndarray:
    """Return an RGB image with pose axes and a 3D bbox overlay (no file I/O)."""
    _, _, _, _, draw_posed_3d_box, draw_xyz_axis = _load_foundationpose_deps()

    assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
    assert color_rgb.dtype == np.uint8, f"color_rgb dtype must be uint8, got {color_rgb.dtype}"
    assert pose_cam.shape == (4, 4), f"pose_cam must be 4x4, got {pose_cam.shape}"
    assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
    assert axis_len > 0, f"axis_len must be > 0, got {axis_len}"

    mesh_path = Path(mesh_path)
    assert mesh_path.is_file(), f"Mesh not found: {mesh_path}"

    img = color_rgb.copy()
    mesh = trimesh.load(str(mesh_path), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(mesh)}"
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    center_pose = pose_cam @ np.linalg.inv(to_origin)
    img = draw_posed_3d_box(K, img=img, ob_in_cam=center_pose, bbox=bbox)
    img = draw_xyz_axis(img, ob_in_cam=center_pose, scale=axis_len, K=K, thickness=3, transparency=0, is_input_rgb=True)
    assert img.shape == color_rgb.shape, f"Overlay changed shape: {img.shape} vs {color_rgb.shape}"
    return img


class FoundationPoseRunner:
    """Keeps scorer/refiner/glctx alive and estimates pose once per mesh (register)."""

    def __init__(self, debug_dir: str | Path = Path("debug_fp")):
        dr, FoundationPose, PoseRefinePredictor, ScorePredictor, _, _ = _load_foundationpose_deps()
        self._FoundationPose = FoundationPose
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

    def estimate_pose(
        self,
        mesh_file: str | Path,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
    ) -> np.ndarray:
        """Return a 4x4 object pose in the camera frame via FoundationPose.register."""
        mesh_file = Path(mesh_file)
        assert mesh_file.is_file(), f"Mesh not found: {mesh_file}"
        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
        assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
        assert depth_m.shape == color_rgb.shape[:2], f"depth shape {depth_m.shape} != color {color_rgb.shape[:2]}"
        assert mask.shape == color_rgb.shape[:2], f"mask shape {mask.shape} != color {color_rgb.shape[:2]}"
        assert (
            mask.dtype == bool or np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.floating)
        ), f"Unexpected mask dtype: {mask.dtype}"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert iteration >= 1, f"iteration must be >= 1, got {iteration}"

        mesh = trimesh.load(str(mesh_file), force="mesh")
        assert isinstance(mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(mesh)}"
        assert len(mesh.vertices) > 0, f"Mesh has no vertices: {mesh_file}"

        est = self._FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug_dir=str(self.debug_dir),
            debug=0,
        )
        pose_cam = est.register(
            K=K,
            rgb=color_rgb,
            depth=depth_m,
            ob_mask=mask.astype(bool),
            iteration=iteration,
        )
        pose_cam = np.asarray(pose_cam, dtype=np.float64)
        assert pose_cam.shape == (4, 4), f"Expected 4x4 pose, got {pose_cam.shape}"
        return pose_cam


class FoundationPoseTracker:
    """Keeps a single FoundationPose instance alive: register on frame 0, track after.

    Optionally re-anchors FoundationPose's (x, y) translation each frame using a Cutie 2D
    tracker (``use_2d_tracker``), and/or fuses that measurement with FoundationPose's own
    pose via a 6-DoF Kalman filter (``use_kalman_filter``) instead of overwriting it outright.
    This mirrors the tracking loop in FoundationPose++
    (https://github.com/lidingsheng/FoundationPose-plus-plus, ``src/obj_pose_track.py``).
    """

    def __init__(
        self,
        mesh_file: str | Path,
        debug_dir: str | Path = Path("debug_fp"),
        use_2d_tracker: bool = False,
        use_kalman_filter: bool = False,
        kalman_measurement_noise_scale: float = 0.05,
    ):
        assert use_2d_tracker or not use_kalman_filter, (
            "use_kalman_filter requires use_2d_tracker=True: the filter fuses the 2D tracker's "
            "image-plane measurement with FoundationPose's own pose estimate each frame."
        )
        dr, FoundationPose, PoseRefinePredictor, ScorePredictor, _, _ = _load_foundationpose_deps()
        mesh_file = Path(mesh_file)
        assert mesh_file.is_file(), f"Mesh not found: {mesh_file}"

        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._mesh_names = [mesh_file.stem]
        self._mesh_paths = [mesh_file]
        self.mesh = trimesh.load(str(mesh_file), force="mesh")
        assert isinstance(self.mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(self.mesh)}"
        assert len(self.mesh.vertices) > 0, f"Mesh has no vertices: {mesh_file}"

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug_dir=str(self.debug_dir),
            debug=0,
        )
        self.last_pose_cam: np.ndarray | None = None
        self._initial_mask: np.ndarray | None = None

        self._tracker_2d = None
        if use_2d_tracker:
            from r2st.cutie_tracker import CutieTracker

            self._tracker_2d = CutieTracker()

        self._kf = None
        if use_kalman_filter:
            from r2st.kalman_filter_6d import KalmanFilter6D

            self._kf = KalmanFilter6D(kalman_measurement_noise_scale)
        self._kf_mean: np.ndarray | None = None
        self._kf_covariance: np.ndarray | None = None

    @property
    def initial_mask(self) -> np.ndarray | None:
        return self._initial_mask

    @property
    def mesh_paths(self) -> list[Path]:
        return list(self._mesh_paths)

    @property
    def mesh_names(self) -> list[str]:
        return list(self._mesh_names)

    def register(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
    ) -> np.ndarray:
        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
        assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
        assert depth_m.shape == color_rgb.shape[:2], f"depth shape {depth_m.shape} != color {color_rgb.shape[:2]}"
        assert mask.shape == color_rgb.shape[:2], f"mask shape {mask.shape} != color {color_rgb.shape[:2]}"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert iteration >= 1, f"iteration must be >= 1, got {iteration}"

        self._initial_mask = mask.astype(bool)
        pose_cam = self.est.register(
            K=K,
            rgb=color_rgb,
            depth=depth_m,
            ob_mask=self._initial_mask,
            iteration=iteration,
        )
        pose_cam = np.asarray(pose_cam, dtype=np.float64)
        assert pose_cam.shape == (4, 4), f"Expected 4x4 pose, got {pose_cam.shape}"
        self.last_pose_cam = pose_cam

        if self._tracker_2d is not None:
            self._tracker_2d.initialize(color_rgb, self._initial_mask)
        if self._kf is not None:
            self._kf_mean, self._kf_covariance = self._kf.initiate(_pose_mat_to_6d(self.est.pose_last))

        return pose_cam

    def track(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        iteration: int = 5,
    ) -> np.ndarray:
        assert self.last_pose_cam is not None, "Must call register() before track()"
        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
        assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
        assert depth_m.shape == color_rgb.shape[:2], f"depth shape {depth_m.shape} != color {color_rgb.shape[:2]}"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        assert iteration >= 1, f"iteration must be >= 1, got {iteration}"

        if self._tracker_2d is not None:
            bbox_xywh = self._tracker_2d.track(color_rgb)
            if bbox_xywh is not None:
                px = bbox_xywh[0] + bbox_xywh[2] / 2
                py = bbox_xywh[1] + bbox_xywh[3] / 2
                tx, ty = _pose_xy_at_image_point(self.est.pose_last, K, px, py)
                if self._kf is None:
                    self.est.pose_last = _reanchor_pose_xy(self.est.pose_last, tx, ty)
                else:
                    self._kf_mean, self._kf_covariance = self._kf.update(
                        self._kf_mean, self._kf_covariance, _pose_mat_to_6d(self.est.pose_last)
                    )
                    self._kf_mean, self._kf_covariance = self._kf.update_from_xy(
                        self._kf_mean, self._kf_covariance, np.array([tx, ty])
                    )
                    fused_pose = _pose_6d_to_mat(self._kf_mean[:6])
                    self.est.pose_last = self.est.pose_last.new_tensor(fused_pose).reshape(1, 4, 4)

        pose_cam = self.est.track_one(rgb=color_rgb, depth=depth_m, K=K, iteration=iteration)

        if self._kf is not None:
            self._kf_mean, self._kf_covariance = self._kf.predict(self._kf_mean, self._kf_covariance)

        pose_cam = np.asarray(pose_cam, dtype=np.float64)
        assert pose_cam.shape == (4, 4), f"Expected 4x4 pose, got {pose_cam.shape}"
        self.last_pose_cam = pose_cam
        return pose_cam
