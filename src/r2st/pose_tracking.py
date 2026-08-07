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
        f"FoundationPose not found at {FOUNDATIONPOSE_DIR}. "
        "Vendor or symlink the FoundationPose repo there."
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
        assert mask.dtype == bool or np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.floating), (
            f"Unexpected mask dtype: {mask.dtype}"
        )
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
    """Keeps a single FoundationPose instance alive: register on frame 0, track after."""

    def __init__(self, mesh_file: str | Path, debug_dir: str | Path = Path("debug_fp")):
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

        pose_cam = self.est.track_one(rgb=color_rgb, depth=depth_m, K=K, iteration=iteration)
        pose_cam = np.asarray(pose_cam, dtype=np.float64)
        assert pose_cam.shape == (4, 4), f"Expected 4x4 pose, got {pose_cam.shape}"
        self.last_pose_cam = pose_cam
        return pose_cam
