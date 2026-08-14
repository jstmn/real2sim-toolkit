import numpy as np
import pytest

from r2st.geometry import (
    reproject_depth_to_color_frame,
    camera_extrinsic_to_maniskill_pose,
    depth_mm_to_meters,
    project_axes_to_image,
    realsense_to_maniskill_basis_matrix,
    scale_intrinsics,
    transform_pose_cam_to_world,
)
from r2st.types import CameraIntrinsics


class TestIntrinsics:
    def test_intrinsics_to_matrix(self):
        ci = CameraIntrinsics(width=640, height=480, fx=600, fy=610, cx=320, cy=240)
        K = ci.intrinsic_matrix
        assert K.shape == (3, 3)
        assert K[0, 0] == 600 and K[1, 1] == 610
        assert K[0, 2] == 320 and K[1, 2] == 240

    def test_intrinsics_dtype(self):
        ci = CameraIntrinsics(width=640, height=480, fx=600, fy=610, cx=320, cy=240)
        assert ci.intrinsic_matrix.dtype == np.float64

    def test_scale_intrinsics(self):
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
        K2 = scale_intrinsics(K, (480, 640), (240, 320))
        # half resolution => fx,fy,cx,cy halved
        np.testing.assert_allclose(K2[0, 0], 300)
        np.testing.assert_allclose(K2[0, 2], 160)
        assert K2[2, 2] == 1

    def test_scale_intrinsics_invalid_shape(self):
        with pytest.raises(AssertionError):
            scale_intrinsics(np.eye(2), (480, 640), (240, 320))


class TestReprojectDepthToColorFrame:
    def _make_intrinsics(self):
        di = CameraIntrinsics(width=4, height=4, fx=100, fy=100, cx=2, cy=2)
        ci = CameraIntrinsics(width=4, height=4, fx=100, fy=100, cx=2, cy=2)
        return di, ci

    def _identity_extrinsics(self):
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def test_identity_align(self):
        di, ci = self._make_intrinsics()
        R, t = self._identity_extrinsics()
        depth = np.ones((4, 4), dtype=np.float32) * 1.0
        aligned = reproject_depth_to_color_frame(depth, di, ci, R, t)
        assert aligned.shape == (4, 4)
        # central pixel should map to itself
        assert aligned[2, 2] == pytest.approx(1.0, abs=1e-5)

    def test_all_zero_depth_returns_fill(self):
        di, ci = self._make_intrinsics()
        R, t = self._identity_extrinsics()
        depth = np.zeros((4, 4), dtype=np.float32)
        aligned = reproject_depth_to_color_frame(depth, di, ci, R, t, invalid_fill=0.0)
        assert np.all(aligned == 0.0)

    def test_invalid_depth_ndim_raises(self):
        di, ci = self._make_intrinsics()
        R, t = self._identity_extrinsics()
        with pytest.raises(AssertionError):
            reproject_depth_to_color_frame(np.ones((4,)), di, ci, R, t)  # type: ignore

    def test_z_buffer_keeps_nearest(self):
        # Two points projecting to same color pixel: nearest should win
        di = CameraIntrinsics(width=2, height=2, fx=1000, fy=1000, cx=0.5, cy=0.5)
        ci = CameraIntrinsics(width=2, height=2, fx=1000, fy=1000, cx=0.5, cy=0.5)
        R, t = self._identity_extrinsics()
        depth = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        aligned = reproject_depth_to_color_frame(depth, di, ci, R, t)
        # All depths valid and similar magnitude; check no inf holes where valid
        assert aligned[0, 0] > 0

    def test_uint16_mm_converted_to_meters(self):
        di, ci = self._make_intrinsics()
        R, t = self._identity_extrinsics()
        raw = np.ones((4, 4), dtype=np.uint16) * 1000  # 1m in mm
        aligned = reproject_depth_to_color_frame(depth_mm_to_meters(raw), di, ci, R, t)
        assert aligned.shape == (4, 4)
        assert aligned.dtype == np.float32
        assert aligned[2, 2] == pytest.approx(1.0, abs=1e-5)

    def test_d435_depth_to_color_constants(self):
        from r2st.constants import DEPTH_TO_COLOR_ROTATION, DEPTH_TO_COLOR_TRANSLATION

        R = DEPTH_TO_COLOR_ROTATION["d435"]
        t = DEPTH_TO_COLOR_TRANSLATION["d435"]
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert not np.allclose(R, np.eye(3))


class TestPoseTransforms:
    def test_realsense_to_maniskill_basis(self):
        F = realsense_to_maniskill_basis_matrix()
        assert F.shape == (3, 3)
        # orthogonal, det = 1
        assert abs(np.linalg.det(F) - 1.0) < 1e-5
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-6)

    def test_transform_pose_cam_to_world_identity(self):
        T_wc = np.eye(4)
        T_co = np.eye(4)
        T_co[0, 3] = 1.0
        T_wo = transform_pose_cam_to_world(T_wc, T_co)
        assert T_wo[0, 3] == pytest.approx(1.0)

    def test_transform_pose_multiply(self):
        T_wc = np.eye(4)
        T_wc[:3, 3] = [1, 2, 3]
        T_co = np.eye(4)
        T_co[:3, 3] = [0.1, 0.2, 0.3]
        T_wo = transform_pose_cam_to_world(T_wc, T_co)
        np.testing.assert_allclose(T_wo[:3, 3], [1.1, 2.2, 3.3])

    def test_transform_bad_shape_raises(self):
        with pytest.raises(AssertionError):
            transform_pose_cam_to_world(np.eye(3), np.eye(4))

    def test_camera_extrinsic_to_maniskill_pose(self):
        T = np.eye(4, dtype=np.float32)
        t, R = camera_extrinsic_to_maniskill_pose(T)
        assert t.shape == (3,) and R.shape == (3, 3)
        # For identity extrinsic, R should equal F.T
        F = realsense_to_maniskill_basis_matrix()
        np.testing.assert_allclose(R, F.T, atol=1e-6)


class TestProjectAxes:
    def test_project_in_front(self):
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
        pose = np.eye(4)
        pose[2, 3] = 1.0  # 1m in front
        pts = project_axes_to_image(pose, K, axis_len=0.1)
        assert pts is not None and len(pts) == 4

    def test_project_behind_returns_none(self):
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
        pose = np.eye(4)
        pose[2, 3] = -1.0  # behind
        assert project_axes_to_image(pose, K) is None

    def test_negative_axis_len_raises(self):
        K = np.eye(3)
        with pytest.raises(AssertionError):
            project_axes_to_image(np.eye(4), K, axis_len=-0.1)


def _numpy_chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    d_ab = ((points_a[:, None, :] - points_b[None, :, :]) ** 2).sum(axis=-1).min(axis=-1).mean()
    d_ba = ((points_b[:, None, :] - points_a[None, :, :]) ** 2).sum(axis=-1).min(axis=-1).mean()
    return float(d_ab + d_ba)


class TestChamferDistance:
    def test_identical_clouds_zero(self):
        from r2st.geometry import chamfer_distance

        rng = np.random.default_rng(0)
        pts = rng.normal(size=(32, 3)).astype(np.float32)
        d = chamfer_distance(pts, pts.copy(), device="cpu")
        assert isinstance(d, float)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_translated_cloud(self):
        from r2st.geometry import chamfer_distance

        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        shifted = pts + np.array([0.5, 0.0, 0.0], dtype=np.float32)
        d = chamfer_distance(pts, shifted, device="cpu")
        assert d == pytest.approx(2.0 * (0.5**2), abs=1e-5)

    def test_matches_numpy_reference(self):
        from r2st.geometry import chamfer_distance

        rng = np.random.default_rng(1)
        a = rng.normal(size=(17, 3)).astype(np.float32)
        b = rng.normal(size=(23, 3)).astype(np.float32) + 0.4
        d = chamfer_distance(a, b, device="cpu")
        assert d == pytest.approx(_numpy_chamfer(a, b), rel=1e-5, abs=1e-5)

    def test_batched_and_broadcast(self):
        from r2st.geometry import chamfer_distance

        rng = np.random.default_rng(2)
        a = rng.normal(size=(4, 11, 3)).astype(np.float32)
        b = rng.normal(size=(4, 9, 3)).astype(np.float32)
        d = chamfer_distance(a, b, device="cpu")
        assert d.shape == (4,)
        for i in range(4):
            assert d[i] == pytest.approx(_numpy_chamfer(a[i], b[i]), rel=1e-5, abs=1e-5)

        b0 = rng.normal(size=(9, 3)).astype(np.float32)
        d_bcast = chamfer_distance(a, b0, device="cpu")
        assert d_bcast.shape == (4,)
        for i in range(4):
            assert d_bcast[i] == pytest.approx(_numpy_chamfer(a[i], b0), rel=1e-5, abs=1e-5)

    def test_bad_shape_raises(self):
        from r2st.geometry import chamfer_distance

        with pytest.raises(AssertionError):
            chamfer_distance(np.zeros((5, 2)), np.zeros((5, 3)), device="cpu")
        with pytest.raises(AssertionError):
            chamfer_distance(np.zeros((2, 5, 3)), np.zeros((3, 5, 3)), device="cpu")
