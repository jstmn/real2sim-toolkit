from pathlib import Path

import numpy as np
import pytest

from r2st.types import (
    CameraIntrinsics,
    ObjectAssets,
    ObjectPose,
    WorkspaceBounds,
)


class TestCameraIntrinsics:
    def test_values_and_matrix(self):
        ci = CameraIntrinsics(width=640, height=480, fx=600, fy=600, cx=320, cy=240)
        assert ci.values == (600, 600, 320, 240)
        np.testing.assert_allclose(ci.intrinsic_matrix, np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]))

    def test_from_matrix(self):
        m = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
        ci = CameraIntrinsics.from_matrix(m, 640, 480)
        assert ci.fx == 600 and ci.cx == 320

    def test_invalid_dimensions_raise(self):
        with pytest.raises(AssertionError):
            CameraIntrinsics(width=0, height=480, fx=600, fy=600, cx=320, cy=240)
        with pytest.raises(AssertionError):
            CameraIntrinsics(width=640, height=480, fx=-1, fy=600, cx=320, cy=240)


class TestWorkspaceBounds:
    def test_extents_and_center(self):
        wb = WorkspaceBounds(min_x_m=0, max_x_m=2, min_y_m=-1, max_y_m=1, min_z_m=0, max_z_m=1)
        assert wb.extents == (2, 2, 1)
        assert wb.center == (1, 0, 0.5)

    def test_invalid_bounds_raise(self):
        with pytest.raises(AssertionError):
            WorkspaceBounds(min_x_m=2, max_x_m=1, min_y_m=0, max_y_m=1, min_z_m=0, max_z_m=1)


class TestObjectAssets:
    def test_valid(self):
        mask = np.zeros((480, 640), dtype=bool)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        oa = ObjectAssets(object_name="red cup", mask=mask, image=img)
        assert oa.object_name == "red cup"

    def test_bad_mask_shape_raises(self):
        mask = np.zeros((480,), dtype=bool)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            ObjectAssets(object_name="x", mask=mask, image=img)  # type: ignore


class TestObjectPose:
    def test_valid(self):
        pose = np.eye(4)
        mask = np.zeros((10, 10), dtype=bool)
        op = ObjectPose(object_name="red cup", pose_cam=pose, pose_world=pose, mesh_path=Path("mesh.obj"), mask=mask)
        assert op.object_name == "red cup"

    def test_bad_pose_shape(self):
        with pytest.raises(AssertionError):
            ObjectPose(
                object_name="x",
                pose_cam=np.eye(3),
                pose_world=np.eye(4),
                mesh_path=Path("m"),
                mask=np.zeros((2, 2), dtype=bool),
            )
