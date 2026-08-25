import numpy as np
import pytest
import yaml

from examples.track_object import _load_camera_extrinsics_yaml
from r2st.utils import _transform_points_se3


def _write_extrinsics(path, *, camera="cam_1", camera_model_id="d435", matrix=None):
    if matrix is None:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = [0.7, -0.7, 0.6]
    payload = {
        "camera": camera,
        "camera_model_id": camera_model_id,
        "matrix": np.asarray(matrix).tolist(),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_load_camera_extrinsics_yaml(tmp_path):
    path = tmp_path / "extrinsics.yaml"
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.71, -0.75, 0.64]
    _write_extrinsics(path, matrix=T)
    T_loaded = _load_camera_extrinsics_yaml(path, "cam_1", "d435")
    assert np.allclose(T_loaded, T)


def test_load_camera_extrinsics_yaml_rejects_camera_mismatch(tmp_path):
    path = tmp_path / "extrinsics.yaml"
    _write_extrinsics(path, camera="cam_2")
    with pytest.raises(AssertionError, match="camera"):
        _load_camera_extrinsics_yaml(path, "cam_1", "d435")


def test_load_camera_extrinsics_yaml_rejects_model_mismatch(tmp_path):
    path = tmp_path / "extrinsics.yaml"
    _write_extrinsics(path, camera_model_id="d455")
    with pytest.raises(AssertionError, match="camera_model_id"):
        _load_camera_extrinsics_yaml(path, "cam_1", "d435")


def test_load_camera_extrinsics_yaml_rejects_missing_file(tmp_path):
    with pytest.raises(AssertionError, match="not found"):
        _load_camera_extrinsics_yaml(tmp_path / "missing.yaml", "cam_1", "d435")


def test_transform_points_se3_translates():
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1.0, 2.0, 3.0]
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    out = _transform_points_se3(T, pts)
    assert np.allclose(out, [[1.0, 2.0, 3.0], [1.1, 2.0, 3.0]])
