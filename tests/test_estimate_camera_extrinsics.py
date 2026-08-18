import numpy as np
import pytest
from transforms3d.axangles import axangle2mat

from examples.estimate_camera_extrinsics import (
    _step_pose_world,
    _validate_seed_mode,
)


@pytest.mark.parametrize(("seed_automatically", "seed_from_gui"), [(True, False), (False, True)])
def test_exactly_one_seed_mode_is_valid(seed_automatically: bool, seed_from_gui: bool):
    _validate_seed_mode(seed_automatically, seed_from_gui)


@pytest.mark.parametrize(("seed_automatically", "seed_from_gui"), [(False, False), (True, True)])
def test_zero_or_two_seed_modes_raise(seed_automatically: bool, seed_from_gui: bool):
    with pytest.raises(AssertionError, match="Exactly one"):
        _validate_seed_mode(seed_automatically, seed_from_gui)


def test_world_translation_ignores_camera_orientation():
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    T_next = _step_pose_world(T, np.array([0.0, 0.0, 0.2]), np.zeros(3))
    assert np.allclose(T_next[:3, 3], [0.0, 0.0, 0.2])
    assert np.allclose(T_next[:3, :3], T[:3, :3])


def test_world_rotation_premultiplies_orientation():
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    T_next = _step_pose_world(T, np.zeros(3), np.array([0.0, 0.0, np.pi / 2.0]))
    expected_R = axangle2mat(np.array([0.0, 0.0, 1.0]), np.pi / 2.0) @ T[:3, :3]
    assert np.allclose(T_next[:3, :3], expected_R)
    assert np.allclose(T_next[:3, 3], np.zeros(3))
