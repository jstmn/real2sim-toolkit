import pathlib

import numpy as np
import pytest
import yaml
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat

from examples.estimate_camera_extrinsics import (
    _CAPTURE_AXES,
    _CAPTURE_DISTANCE_M,
    _axis_aligned_capture_poses,
)
from r2st.constants import CAMERA_MOUNT_RADIUS_M
from r2st.extrinsics import (
    SEED_LOOKAT_TARGET,
    SEED_UP,
    WORKSPACE_BOTTOM_PADDING_M,
    WORKSPACE_PADDING_M,
    compute_propagated_robot_masks,
    expand_demo_qpos_to_joint_names,
    look_at_opencv,
    look_at_with_roll,
    n_consecutive_cached_masks,
    per_frame_robot_mask_cache_path,
    per_frame_robot_mask_video_path,
    pose_from_xyz_wxyz,
    propagate_robot_masks,
    propagated_robot_mask_cache_path,
    propagated_robot_mask_video_path,
    remove_radius_outliers,
    robot_mask_overlay_path,
    step_pose_world,
    transform_points,
    validate_seed_mode,
    workspace_filtered_rgb_dir,
    world_points_to_camera,
    write_extrinsics_yaml,
)

_SEED_POSE = (0.80, 0.00, 0.50, 1.00, 0.00, 0.00, 0.00)


@pytest.mark.parametrize(
    ("seed_automatically", "seed_from_gui", "seed_pose"),
    [
        (True, False, None),
        (False, True, None),
        (False, False, _SEED_POSE),
    ],
)
def test_exactly_one_seed_mode_is_valid(
    seed_automatically: bool,
    seed_from_gui: bool,
    seed_pose: tuple[float, float, float, float, float, float, float] | None,
):
    validate_seed_mode(seed_automatically, seed_from_gui, seed_pose)


@pytest.mark.parametrize(
    ("seed_automatically", "seed_from_gui", "seed_pose"),
    [
        (False, False, None),
        (True, True, None),
        (True, False, _SEED_POSE),
        (False, True, _SEED_POSE),
        (True, True, _SEED_POSE),
    ],
)
def test_zero_or_two_seed_modes_raise(
    seed_automatically: bool,
    seed_from_gui: bool,
    seed_pose: tuple[float, float, float, float, float, float, float] | None,
):
    with pytest.raises(AssertionError, match="Exactly one"):
        validate_seed_mode(seed_automatically, seed_from_gui, seed_pose)


def test_pose_from_xyz_wxyz_roundtrip():
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.1, -0.2, 0.3]
    T[:3, :3] = axangle2mat(np.array([0.0, 0.0, 1.0]), 0.4)
    q = mat2quat(T[:3, :3])
    T_back = pose_from_xyz_wxyz(np.concatenate([T[:3, 3], q]))
    assert np.allclose(T_back, T)


def test_pose_from_xyz_wxyz_rejects_zero_quaternion():
    with pytest.raises(AssertionError, match="zero"):
        pose_from_xyz_wxyz((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_pose_from_xyz_wxyz_rejects_nonunit_quaternion():
    with pytest.raises(AssertionError, match="unit"):
        pose_from_xyz_wxyz((0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0))


def test_world_translation_ignores_camera_orientation():
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    T_next = step_pose_world(T, np.array([0.0, 0.0, 0.2]), np.zeros(3))
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
    T_next = step_pose_world(T, np.zeros(3), np.array([0.0, 0.0, np.pi / 2.0]))
    expected_R = axangle2mat(np.array([0.0, 0.0, 1.0]), np.pi / 2.0) @ T[:3, :3]
    assert np.allclose(T_next[:3, :3], expected_R)
    assert np.allclose(T_next[:3, 3], np.zeros(3))


def test_axis_aligned_capture_poses_look_at_center():
    lookat = SEED_LOOKAT_TARGET
    distance_m = _CAPTURE_DISTANCE_M
    poses = _axis_aligned_capture_poses(lookat, distance_m)
    assert len(poses) == 6
    eyes = np.stack([T[:3, 3] for T in poses], axis=0)
    expected_eyes = lookat[None, :] + distance_m * _CAPTURE_AXES
    assert np.allclose(eyes, expected_eyes)
    for T, axis in zip(poses, _CAPTURE_AXES, strict=True):
        look = lookat - T[:3, 3]
        look_u = look / np.linalg.norm(look)
        assert np.allclose(T[:3, 2], look_u)
        assert np.allclose(axis / np.linalg.norm(axis), -look_u)


def test_look_at_roll_pi_flips_camera_x_and_y():
    eye = np.array([0.80, 0.00, 0.50], dtype=np.float64)
    T0 = look_at_opencv(eye, SEED_LOOKAT_TARGET, SEED_UP)
    T = look_at_with_roll(eye, SEED_LOOKAT_TARGET, np.pi, SEED_UP)
    assert np.allclose(T[:3, 3], T0[:3, 3])
    assert np.allclose(T[:3, 2], T0[:3, 2])
    assert np.allclose(T[:3, 0], -T0[:3, 0])
    assert np.allclose(T[:3, 1], -T0[:3, 1])


def test_remove_radius_outliers_drops_isolated_points():
    xs, ys, zs = np.mgrid[0:5, 0:5, 0:2]
    cluster = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float64) * 0.01
    isolated = np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64)
    out = remove_radius_outliers(np.vstack([cluster, isolated]), nb_points=10, radius_m=0.10)
    assert out.shape[0] == 50
    assert np.all(np.linalg.norm(out, axis=1) < 0.2)


def test_remove_radius_outliers_drops_all_isolated_points():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    out = remove_radius_outliers(pts, nb_points=10, radius_m=0.10)
    assert out.shape == (0, 3)


def test_world_points_to_camera_inverts_transform_points():
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.1, -0.2, 0.3]
    T[:3, :3] = axangle2mat(np.array([0.0, 0.0, 1.0]), 0.4)
    pts_cam = np.array([[0.0, 0.0, 1.0], [0.05, -0.1, 0.8]], dtype=np.float64)
    pts_world = transform_points(T, pts_cam)
    pts_cam_back = world_points_to_camera(T, pts_world)
    assert np.allclose(pts_cam_back, pts_cam, atol=1.0e-5)


class _FakeLimit:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper


class _FakeJoint:
    def __init__(self, name: str, lower: float = 0.0, upper: float = 0.85, mimic: object | None = None):
        self.name = name
        self.limit = _FakeLimit(lower, upper)
        self.mimic = mimic


class _FakeRobot:
    def __init__(self, joints: list[_FakeJoint]):
        self.name = "xarm7__gripper"
        self.actuated_joints = joints
        self.actuated_joint_names = [j.name for j in joints]


def _fake_xarm7__gripper() -> _FakeRobot:
    mimic = object()
    return _FakeRobot(
        [
            _FakeJoint("joint1", lower=-6.28, upper=6.28),
            _FakeJoint("joint2", lower=-2.06, upper=2.09),
            _FakeJoint("joint3", lower=-6.28, upper=6.28),
            _FakeJoint("joint4", lower=-0.19, upper=3.75),
            _FakeJoint("joint5", lower=-6.28, upper=6.28),
            _FakeJoint("joint6", lower=-1.69, upper=3.14),
            _FakeJoint("joint7", lower=-6.28, upper=6.28),
            _FakeJoint("drive_joint"),
            _FakeJoint("left_finger_joint", mimic=mimic),
            _FakeJoint("left_inner_knuckle_joint", mimic=mimic),
            _FakeJoint("right_outer_knuckle_joint", mimic=mimic),
            _FakeJoint("right_finger_joint", mimic=mimic),
            _FakeJoint("right_inner_knuckle_joint", mimic=mimic),
        ]
    )


def test_expand_demo_qpos_pads_eef_and_drops_mimics():
    robot = _fake_xarm7__gripper()
    qpos = np.arange(14, dtype=np.float64).reshape(2, 7)
    sapien_names = robot.actuated_joint_names[:8]
    out = expand_demo_qpos_to_joint_names(qpos, robot, sapien_names)
    assert out.shape == (2, 8)
    assert np.allclose(out[:, :7], qpos)
    assert np.allclose(out[:, 7], 0.0)


def test_expand_demo_qpos_identity_when_dims_match():
    robot = _fake_xarm7__gripper()
    names = robot.actuated_joint_names[:7]
    robot7 = _FakeRobot(robot.actuated_joints[:7])
    qpos = np.ones((3, 7), dtype=np.float64)
    out = expand_demo_qpos_to_joint_names(qpos, robot7, names)
    assert np.allclose(out, qpos)


def test_expand_demo_qpos_rejects_too_many_demo_joints():
    robot = _FakeRobot([_FakeJoint("joint1")])
    with pytest.raises(AssertionError, match=">"):
        expand_demo_qpos_to_joint_names(np.zeros((2, 2)), robot, ["joint1"])


def test_expand_demo_qpos_rejects_unknown_sapien_joint():
    robot = _fake_xarm7__gripper()
    with pytest.raises(AssertionError, match="not in Jrl2"):
        expand_demo_qpos_to_joint_names(np.zeros((1, 7)), robot, ["joint1", "not_a_joint"])


def test_expand_demo_qpos_rejects_unused_non_mimic_joint():
    robot = _FakeRobot([_FakeJoint("joint1"), _FakeJoint("joint2")])
    with pytest.raises(AssertionError, match="has no mimic"):
        expand_demo_qpos_to_joint_names(np.zeros((1, 1)), robot, ["joint1"])


def test_expand_demo_qpos_rejects_pin_outside_limits():
    robot = _FakeRobot(
        [
            _FakeJoint("joint1", lower=-1.0, upper=1.0),
            _FakeJoint("finger", lower=0.1, upper=0.5),
        ]
    )
    with pytest.raises(AssertionError, match="Cannot pin EEF joint"):
        expand_demo_qpos_to_joint_names(np.zeros((1, 1)), robot, ["joint1", "finger"])


class _FakePropagatePredictor:
    def __init__(self):
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def propagate_from_mask(self, image_bgr, mask_input, box_xyxy):
        self.calls.append((np.asarray(image_bgr).copy(), np.asarray(mask_input).copy(), np.asarray(box_xyxy).copy()))
        t = len(self.calls)
        height, width = image_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[2 : 2 + t, 3 : 5 + t] = True
        low_res = np.full((1, 288, 288), float(t), dtype=np.float32)
        return mask, 0.9, low_res


def test_propagate_robot_masks_feeds_union_then_logits():
    from r2st.core import bbox_xyxy_from_mask, binary_mask_to_sam_mask_input

    rgb = np.zeros((3, 16, 20, 3), dtype=np.uint8)
    rgb[1] = 1
    rgb[2] = 2
    seed = np.zeros((16, 20), dtype=bool)
    seed[4:8, 6:11] = True
    predictor = _FakePropagatePredictor()
    out = propagate_robot_masks(rgb, predictor, seed)
    assert out.shape == (3, 16, 20)
    assert np.array_equal(out[0], seed)
    assert len(predictor.calls) == 2
    _, mask_input0, box0 = predictor.calls[0]
    assert np.allclose(mask_input0, binary_mask_to_sam_mask_input(seed))
    assert np.allclose(box0, bbox_xyxy_from_mask(seed))
    expected1 = np.zeros((16, 20), dtype=bool)
    expected1[2:3, 3:6] = True
    expected2 = np.zeros((16, 20), dtype=bool)
    expected2[2:4, 3:7] = True
    assert np.array_equal(out[1], expected1)
    assert np.array_equal(out[2], expected2)
    _, mask_input1, box1 = predictor.calls[1]
    assert np.allclose(mask_input1, np.full((1, 288, 288), 1.0, dtype=np.float32))
    assert np.allclose(box1, bbox_xyxy_from_mask(expected1))


def test_propagate_robot_masks_rejects_empty_seed():
    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(AssertionError, match="empty"):
        propagate_robot_masks(rgb, _FakePropagatePredictor(), np.zeros((8, 8), dtype=bool))


def test_propagate_robot_masks_writes_cache_each_frame(tmp_path):
    rgb = np.zeros((3, 16, 20, 3), dtype=np.uint8)
    seed = np.zeros((16, 20), dtype=bool)
    seed[4:8, 6:11] = True
    cache_paths = [tmp_path / f"mask_{t}.npy" for t in range(3)]
    out = propagate_robot_masks(rgb, _FakePropagatePredictor(), seed, cache_paths=cache_paths)
    for t, path in enumerate(cache_paths):
        assert path.is_file(), f"missing cache {path}"
        assert np.array_equal(np.load(path), out[t])


def test_n_consecutive_cached_masks_stops_at_first_gap(tmp_path):
    paths = [tmp_path / f"mask_{t}.npy" for t in range(4)]
    for t in (0, 1, 3):
        (tmp_path / f"mask_{t}.npy").write_bytes(b"x")
    assert n_consecutive_cached_masks(paths) == 2
    assert n_consecutive_cached_masks([]) == 0
    assert n_consecutive_cached_masks([tmp_path / "missing.npy"]) == 0


def test_robot_mask_cache_layout():
    h5_path = pathlib.Path("/d/traj_0.h5")
    prop = pathlib.Path("/d/traj_0/default/propagated_mask/cam_1__kmax=5__score_threshold=0.3")
    npy = propagated_robot_mask_cache_path(h5_path, "cam_1", 42, 5, 0.3, None)
    assert npy == prop / "npy/00042.npy"
    assert robot_mask_overlay_path(npy) == prop / "rgb/00042.png"
    assert propagated_robot_mask_video_path(h5_path, "cam_1", 5, 0.3, None) == prop / "masks.mp4"
    per_frame = pathlib.Path(
        f"/d/traj_0/workspace_filtered__pad={WORKSPACE_PADDING_M}__bottompad={WORKSPACE_BOTTOM_PADDING_M}"
        f"__mountradius={CAMERA_MOUNT_RADIUS_M}"
        "/per_frame_mask/cam_1__ws__prompt=robot_arm__kmax=5__score_threshold=0.3"
    )
    assert (
        per_frame_robot_mask_cache_path(h5_path, "cam_1", "robot arm", 7, 5, 0.3, "ws") == per_frame / "npy/00007.npy"
    )
    assert per_frame_robot_mask_video_path(h5_path, "cam_1", "robot arm", 5, 0.3, "ws") == per_frame / "masks.mp4"
    assert workspace_filtered_rgb_dir(h5_path, "cam_1", "ws") == per_frame.parent.parent / "blacked_out_rgb/cam_1__ws"


def test_compute_propagated_robot_masks_loads_full_cache(tmp_path):
    n_frames, height, width = 3, 8, 10
    rgb = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    h5_path = tmp_path / "merged_sensor_data.h5"
    expected = []
    for t in range(n_frames):
        mask = np.zeros((height, width), dtype=bool)
        mask[1:3, 1 + t : 4 + t] = True
        expected.append(mask)
        path = propagated_robot_mask_cache_path(h5_path, "cam_1", t, 5, 0.3, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)
    out = compute_propagated_robot_masks(rgb, h5_path, "cam_1", "robot arm", 5, 0.3, True, False, None)
    assert np.array_equal(out, np.stack(expected, axis=0))
    for t in range(n_frames):
        overlay = robot_mask_overlay_path(propagated_robot_mask_cache_path(h5_path, "cam_1", t, 5, 0.3, None))
        assert overlay.is_file(), f"missing demo overlay {overlay}"
    video_path = propagated_robot_mask_video_path(h5_path, "cam_1", 5, 0.3, None)
    assert video_path.is_file(), f"missing robot-mask video {video_path}"
    assert video_path.stat().st_size > 0, f"empty robot-mask video {video_path}"


def test_compute_propagated_robot_masks_resumes_from_partial_cache(tmp_path, monkeypatch):
    import r2st.extrinsics as ece

    n_frames, height, width = 3, 8, 10
    rgb = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    h5_path = tmp_path / "merged_sensor_data.h5"
    cached = []
    for t in range(2):
        mask = np.zeros((height, width), dtype=bool)
        mask[1:3, 1 + t : 4 + t] = True
        cached.append(mask)
        path = propagated_robot_mask_cache_path(h5_path, "cam_1", t, 5, 0.3, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)

    fake = _FakePropagatePredictor()
    monkeypatch.setattr(ece, "load_sam3", lambda: fake)
    out = ece.compute_propagated_robot_masks(rgb, h5_path, "cam_1", "robot arm", 5, 0.3, True, False, None)
    assert np.array_equal(out[0], cached[0])
    assert np.array_equal(out[1], cached[1])
    assert len(fake.calls) == 1
    expected2 = np.zeros((height, width), dtype=bool)
    expected2[2:3, 3:6] = True
    assert np.array_equal(out[2], expected2)
    path2 = propagated_robot_mask_cache_path(h5_path, "cam_1", 2, 5, 0.3, None)
    assert path2.is_file()
    assert np.array_equal(np.load(path2), expected2)


def test_in_workspace_pixels_drops_points_outside_bounds_and_missing_depth():
    from r2st.extrinsics import in_workspace_pixels
    from r2st.types import WorkspaceBounds

    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    depth_m = np.full((3, 5), 1.0)
    depth_m[0, 0] = 3.0  # behind the workspace
    depth_m[2, 4] = np.nan  # no depth
    depth_m[1, 2] = 0.0  # no depth
    # Camera at the origin looking along +z, so camera points are robot-base points
    bounds = WorkspaceBounds(min_x_m=-1.0, max_x_m=1.0, min_y_m=-1.0, max_y_m=1.0, min_z_m=0.5, max_z_m=1.5)
    keep = in_workspace_pixels(depth_m, K, np.eye(4), bounds)
    expected = np.ones((3, 5), dtype=bool)
    expected[0, 0] = False
    expected[2, 4] = False
    expected[1, 2] = False
    assert np.array_equal(keep, expected)


def test_workspace_mask_tag_changes_with_prior_content(tmp_path):
    from r2st.extrinsics import workspace_mask_tag

    prior = tmp_path / "extrinsics_cam.yaml"
    prior.write_text("matrix: [1]\n")
    tag_a = workspace_mask_tag("vention-franka-table", prior)
    assert tag_a.startswith("workspace=vention-franka-table-")
    assert "__prior=extrinsics_cam-" in tag_a
    prior.write_text("matrix: [2]\n")
    assert workspace_mask_tag("vention-franka-table", prior) != tag_a


def test_workspace_bounds_padded_contains():
    from r2st.constants import get_workspace_bounds

    bounds = get_workspace_bounds("vention-franka-table")
    padded = bounds.padded(0.10)
    pts = np.array([[0.0, 0.0, 0.3], [1.05, 0.0, 0.3], [1.2, 0.0, 0.3]])
    assert bounds.contains(pts).tolist() == [True, False, False]
    assert padded.contains(pts).tolist() == [True, True, False]


def test_padded_workspace_bounds_pads_the_bottom_less():
    from r2st.constants import get_workspace_bounds
    from r2st.extrinsics import (
        WORKSPACE_BOTTOM_PADDING_M,
        WORKSPACE_PADDING_M,
        padded_workspace_bounds,
    )

    bounds = get_workspace_bounds("vention-franka-table")
    padded = padded_workspace_bounds("vention-franka-table")
    assert padded.min_z_m == pytest.approx(-0.06)
    assert padded.min_z_m == bounds.min_z_m - WORKSPACE_BOTTOM_PADDING_M
    assert padded.max_z_m == bounds.max_z_m + WORKSPACE_PADDING_M
    assert padded.min_y_m == bounds.min_y_m - WORKSPACE_PADDING_M


def test_frames_with_robot_mask_skips_empty_and_tiny_per_frame_masks():
    from r2st.extrinsics import MIN_ROBOT_MASK_PIXELS, frames_with_robot_mask

    masks = np.zeros((4, 40, 40), dtype=bool)
    masks[0, :25, :25] = True  # 625 px
    masks[2].flat[:MIN_ROBOT_MASK_PIXELS] = True  # tiny
    masks[3, :30, :30] = True
    has_mask, no_mask = frames_with_robot_mask(masks, per_frame_robot_masks=True)
    assert has_mask.tolist() == [0, 3]
    assert no_mask.tolist() == [1, 2]
    with pytest.raises(AssertionError, match="Propagated robot mask is empty"):
        frames_with_robot_mask(masks, per_frame_robot_masks=False)
    masks[1, :25, :25] = True
    with pytest.raises(AssertionError, match="lost tracking"):
        frames_with_robot_mask(masks, per_frame_robot_masks=False)


def test_write_extrinsics_yaml_backs_up_previous_file(tmp_path):
    out = tmp_path / "extrinsics_cam.yaml"
    write_extrinsics_yaml(out, {"cost": 1.0})
    assert not (tmp_path / "extrinsics_cam.backup.yaml").exists()
    write_extrinsics_yaml(out, {"cost": 2.0})
    assert yaml.safe_load((tmp_path / "extrinsics_cam.backup.yaml").read_text()) == {"cost": 1.0}
    assert yaml.safe_load(out.read_text()) == {"cost": 2.0}
