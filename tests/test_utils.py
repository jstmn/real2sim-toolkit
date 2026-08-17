import numpy as np
import pytest

from r2st.utils import farthest_point_sample_naive, farthest_point_sample_pyg_lib


def _circular_dist_sq(points: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Min squared circular distance from each point to the already-selected set."""
    delta = points[:, None, :] - selected[None, :, :]
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return np.sum(wrapped * wrapped, axis=-1).min(axis=-1)


class TestFarthestPointSampleNaive:
    def test_single_sample_is_start_index(self):
        pts = np.arange(10, dtype=np.float64).reshape(10, 1)
        idx = farthest_point_sample_naive(pts, 1, start_index=3, distance="euclidean")
        assert idx.shape == (1,)
        assert idx.dtype == np.int64
        assert idx[0] == 3

    def test_samples_all_unique_indices(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(20, 3))
        idx = farthest_point_sample_naive(pts, 20, start_index=0, distance="euclidean")
        assert idx.shape == (20,)
        assert len(np.unique(idx)) == 20
        np.testing.assert_array_equal(np.sort(idx), np.arange(20))

    def test_line_picks_endpoints_then_middle_euclidean(self):
        pts = np.stack([np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)], axis=1)
        idx = farthest_point_sample_naive(pts, 3, start_index=0, distance="euclidean")
        np.testing.assert_array_equal(idx, np.array([0, 9, 4]))

    def test_wraparound_euclidean_vs_circular(self):
        # 0 and 2π-0.1 are almost 2π apart in R, but only 0.1 apart on S¹.
        pts = np.array([[0.0], [0.1], [np.pi], [2.0 * np.pi - 0.1]])
        idx_euc = farthest_point_sample_naive(pts, 2, start_index=0, distance="euclidean")
        idx_circ = farthest_point_sample_naive(pts, 2, start_index=0, distance="circular")
        np.testing.assert_array_equal(idx_euc, np.array([0, 3]))
        np.testing.assert_array_equal(idx_circ, np.array([0, 2]))

    def test_circular_treats_zero_and_two_pi_as_equal(self):
        pts = np.array([[0.0], [2.0 * np.pi], [np.pi]])
        idx = farthest_point_sample_naive(pts, 2, start_index=0, distance="circular")
        np.testing.assert_array_equal(idx, np.array([0, 2]))

    def test_joint_angle_feature_dim_circular(self):
        rng = np.random.default_rng(1)
        qpos = rng.uniform(low=-np.pi, high=np.pi, size=(50, 7))
        idx = farthest_point_sample_naive(qpos, 8, start_index=0, distance="circular")
        assert idx.shape == (8,)
        assert len(np.unique(idx)) == 8
        sampled = qpos[idx]
        assert sampled.shape == (8, 7)

    def test_batched_independent_euclidean(self):
        line = np.stack([np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)], axis=1)
        pts = np.stack([line, line[::-1].copy()], axis=0)
        idx = farthest_point_sample_naive(pts, 3, start_index=0, distance="euclidean")
        assert idx.shape == (2, 3)
        np.testing.assert_array_equal(idx[0], np.array([0, 9, 4]))
        np.testing.assert_array_equal(idx[1], np.array([0, 9, 4]))

    def test_greedy_picks_farthest_from_selected_set_euclidean(self):
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(40, 3))
        idx = farthest_point_sample_naive(pts, 6, start_index=0, distance="euclidean")
        selected = pts[idx]
        for k in range(1, len(idx)):
            min_dist_sq_to_selected = ((pts[:, None, :] - selected[None, :k, :]) ** 2).sum(axis=-1).min(axis=-1)
            min_dist_sq_to_selected[idx[:k]] = -np.inf
            expected = int(np.argmax(min_dist_sq_to_selected))
            assert idx[k] == expected

    def test_greedy_picks_farthest_from_selected_set_circular(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(low=0.0, high=2.0 * np.pi, size=(40, 2))
        idx = farthest_point_sample_naive(pts, 6, start_index=0, distance="circular")
        selected = pts[idx]
        for k in range(1, len(idx)):
            min_dist_sq_to_selected = _circular_dist_sq(pts, selected[:k])
            min_dist_sq_to_selected[idx[:k]] = -np.inf
            expected = int(np.argmax(min_dist_sq_to_selected))
            assert idx[k] == expected

    def test_bad_args_raise(self):
        pts = np.zeros((5, 3))
        with pytest.raises(AssertionError):
            farthest_point_sample_naive(pts, 0)
        with pytest.raises(AssertionError):
            farthest_point_sample_naive(pts, 6)
        with pytest.raises(AssertionError):
            farthest_point_sample_naive(pts, 2, start_index=5)
        with pytest.raises(AssertionError):
            farthest_point_sample_naive(np.zeros((5,)), 2)
        with pytest.raises(AssertionError):
            farthest_point_sample_naive(pts, 2, distance="angular")


def _torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class TestFarthestPointSamplePygLib:
    def test_returns_unique_indices_of_requested_count(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(200, 3)).astype(np.float32)
        idx = farthest_point_sample_pyg_lib(pts, 32, device=_torch_device(), random_start=False)
        assert idx.shape == (32,)
        assert idx.dtype == np.int64
        assert idx.min() >= 0
        assert idx.max() < 200
        assert len(np.unique(idx)) == 32

    def test_batched_returns_per_cloud_indices(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(3, 80, 3)).astype(np.float32)
        idx = farthest_point_sample_pyg_lib(pts, 10, device=_torch_device(), random_start=False)
        assert idx.shape == (3, 10)
        assert idx.dtype == np.int64
        assert idx.min() >= 0
        assert idx.max() < 80
        for b in range(3):
            assert len(np.unique(idx[b])) == 10

    def test_deterministic_when_random_start_false(self):
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(100, 3)).astype(np.float32)
        device = _torch_device()
        idx0 = farthest_point_sample_pyg_lib(pts, 16, device=device, random_start=False)
        idx1 = farthest_point_sample_pyg_lib(pts, 16, device=device, random_start=False)
        np.testing.assert_array_equal(idx0, idx1)

    def test_bad_args_raise(self):
        pts = np.zeros((5, 3), dtype=np.float32)
        device = _torch_device()
        with pytest.raises(AssertionError):
            farthest_point_sample_pyg_lib(pts, 0, device=device)
        with pytest.raises(AssertionError):
            farthest_point_sample_pyg_lib(pts, 6, device=device)
        with pytest.raises(AssertionError):
            farthest_point_sample_pyg_lib(np.zeros((5,), dtype=np.float32), 2, device=device)
        with pytest.raises(AssertionError):
            farthest_point_sample_pyg_lib(pts, 2, device="")
