"""6-DoF Kalman filter for smoothing FoundationPose tracking output.

Ported from FoundationPose++ (https://github.com/lidingsheng/FoundationPose-plus-plus),
``src/utils/kalman_filter_6d.py``. The state is a 12-vector of
``[tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz]`` (translation +
Euler angles, plus their velocities). Two measurement models are supported: a
full 6-DoF pose (``update``) and a 2D image-plane point already converted to
camera-frame ``(x, y)`` (``update_from_xy``), which is what a 2D tracker (e.g.
Cutie) supplies each frame.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

_NDIM = 6
_DT = 1.0


class KalmanFilter6D:
    def __init__(self, measurement_noise_scale: float):
        assert measurement_noise_scale > 0, f"measurement_noise_scale must be > 0, got {measurement_noise_scale}"
        self.measurement_noise_scale = measurement_noise_scale

        self._motion_mat = np.eye(2 * _NDIM, 2 * _NDIM)
        for i in range(_NDIM):
            self._motion_mat[i, _NDIM + i] = _DT
        self._update_mat = np.eye(_NDIM, 2 * _NDIM)
        self._update_mat_xy = np.zeros((2, 2 * _NDIM))
        self._update_mat_xy[0, 0] = 1
        self._update_mat_xy[1, 1] = 1

        self._std_weight_trans_xy = 1.0 / 40  # xy sensor (2D tracker) has a lower noise floor
        self._std_weight_trans = 1.0 / 10
        self._std_weight_rot = 1.0 / 20
        self._std_weight_vel_trans = 1.0 / 20
        self._std_weight_vel_rot = 1.0 / 40

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initialize a track from an unassociated 6-DoF pose measurement."""
        assert measurement.shape == (6,), f"measurement must be a 6-vector, got {measurement.shape}"
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        scale_xyz = max(np.linalg.norm(measurement[:3]), 1e-5)
        scale_rot = max(np.linalg.norm(measurement[3:]), 1e-5)

        std = [
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the state by one time step (no measurement)."""
        scale_xyz = mean[2]
        scale_rot = mean[5]

        std_pos = [
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
        ]
        std_vel = [
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state into full 6-DoF pose measurement space."""
        scale_xyz = mean[2]
        scale_rot = mean[5]

        std = [
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return projected_mean, projected_cov

    def project_for_xy(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state into 2D (x, y) measurement space."""
        scale_xy = max(np.linalg.norm(mean[:2]), 1e-5)
        std_xy = [self._std_weight_trans_xy * scale_xy, self._std_weight_trans_xy * scale_xy]
        innovation_cov = np.diag(np.square(std_xy))
        projected_mean = np.dot(self._update_mat_xy, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat_xy, covariance, self._update_mat_xy.T)) + innovation_cov
        return projected_mean, projected_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with a full 6-DoF pose measurement."""
        assert measurement.shape == (6,), f"measurement must be a 6-vector, got {measurement.shape}"
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def update_from_xy(
        self, mean: np.ndarray, covariance: np.ndarray, measurement_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with a camera-frame (x, y) measurement (e.g. from a 2D tracker)."""
        assert measurement_xy.shape == (2,), f"measurement_xy must be a 2-vector, got {measurement_xy.shape}"
        projected_mean, projected_cov = self.project_for_xy(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat_xy.T).T, check_finite=False
        ).T

        innovation = measurement_xy - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
