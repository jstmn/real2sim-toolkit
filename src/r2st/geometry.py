import numpy as np

from r2st.types import CameraIntrinsics

# Depth -> Color extrinsics for RealSense D435 only.
REALSENSE_D435_DEPTH_TO_COLOR_ROTATION = np.array(
    [[0.999749, 0.021574, 0.00599926], [-0.021599, 0.999758, 0.0041374], [-0.00590855, -0.00426594, 0.999973]],
    dtype=np.float64,
)
REALSENSE_D435_DEPTH_TO_COLOR_TRANSLATION = np.array([0.0146080, -0.00004137, 0.0008026], dtype=np.float64)
MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 2.0


def scale_intrinsics(K: np.ndarray, orig_hw: tuple[int, int], new_hw: tuple[int, int]) -> np.ndarray:
    """Scale intrinsic matrix K from original (H, W) to new (H, W)."""
    assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
    assert len(orig_hw) == 2 and len(new_hw) == 2
    H0, W0 = orig_hw
    H1, W1 = new_hw
    assert H0 > 0 and W0 > 0 and H1 > 0 and W1 > 0
    sx = W1 / W0
    sy = H1 / H0
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled


def align_depth_to_color(
    depth_image: np.ndarray,
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
    R_dc: np.ndarray | None = None,
    t_dc: np.ndarray | None = None,
    invalid_fill: float = 0.0,
) -> np.ndarray:
    """Align depth to color using pinhole projection."""
    assert isinstance(depth_intrinsics, CameraIntrinsics), f"must be CameraIntrinsics, got {type(depth_intrinsics)}"
    assert isinstance(color_intrinsics, CameraIntrinsics), f"must be CameraIntrinsics, got {type(color_intrinsics)}"
    assert depth_image.ndim == 2, f"depth_image must be 2D, got {depth_image.ndim}D"
    Hd, Wd = depth_image.shape
    fx_d, fy_d, cx_d, cy_d = depth_intrinsics.values
    Hc, Wc = color_intrinsics.height, color_intrinsics.width
    fx_c, fy_c, cx_c, cy_c = color_intrinsics.values

    if R_dc is None:
        R_dc = np.eye(3, dtype=np.float64)
    if t_dc is None:
        t_dc = np.zeros(3, dtype=np.float64)
    assert R_dc.shape == (3, 3)
    assert t_dc.shape == (3,)

    u = np.arange(Wd, dtype=np.float64)
    v = np.arange(Hd, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    z = depth_image.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)

    x = (uu[valid] - cx_d) * z[valid] / fx_d
    y = (vv[valid] - cy_d) * z[valid] / fy_d
    pts_d = np.stack([x, y, z[valid]], axis=1)

    pts_c = (R_dc @ pts_d.T + t_dc.reshape(3, 1)).T
    xc, yc, zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
    pos = zc > 0
    xc, yc, zc = xc[pos], yc[pos], zc[pos]

    uc = fx_c * (xc / zc) + cx_c
    vc = fy_c * (yc / zc) + cy_c

    aligned = np.full((Hc, Wc), invalid_fill, dtype=np.float32)
    ui = np.rint(uc).astype(np.int32)
    vi = np.rint(vc).astype(np.int32)
    inside = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    ui, vi, zc = ui[inside], vi[inside], zc[inside]

    flat_idx = vi * Wc + ui
    zbuf = np.full(Hc * Wc, np.inf, dtype=np.float64)
    np.minimum.at(zbuf, flat_idx, zc)
    zbuf = zbuf.reshape(Hc, Wc)
    aligned[zbuf != np.inf] = zbuf[zbuf != np.inf].astype(np.float32)
    return aligned


def realsense_to_maniskill_basis_matrix() -> np.ndarray:
    """Basis transform RealSense optical (x right, y down, z forward) -> ManiSkill (x forward, y left, z up)."""
    return np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )


def transform_pose_cam_to_world(T_world_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    """Transform object pose from camera frame to world frame: T_world_obj = T_world_cam @ T_cam_obj."""
    assert T_world_cam.shape == (4, 4), f"T_world_cam must be 4x4, got {T_world_cam.shape}"
    assert T_cam_obj.shape == (4, 4), f"T_cam_obj must be 4x4, got {T_cam_obj.shape}"
    return T_world_cam @ T_cam_obj


def camera_extrinsic_to_maniskill_pose(T_world_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera extrinsic to ManiSkill camera pose (position, rotation)."""
    assert T_world_cam.shape == (4, 4)
    R_wc_rs = T_world_cam[:3, :3]
    t_wc = T_world_cam[:3, 3]
    F = realsense_to_maniskill_basis_matrix()
    R_wc_ms = R_wc_rs @ F.T
    return t_wc, R_wc_ms


def align_ros_depth_to_color(
    depth_raw: np.ndarray,
    depth_intrinsics: CameraIntrinsics,
    rgb_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Align ROS depth (uint16 mm) to color frame for RealSense D435 only."""
    assert depth_raw.ndim == 2
    depth_m = depth_raw.astype(np.float32) / 1000.0
    depth_aligned = align_depth_to_color(
        depth_m,
        depth_intrinsics=depth_intrinsics,
        color_intrinsics=rgb_intrinsics,
        R_dc=REALSENSE_D435_DEPTH_TO_COLOR_ROTATION,
        t_dc=REALSENSE_D435_DEPTH_TO_COLOR_TRANSLATION,
    )
    depth_aligned[(depth_aligned < MIN_DEPTH_M) | (depth_aligned > MAX_DEPTH_M)] = 0
    return depth_aligned


def depth_rgb_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    *,
    min_depth_m: float = MIN_DEPTH_M,
    max_depth_m: float = MAX_DEPTH_M,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a metric depth map to camera-frame points colored by RGB.

    Returns `(N, 3)` float32 XYZ and `(N, 3)` uint8 colors. Pixel coordinates use
    OpenCV convention (+Z forward, +Y down), matching FoundationPose poses.
    """
    assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
    assert rgb.shape[:2] == depth_m.shape, f"rgb {rgb.shape[:2]} != depth {depth_m.shape}"
    assert rgb.dtype == np.uint8, f"rgb dtype must be uint8, got {rgb.dtype}"
    assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
    assert stride >= 1, f"stride must be >= 1, got {stride}"
    assert min_depth_m > 0, f"min_depth_m must be > 0, got {min_depth_m}"
    assert max_depth_m > min_depth_m, f"max_depth_m must be > min_depth_m, got {max_depth_m} vs {min_depth_m}"
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    assert fx > 0 and fy > 0, f"fx/fy must be > 0, got fx={fx} fy={fy}"

    H, W = depth_m.shape
    us = np.arange(0, W, stride, dtype=np.float64)
    vs = np.arange(0, H, stride, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    ui = uu.astype(np.int64)
    vi = vv.astype(np.int64)
    z = depth_m[vi, ui].astype(np.float64)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    assert valid.any(), "No valid depth pixels to unproject"
    uu, vv, z = uu[valid], vv[valid], z[valid]
    pts = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=-1).astype(np.float32)
    colors = rgb[vi[valid], ui[valid]]
    return pts, colors


def project_axes_to_image(pose_cam: np.ndarray, K: np.ndarray, axis_len: float = 0.1):
    """Project 3D axes in camera frame to image plane."""
    assert pose_cam.shape == (4, 4)
    assert K.shape == (3, 3)
    assert axis_len > 0
    R = pose_cam[:3, :3]
    t = pose_cam[:3, 3]
    axes = np.stack(
        [np.zeros(3), np.array([axis_len, 0, 0]), np.array([0, axis_len, 0]), np.array([0, 0, axis_len])], axis=0
    )
    pts_cam = (R @ axes.T + t.reshape(3, 1)).T
    pts_img = []
    for p in pts_cam:
        if p[2] <= 0:
            return None
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        pts_img.append((int(round(u)), int(round(v))))
    return pts_img


def mat_to_sapien_pose_tuple(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 matrix to (position, quaternion wxyz). Requires transforms3d."""
    assert mat.shape == (4, 4)
    from transforms3d.quaternions import mat2quat

    t = mat[:3, 3]
    q_wxyz = mat2quat(mat[:3, :3])
    return t, q_wxyz
