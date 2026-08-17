import numpy as np

from r2st.constants import MAX_DEPTH_M, MIN_DEPTH_M
from r2st.types import CameraIntrinsics


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


def reproject_depth_to_color_frame(
    depth_image: np.ndarray,
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
    R_dc: np.ndarray,
    t_dc: np.ndarray,
    invalid_fill: float = 0.0,
) -> np.ndarray:
    """Unproject *native* (unaligned) metric depth, apply depth-to-color extrinsics, z-buffer into the color image.

    Do not use this on merged sensor-data h5 depth: that depth is already in the color
    pixel grid. Warping it again with depth-camera K (wider FOV) vs color K leaves a
    sparse, misaligned depth map.
    """
    assert isinstance(depth_intrinsics, CameraIntrinsics), f"must be CameraIntrinsics, got {type(depth_intrinsics)}"
    assert isinstance(color_intrinsics, CameraIntrinsics), f"must be CameraIntrinsics, got {type(color_intrinsics)}"
    assert depth_image.ndim == 2, f"depth_image must be 2D, got {depth_image.ndim}D"
    Hd, Wd = depth_image.shape
    fx_d, fy_d, cx_d, cy_d = depth_intrinsics.values
    Hc, Wc = color_intrinsics.height, color_intrinsics.width
    fx_c, fy_c, cx_c, cy_c = color_intrinsics.values
    assert R_dc.shape == (3, 3), f"R_dc must be 3x3, got {R_dc.shape}"
    assert t_dc.shape == (3,), f"t_dc must be (3,), got {t_dc.shape}"

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
    aligned[(aligned < MIN_DEPTH_M) | (aligned > MAX_DEPTH_M)] = 0
    return aligned


def depth_mm_to_meters(depth_raw: np.ndarray) -> np.ndarray:
    """Convert uint16 depth in millimeters to float32 meters."""
    assert depth_raw.dtype == np.uint16, f"depth must be uint16 millimeters, got {depth_raw.dtype}"
    return depth_raw.astype(np.float32) / 1000.0


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


def masked_depth_to_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    min_depth_m: float = MIN_DEPTH_M,
    max_depth_m: float = MAX_DEPTH_M,
) -> np.ndarray:
    """Unproject masked metric depth to camera-frame XYZ (N, 3) float64."""
    assert depth_m.ndim == 2, f"depth_m must be HxW, got {depth_m.shape}"
    assert mask.dtype == bool, f"mask dtype must be bool, got {mask.dtype}"
    assert mask.shape == depth_m.shape, f"mask {mask.shape} != depth {depth_m.shape}"
    assert mask.any(), "Cannot unproject an empty mask"
    assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
    assert min_depth_m > 0, f"min_depth_m must be > 0, got {min_depth_m}"
    assert max_depth_m > min_depth_m, f"max_depth_m must be > min_depth_m, got {max_depth_m} vs {min_depth_m}"
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    assert fx > 0 and fy > 0, f"fx/fy must be > 0, got fx={fx} fy={fy}"

    vs, us = np.where(mask)
    z = depth_m[vs, us].astype(np.float64)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    assert valid.any(), "No valid masked depth pixels to unproject"
    us = us[valid].astype(np.float64)
    vs = vs[valid].astype(np.float64)
    z = z[valid]
    return np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=-1)


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


_WARP_CHAMFER = None


def _warp_chamfer_min_sqdist_kernel():
    """Lazy-load Warp and the brute-force min-squared-distance kernel."""
    global _WARP_CHAMFER
    if _WARP_CHAMFER is not None:
        return _WARP_CHAMFER

    import warp as wp

    wp.init()

    @wp.kernel(enable_backward=False)
    def min_sqdist_kernel(
        src: wp.array(dtype=float, ndim=3),
        dst: wp.array(dtype=float, ndim=3),
        out: wp.array(dtype=float, ndim=2),
        n_dst: int,
    ):
        b, i = wp.tid()
        px = src[b, i, 0]
        py = src[b, i, 1]
        pz = src[b, i, 2]
        # Bare literals are Warp constants; wrap so min_d can update in the dynamic loop.
        min_d = float(1.0e30)  # noqa: UP018
        for j in range(n_dst):
            dx = px - dst[b, j, 0]
            dy = py - dst[b, j, 1]
            dz = pz - dst[b, j, 2]
            d = dx * dx + dy * dy + dz * dz
            min_d = min(min_d, d)
        out[b, i] = min_d

    _WARP_CHAMFER = (wp, min_sqdist_kernel)
    return _WARP_CHAMFER


def _as_batched_pointclouds(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return `(B, N, 3)`, `(B, M, 3)`, and whether the inputs were unbatched `(N, 3)` vs `(M, 3)`."""
    assert isinstance(points_a, np.ndarray), f"points_a must be ndarray, got {type(points_a)}"
    assert isinstance(points_b, np.ndarray), f"points_b must be ndarray, got {type(points_b)}"
    assert points_a.shape[-1] == 3, f"points_a last dim must be 3, got {points_a.shape}"
    assert points_b.shape[-1] == 3, f"points_b last dim must be 3, got {points_b.shape}"
    assert points_a.ndim in (2, 3), f"points_a must be (N, 3) or (B, N, 3), got {points_a.shape}"
    assert points_b.ndim in (2, 3), f"points_b must be (M, 3) or (B, M, 3), got {points_b.shape}"

    unbatched = points_a.ndim == 2 and points_b.ndim == 2
    if points_a.ndim == 2:
        points_a = points_a[None, ...]
    if points_b.ndim == 2:
        points_b = points_b[None, ...]
    if points_a.shape[0] == 1 and points_b.shape[0] > 1:
        points_a = np.broadcast_to(points_a, (points_b.shape[0], points_a.shape[1], 3))
    elif points_b.shape[0] == 1 and points_a.shape[0] > 1:
        points_b = np.broadcast_to(points_b, (points_a.shape[0], points_b.shape[1], 3))
    assert (
        points_a.shape[0] == points_b.shape[0]
    ), f"Batch sizes must match (or one side broadcast from 1), got {points_a.shape[0]} vs {points_b.shape[0]}"
    assert points_a.shape[1] >= 1, f"points_a must have at least 1 point, got {points_a.shape}"
    assert points_b.shape[1] >= 1, f"points_b must have at least 1 point, got {points_b.shape}"
    return (
        np.ascontiguousarray(points_a, dtype=np.float32),
        np.ascontiguousarray(points_b, dtype=np.float32),
        unbatched,
    )


def _mean_min_squared_nn(
    points_src: np.ndarray,
    points_dst: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Batched `(B, N, 3)` vs `(B, M, 3)` → `(B,)` mean over src of min squared distance to dst."""
    assert device in ("cpu", "cuda"), f"device must be 'cpu' or 'cuda', got {device!r}"
    assert points_src.ndim == 3 and points_src.shape[-1] == 3, f"points_src must be (B, N, 3), got {points_src.shape}"
    assert points_dst.ndim == 3 and points_dst.shape[-1] == 3, f"points_dst must be (B, M, 3), got {points_dst.shape}"
    assert (
        points_src.shape[0] == points_dst.shape[0]
    ), f"Batch sizes must match, got {points_src.shape[0]} vs {points_dst.shape[0]}"
    wp, min_sqdist_kernel = _warp_chamfer_min_sqdist_kernel()
    B, n_src, _ = points_src.shape
    n_dst = int(points_dst.shape[1])
    src_wp = wp.array(points_src, dtype=float, device=device)
    dst_wp = wp.array(points_dst, dtype=float, device=device)
    d = wp.zeros((B, n_src), dtype=float, device=device)
    wp.launch(min_sqdist_kernel, dim=[B, n_src], inputs=[src_wp, dst_wp, d, n_dst], device=device)
    out = np.asarray(d.numpy().mean(axis=1), dtype=np.float64)
    assert out.shape == (B,), f"Expected ({B},) distances, got {out.shape}"
    return out


def one_sided_squared_nn_distance(
    points_src: np.ndarray,
    points_dst: np.ndarray,
    *,
    device: str = "cuda",
) -> np.ndarray | float:
    """One-sided squared nearest-neighbor distance: ``mean_i min_j ||src_i - dst_j||^2``.

    Extra points in ``points_dst`` do not increase the cost. Shapes match ``chamfer_distance``.
    """
    batched_src, batched_dst, unbatched = _as_batched_pointclouds(points_src, points_dst)
    dist = _mean_min_squared_nn(batched_src, batched_dst, device=device)
    if unbatched:
        return float(dist[0])
    return dist


def chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    device: str = "cuda",
) -> np.ndarray | float:
    """Symmetric squared Chamfer distance, brute-force nearest neighbors via Warp.

    ``mean_i min_j ||a_i - b_j||^2 + mean_j min_i ||b_j - a_i||^2``.

    Shapes:
      - `(N, 3)` vs `(M, 3)` → Python `float`
      - `(B, N, 3)` vs `(B, M, 3)` → `(B,)` float64
      - `(B, N, 3)` vs `(M, 3)` or `(N, 3)` vs `(B, M, 3)` → broadcast the unbatched side
    """
    batched_a, batched_b, unbatched = _as_batched_pointclouds(points_a, points_b)
    dist = _mean_min_squared_nn(batched_a, batched_b, device=device) + _mean_min_squared_nn(
        batched_b, batched_a, device=device
    )
    if unbatched:
        return float(dist[0])
    return dist
