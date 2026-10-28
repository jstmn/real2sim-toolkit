import numpy as np
import pytest

from r2st.mesh_scaling import (
    OBJECT_VOXEL_M,
    fit_plane_ransac,
    fit_table_plane,
    floor_points_and_voxelize,
    pca_uniform_scale,
    voxel_downsample,
)


def _table_points(rng: np.random.Generator, n: int, z: float = 0.0) -> np.ndarray:
    xy = rng.uniform(-0.25, 0.25, size=(n, 2))
    return np.column_stack([xy, z + rng.normal(0.0, 0.001, size=n)])


def _cube_surface(rng: np.random.Generator, side: float, n_per_face: int, faces: str) -> np.ndarray:
    """Points on the given faces ('+z', '-x', ...) of a cube resting on z=0, centered over the origin."""
    half = side / 2
    pts = []
    for face in faces.split():
        sign, axis = (1.0 if face[0] == "+" else -1.0), "xyz".index(face[1])
        p = rng.uniform(-half, half, size=(n_per_face, 3))
        p[:, axis] = sign * half
        pts.append(p)
    return np.concatenate(pts) + np.array([0.0, 0.0, half])


def test_fit_plane_ransac_ignores_outliers():
    rng = np.random.default_rng(0)
    plane = _table_points(rng, 2000, z=0.1)
    outliers = rng.uniform(-0.25, 0.25, size=(500, 3)) + np.array([0.0, 0.0, 0.3])
    normal, offset = fit_plane_ransac(np.concatenate([plane, outliers]), 0.004, 200, seed=0)
    assert abs(normal[2]) == pytest.approx(1.0, abs=1e-3)
    assert -offset / normal[2] == pytest.approx(0.1, abs=1e-3)


def test_fit_table_plane_normal_points_toward_object():
    rng = np.random.default_rng(0)
    object_pts = _cube_surface(rng, 0.06, 200, "+z +x")
    normal, offset = fit_table_plane(object_pts, _table_points(rng, 5000))
    assert normal[2] == pytest.approx(1.0, abs=1e-3)
    assert object_pts.mean(axis=0) @ normal + offset > 0


def test_voxel_downsample_keeps_one_centroid_per_voxel():
    pts = np.array([[0.001, 0.001, 0.001], [0.003, 0.003, 0.003], [0.011, 0.0, 0.0]])
    out = voxel_downsample(pts, 0.005)
    assert len(out) == 2
    assert any(np.allclose(p, [0.002, 0.002, 0.002]) for p in out)


def test_floor_points_recover_cube_size():
    # Only the top and two sides are visible, with the near side sampled densely, as from a camera
    rng = np.random.default_rng(0)
    side = 0.06
    seen = np.concatenate([_cube_surface(rng, side, 800, "+z +x"), _cube_surface(rng, side, 3000, "-y")])
    seen = seen[seen[:, 2] > 0.008]  # the depth rarely reaches the bottom edge
    mesh = _cube_surface(rng, 1.0, 1000, "+x -x +y -y +z -z")
    table = _table_points(rng, 5000)

    processed, summary = floor_points_and_voxelize(seen, table)
    assert "touching" in summary
    assert abs(side / pca_uniform_scale(mesh, seen) - 1) > 0.05
    assert pca_uniform_scale(mesh, processed) == pytest.approx(side, rel=0.05)


def test_floor_points_skipped_when_object_does_not_touch_table():
    rng = np.random.default_rng(0)
    lifted = _cube_surface(rng, 0.06, 500, "+z +x") + np.array([0.0, 0.0, 0.10])
    processed, summary = floor_points_and_voxelize(lifted, _table_points(rng, 5000))
    assert "not touching" in summary
    np.testing.assert_allclose(processed, voxel_downsample(lifted, OBJECT_VOXEL_M))
