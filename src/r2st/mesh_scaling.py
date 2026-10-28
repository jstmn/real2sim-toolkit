"""Uniform mesh scale via PCA size matching against a RealSense object cloud.

Compares covariance eigenvalues of Meshy surface samples to those of a camera-frame
point cloud (masked depth). Eigenvalues are pose-invariant variances; they scale as
s^2, so the least-squares scale is sqrt((ev_mesh · ev_real) / (ev_mesh · ev_mesh)).

The cameras only see the object's top and near sides, so the observed cloud is less spread out than the mesh's
surface samples and the scale comes out small (about 13% on 4 and 6 cm cubes). `floor_points_and_voxelize` corrects
most of this (to about 1.5%): it adds the object's footprint on the table it rests on (the unseen bottom face) and
evens out the point density with a voxel grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# Scene points within this distance of the object's centroid are used to fit the table plane
TABLE_FIT_RADIUS_M = 0.30
# RANSAC inlier distance for the table plane
TABLE_PLANE_INLIER_DIST_M = 0.004
TABLE_PLANE_RANSAC_ITERS = 1000
# RANSAC runs on at most this many of the scene points
TABLE_PLANE_MAX_POINTS = 20000
# The object rests on the table if its lowest points (TABLE_TOUCH_PERCENTILE of the heights above the plane) are within
# this distance of the plane. The masked depth rarely reaches the bottom edge: on the cubes the lowest points were 7-9
# mm up.
TABLE_TOUCH_DIST_M = 0.015
TABLE_TOUCH_PERCENTILE = 2.0
# 5 mm beat 2.5 mm on the cubes: smaller voxels keep more of the near faces' extra density
OBJECT_VOXEL_M = 0.005


def pca_eigenvalues(points: np.ndarray) -> np.ndarray:
    """Return the 3 covariance eigenvalues of `points`, largest first."""
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be Nx3, got {points.shape}"
    assert len(points) >= 3, f"Need at least 3 points for PCA, got {len(points)}"
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    assert covariance.shape == (3, 3), f"Expected 3x3 covariance, got {covariance.shape}"
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    assert eigenvalues.shape == (3,), f"Expected 3 eigenvalues, got {eigenvalues.shape}"
    assert float(eigenvalues[0]) > 0, f"Degenerate point cloud (largest eigenvalue {eigenvalues[0]})"
    return eigenvalues


def pca_uniform_scale(source_points: np.ndarray, target_points: np.ndarray) -> float:
    """Uniform scale that maps `source_points` size to `target_points` size."""
    ev_source = pca_eigenvalues(source_points)
    ev_target = pca_eigenvalues(target_points)
    denominator = float(np.dot(ev_source, ev_source))
    assert denominator > 0, f"PCA scale denominator must be > 0, got {denominator}"
    numerator = float(np.dot(ev_source, ev_target))
    assert numerator > 0, f"PCA scale numerator must be > 0, got {numerator}"
    scale = float(np.sqrt(numerator / denominator))
    assert np.isfinite(scale) and scale > 0, f"Invalid PCA scale {scale}"
    return scale


def scale_glb_to_pointcloud(
    glb_path: str | Path,
    real_points: np.ndarray,
    output_path: str | Path,
    n_sample: int = 5000,
) -> tuple[Path, float]:
    """Scale `glb_path` to match `real_points` and write a new GLB. Returns (path, scale)."""
    glb_path = Path(glb_path)
    output_path = Path(output_path)
    assert glb_path.is_file(), f"GLB not found: {glb_path}"
    assert output_path.suffix == ".glb", f"output_path must be a .glb, got {output_path}"
    assert n_sample >= 3, f"n_sample must be >= 3, got {n_sample}"
    assert real_points.ndim == 2 and real_points.shape[1] == 3, f"real_points must be Nx3, got {real_points.shape}"
    assert len(real_points) >= 3, f"Need at least 3 real points, got {len(real_points)}"

    scene = trimesh.load(str(glb_path), force="scene")
    assert isinstance(scene, trimesh.Scene), f"Expected trimesh.Scene, got {type(scene)}"
    assert len(scene.geometry) > 0, f"GLB has no geometry: {glb_path}"
    mesh = scene.to_geometry()
    assert isinstance(mesh, trimesh.Trimesh), f"Expected concatenated Trimesh, got {type(mesh)}"
    assert len(mesh.faces) > 0, f"Mesh has no faces: {glb_path}"
    assert len(mesh.vertices) > 0, f"Mesh has no vertices: {glb_path}"

    mesh_points, _ = trimesh.sample.sample_surface(mesh, count=n_sample)
    mesh_points = np.asarray(mesh_points, dtype=np.float64)
    assert mesh_points.shape == (n_sample, 3), f"Expected ({n_sample}, 3) samples, got {mesh_points.shape}"

    scale = pca_uniform_scale(mesh_points, np.asarray(real_points, dtype=np.float64))
    scene.apply_scale(scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    assert output_path.is_file() and output_path.stat().st_size > 0, f"Failed to write scaled GLB to {output_path}"
    return output_path, scale


def fit_plane_ransac(points: np.ndarray, inlier_dist_m: float, n_iters: int, seed: int) -> tuple[np.ndarray, float]:
    """Dominant plane `normal · p + offset = 0` of `(N, 3)` points: seeded RANSAC, then a least-squares refit on the
    inliers. Returns (unit normal, offset)."""
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be Nx3, got {points.shape}"
    assert len(points) >= 3, f"Need at least 3 points to fit a plane, got {len(points)}"
    assert inlier_dist_m > 0, f"inlier_dist_m must be > 0, got {inlier_dist_m}"
    assert n_iters >= 1, f"n_iters must be >= 1, got {n_iters}"
    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(n_iters):
        p0, p1, p2 = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:  # collinear sample
            continue
        normal /= norm
        inliers = np.abs((points - p0) @ normal) <= inlier_dist_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    assert best_inliers is not None and best_inliers.sum() >= 3, "RANSAC found no plane (all samples collinear?)"
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    normal = np.linalg.svd(inlier_pts - centroid, full_matrices=False)[2][-1]
    return normal, float(-normal @ centroid)


def fit_table_plane(object_points: np.ndarray, scene_points: np.ndarray) -> tuple[np.ndarray, float]:
    """Plane of the surface under the object: the dominant plane of the `scene_points` (not on the object) within
    `TABLE_FIT_RADIUS_M` of the object's centroid. The normal points toward the object. Returns (unit normal, offset).
    """
    assert (
        object_points.ndim == 2 and object_points.shape[1] == 3
    ), f"object_points must be Nx3, got {object_points.shape}"
    assert scene_points.ndim == 2 and scene_points.shape[1] == 3, f"scene_points must be Nx3, got {scene_points.shape}"
    centroid = object_points.mean(axis=0)
    nearby = scene_points[np.linalg.norm(scene_points - centroid, axis=1) <= TABLE_FIT_RADIUS_M]
    assert len(nearby) >= 3, f"Only {len(nearby)} scene points within {TABLE_FIT_RADIUS_M} m of the object"
    if len(nearby) > TABLE_PLANE_MAX_POINTS:
        nearby = nearby[np.random.default_rng(0).choice(len(nearby), size=TABLE_PLANE_MAX_POINTS, replace=False)]
    normal, offset = fit_plane_ransac(nearby, TABLE_PLANE_INLIER_DIST_M, TABLE_PLANE_RANSAC_ITERS, seed=0)
    if normal @ centroid + offset < 0:
        normal, offset = -normal, -offset
    return normal, offset


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """One point per occupied `voxel_m` voxel: the centroid of the points in it."""
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be Nx3, got {points.shape}"
    assert voxel_m > 0, f"voxel_m must be > 0, got {voxel_m}"
    _, inverse, counts = np.unique(
        np.floor(points / voxel_m).astype(np.int64), axis=0, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse.ravel(), points)
    return sums / counts[:, None]


def floor_points_and_voxelize(object_points: np.ndarray, scene_points: np.ndarray) -> tuple[np.ndarray, str]:
    """Object points for PCA scaling: if the object rests on the table (see `fit_table_plane`), add every object point
    projected onto the table plane; then `voxel_downsample` everything at `OBJECT_VOXEL_M`. Both clouds must be in the
    same frame. Returns the points and a one-line summary for logging."""
    normal, offset = fit_table_plane(object_points, scene_points)
    heights = object_points @ normal + offset
    lowest_m = float(np.percentile(heights, TABLE_TOUCH_PERCENTILE))
    touches = lowest_m <= TABLE_TOUCH_DIST_M
    summary = (
        f"table plane normal {np.round(normal, 4)} offset {offset:.4f} m; object {lowest_m * 100:.1f}-"
        f"{float(heights.max()) * 100:.1f} cm above it"
    )
    if touches:
        points = np.concatenate([object_points, object_points - np.outer(heights, normal)], axis=0)
        summary += f", touching: added {len(object_points)} floor points"
    else:
        points = object_points
        summary += f", not touching (lowest > {TABLE_TOUCH_DIST_M * 100:.1f} cm): no floor points"
    voxelized = voxel_downsample(points, OBJECT_VOXEL_M)
    summary += f"; {len(points)} -> {len(voxelized)} points in {OBJECT_VOXEL_M * 1000:g} mm voxels"
    return voxelized, summary
