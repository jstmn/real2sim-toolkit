"""Uniform mesh scale via PCA size matching against a RealSense object cloud.

Compares covariance eigenvalues of Meshy surface samples to those of a camera-frame
point cloud (masked depth). Eigenvalues are pose-invariant variances; they scale as
s^2, so the least-squares scale is sqrt((ev_mesh · ev_real) / (ev_mesh · ev_mesh)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


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
