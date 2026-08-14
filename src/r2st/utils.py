"""Shared image, mesh, sampling, and merged-h5 utilities."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

_DISTANCE_EUCLIDEAN = "euclidean"
_DISTANCE_CIRCULAR = "circular"
_DISTANCE_METRICS = (_DISTANCE_EUCLIDEAN, _DISTANCE_CIRCULAR)


def _feature_distance_sq(points: np.ndarray, centroid: np.ndarray, distance: str) -> np.ndarray:
    """Squared distances from each of `points` `(B, N, C)` to `centroid` `(B, C)` → `(B, N)`.

    `euclidean` is ordinary L2. `circular` is the geodesic on the circle S¹ per coordinate
    (shortest arc, identifying 0 with 2π), then L2 over coordinates — i.e. the geodesic on
    the flat torus T^C. That is the usual wrap-aware joint-space metric.
    """
    delta = points - centroid[:, None, :]
    if distance == _DISTANCE_EUCLIDEAN:
        return np.sum(delta * delta, axis=-1)
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return np.sum(wrapped * wrapped, axis=-1)


def farthest_point_sample_naive(
    points: np.ndarray,
    n_samples: int,
    *,
    start_index: int = 0,
    distance: str = _DISTANCE_EUCLIDEAN,
) -> np.ndarray:
    """Greedy farthest-point sampling. Returns indices into the N dimension.

    Naive O(`n_samples` * N) Python loop — not an optimized implementation.
    `from pyg_lib.ops import fps` is substantially faster for large Euclidean point clouds.

    `points` is `(N, C)` or batched `(B, N, C)` (any C, e.g. 3D points or 7-DoF joints).
    Unbatched input returns `(n_samples,)` int64; batched returns `(B, n_samples)`.
    Sampled values are `points[idx]` / `np.take_along_axis(points, idx[..., None], axis=1)`.

    `distance` is `"euclidean"` (L2) or `"circular"` (geodesic on S¹ per coordinate; 0 ≡ 2π).
    """
    assert isinstance(points, np.ndarray), f"points must be ndarray, got {type(points)}"
    assert points.ndim in (2, 3), f"points must be (N, C) or (B, N, C), got {points.shape}"
    unbatched = points.ndim == 2
    if unbatched:
        points = points[None, ...]
    B, N, C = points.shape
    assert C >= 1, f"points must have at least 1 feature dim, got {points.shape}"
    assert N >= 1, f"points must have at least 1 sample, got {points.shape}"
    assert isinstance(n_samples, int), f"n_samples must be int, got {type(n_samples)}"
    assert 1 <= n_samples <= N, f"n_samples must be in [1, {N}], got {n_samples}"
    assert isinstance(start_index, int), f"start_index must be int, got {type(start_index)}"
    assert 0 <= start_index < N, f"start_index must be in [0, {N}), got {start_index}"
    assert distance in _DISTANCE_METRICS, f"distance must be one of {_DISTANCE_METRICS}, got {distance!r}"

    pts = np.ascontiguousarray(points, dtype=np.float64)
    centroids = np.zeros((B, n_samples), dtype=np.int64)
    min_dist_sq = np.full((B, N), np.inf, dtype=np.float64)
    farthest = np.full((B,), start_index, dtype=np.int64)
    batch_idx = np.arange(B)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = pts[batch_idx, farthest]
        dist_sq = _feature_distance_sq(pts, centroid, distance)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
        farthest = np.argmax(min_dist_sq, axis=-1)
    if unbatched:
        return centroids[0]
    return centroids


MERGED_CAMERA_DATASETS = ("rgb", "depth", "timestamp_ms", "rgb_timestamp_ms")


def validate_merged_camera_group(f: h5py.File, h5_path: str | Path, camera: str) -> None:
    """Assert `f` matches the merged sensor-data layout from `examples/merge_camera_streams.py`:
    `obs/sensor_data/{camera}/[rgb, depth, timestamp_ms, rgb_timestamp_ms]`."""
    assert (
        "obs/sensor_data" in f
    ), f"{h5_path}: missing 'obs/sensor_data' group (not a merged sensor-data h5?). keys: {list(f.keys())}"
    sensor_data = f["obs/sensor_data"]
    group_path = f"obs/sensor_data/{camera}"
    assert group_path in f, f"Camera '{camera}' not found in {h5_path}. Available: {sorted(sensor_data.keys())}"
    group = f[group_path]
    for name in MERGED_CAMERA_DATASETS:
        assert name in group, f"{h5_path}:{group_path} missing dataset '{name}'"

    rgb, depth, timestamp_ms, rgb_timestamp_ms = (group[name] for name in MERGED_CAMERA_DATASETS)
    assert rgb.ndim == 4 and rgb.shape[3] == 3, f"{h5_path}:{group_path}/rgb must be NxHxWx3, got {rgb.shape}"
    assert rgb.dtype == np.uint8, f"{h5_path}:{group_path}/rgb must be uint8, got {rgb.dtype}"
    assert depth.ndim == 3, f"{h5_path}:{group_path}/depth must be NxHxW, got {depth.shape}"
    assert timestamp_ms.ndim == 1, f"{h5_path}:{group_path}/timestamp_ms must be 1D, got {timestamp_ms.shape}"
    assert (
        rgb_timestamp_ms.ndim == 1
    ), f"{h5_path}:{group_path}/rgb_timestamp_ms must be 1D, got {rgb_timestamp_ms.shape}"
    num_frames = timestamp_ms.shape[0]
    assert num_frames > 0, f"{h5_path}:{group_path} has no frames"
    assert (
        rgb.shape[0] == num_frames
    ), f"{h5_path}:{group_path}: rgb has {rgb.shape[0]} frames, timestamp_ms has {num_frames}"
    assert (
        depth.shape[0] == num_frames
    ), f"{h5_path}:{group_path}: depth has {depth.shape[0]} frames, timestamp_ms has {num_frames}"
    assert (
        rgb_timestamp_ms.shape[0] == num_frames
    ), f"{h5_path}:{group_path}: rgb_timestamp_ms has {rgb_timestamp_ms.shape[0]} frames, timestamp_ms has {num_frames}"
    assert (
        depth.shape[1:] == rgb.shape[1:3]
    ), f"{h5_path}:{group_path}: depth resolution {depth.shape[1:]} != rgb resolution {rgb.shape[1:3]}"


class ImageUtils:
    """Helpers for image masking and cropping."""

    @staticmethod
    def crop_to_mask(image_bgr: np.ndarray, mask: np.ndarray, pad: int = 8) -> np.ndarray:
        assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
        assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
        assert mask.shape == image_bgr.shape[:2], f"Mask shape {mask.shape} != image {image_bgr.shape[:2]}"
        assert mask.any(), "Cannot crop empty mask"
        assert pad >= 0, f"pad must be >= 0, got {pad}"
        ys, xs = np.where(mask)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
        assert y1 > y0 and x1 > x0, f"Invalid crop box: ({y0}:{y1}, {x0}:{x1})"
        return image_bgr[y0:y1, x0:x1]

    @staticmethod
    def save_masked_debug(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        asset_dir: str | Path,
        prefix: str,
    ) -> Path:
        """Write masked, masked-cropped, and demo overlay PNGs. Returns the cropped path."""
        import cv2

        assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
        assert image_bgr.dtype == np.uint8, f"Image dtype must be uint8, got {image_bgr.dtype}"
        assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
        assert mask.shape == image_bgr.shape[:2], f"Mask shape {mask.shape} != image {image_bgr.shape[:2]}"
        assert mask.any(), "Cannot save debug images for an empty mask"
        assert len(prefix) > 0, "prefix must not be empty"
        asset_dir = Path(asset_dir)
        assert asset_dir.is_dir(), f"asset_dir is not a directory: {asset_dir}"

        masked = image_bgr.copy()
        masked[np.logical_not(mask)] = 0
        masked_cropped = ImageUtils.crop_to_mask(masked, mask)
        demo = image_bgr.copy().astype(np.float32)
        demo[np.logical_not(mask)] *= 0.25
        demo = demo.astype(np.uint8)

        masked_path = asset_dir / f"{prefix}__masked.png"
        masked_cropped_path = asset_dir / f"{prefix}__masked_cropped.png"
        demo_path = asset_dir / f"{prefix}__demo.png"
        assert cv2.imwrite(str(masked_path), masked), f"Failed to write {masked_path}"
        assert cv2.imwrite(str(masked_cropped_path), masked_cropped), f"Failed to write {masked_cropped_path}"
        assert cv2.imwrite(str(demo_path), demo), f"Failed to write {demo_path}"
        print(f"[info] Saved masked image to {masked_path}")
        print(
            f"[info] Saved masked cropped image to {masked_cropped_path} "
            f"({masked_cropped.shape[1]}x{masked_cropped.shape[0]})"
        )
        print(f"[info] Saved demo overlay to {demo_path}")
        return masked_cropped_path

    @staticmethod
    def get_sam_masks_ranked(
        predictor, image_bgr: np.ndarray, object_name: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Run GroundedSAM and return all masks sorted by confidence descending.

        Returns `(N, H, W)` bool, `(N,)` scores, and GroundingDINO phrases.
        """
        assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
        assert len(object_name) > 0, "object_name must not be empty"
        assert predictor._sam_predictor is not None, "GroundedSAM predictor not loaded"
        assert predictor._bert_model is not None, "GroundedSAM bert model not loaded"
        masks, scores, phrases = predictor.get_ranked_sam_masks(image_bgr, object_name)
        assert isinstance(masks, np.ndarray) and masks.dtype == bool, f"Expected bool ndarray masks, got {type(masks)}"
        assert masks.ndim == 3, f"Expected (N, H, W) masks, got {masks.shape}"
        assert masks.shape[0] >= 1, "GroundedSAM returned no masks"
        assert masks.shape[1:] == image_bgr.shape[:2], f"Mask shape {masks.shape[1:]} != image {image_bgr.shape[:2]}"
        assert scores.shape == (masks.shape[0],), f"scores {scores.shape} != n_masks {masks.shape[0]}"
        assert len(phrases) == masks.shape[0], f"phrases {len(phrases)} != n_masks {masks.shape[0]}"
        return masks, scores, phrases

    @staticmethod
    def get_sam_mask(predictor, image_bgr: np.ndarray, object_name: str) -> np.ndarray:
        """Run GroundedSAM and return the highest-confidence bool HxW mask for `object_name`."""
        masks, _, _ = ImageUtils.get_sam_masks_ranked(predictor, image_bgr, object_name)
        return masks[0]


class MeshUtils:
    """Helpers for inspecting and exporting mesh assets."""

    @staticmethod
    def generate_with_meshy(
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        *,
        enable_pbr: bool = True,
    ) -> Path:
        """Generate a GLB via Meshy from one or more images. Reuses `output_dir/model_glb.glb` if present."""
        from r2st.meshy import MESHY_API_KEY, MeshyAPI

        image_paths = [Path(p) for p in image_paths]
        assert len(image_paths) > 0, "image_paths must not be empty"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = output_dir / "model_glb.glb"
        if existing.is_file():
            print(f"[info] Reusing existing Meshy GLB at {existing}")
            return existing
        assert MESHY_API_KEY is not None and len(MESHY_API_KEY) > 0, "MESHY_API_KEY is not set"
        api = MeshyAPI(MESHY_API_KEY)
        result_path = Path(api.image_to_3d(image_paths=image_paths, output_dir=output_dir, enable_pbr=enable_pbr))
        assert result_path.is_file(), f"MeshyAPI did not create result at {result_path}"
        return result_path

    @staticmethod
    def _pose_mat_to_wxyz_position(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from trimesh.transformations import quaternion_from_matrix

        assert pose.shape == (4, 4), f"pose must be 4x4, got {pose.shape}"
        wxyz = np.asarray(quaternion_from_matrix(pose), dtype=np.float64)
        assert wxyz.shape == (4,), f"Expected wxyz quaternion, got {wxyz.shape}"
        return wxyz, pose[:3, 3].astype(np.float64)

    @staticmethod
    def add_xy_grid(server) -> None:
        """Add the same XY ground grid used by `visualize_glb`."""
        server.scene.add_grid(
            "/xy_grid",
            width=2.0,
            height=2.0,
            plane="xy",
            cell_size=0.05,
            section_size=0.25,
            cell_color=(220, 220, 220),
            section_color=(200, 200, 200),
            plane_opacity=0.05,
        )

    @staticmethod
    def visualize_glb(glb_path: str | Path) -> None:
        import viser

        glb_path = Path(glb_path)
        assert glb_path.is_file(), f"GLB not found at {glb_path}"
        glb_data = glb_path.read_bytes()
        assert len(glb_data) > 0, f"GLB file is empty: {glb_path}"
        server = viser.ViserServer()
        server.scene.world_axes.visible = True
        server.scene.world_axes.scale = 0.25
        MeshUtils.add_xy_grid(server)
        server.scene.add_glb(name="/mesh", glb_data=glb_data)
        print(f"[info] Viser serving {glb_path} at http://{server.get_host()}:{server.get_port()}")
        server.sleep_forever()

    @staticmethod
    def visualize_tracking(
        glb_path: str | Path,
        poses_cam: np.ndarray,
        rgb_frames: np.ndarray,
        depth_m_frames: np.ndarray,
        K: np.ndarray,
    ) -> None:
        """Serve a viser scene of the tracked mesh in the camera frame (blocks)."""
        vis = TrackingVisualizer(glb_path, rgb_frames, depth_m_frames, K, mesh_position=np.asarray(poses_cam)[0, :3, 3])
        vis.set_poses(poses_cam)
        vis.wait()

    @staticmethod
    def _scene_to_triangle_mesh(glb_path: Path):
        import trimesh

        loaded = trimesh.load(str(glb_path), force="scene")
        assert isinstance(loaded, trimesh.Scene), f"Expected trimesh.Scene, got {type(loaded)}"
        assert len(loaded.geometry) > 0, f"GLB has no geometry: {glb_path}"
        mesh = loaded.dump(concatenate=True)
        assert isinstance(mesh, trimesh.Trimesh), f"Expected concatenated Trimesh, got {type(mesh)}"
        assert len(mesh.faces) > 0, f"Mesh has no faces: {glb_path}"
        assert len(mesh.vertices) > 0, f"Mesh has no vertices: {glb_path}"
        return mesh

    @staticmethod
    def _base_face_colors(mesh, num_faces: int) -> np.ndarray:
        if mesh.visual.kind == "face" and getattr(mesh.visual, "face_colors", None) is not None:
            face_colors = np.asarray(mesh.visual.face_colors)
            assert len(face_colors) == num_faces, "face_colors length mismatch"
            return face_colors[:, :3] / 255.0
        if mesh.visual.kind == "vertex" and getattr(mesh.visual, "vertex_colors", None) is not None:
            vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
            return vertex_colors[mesh.faces].mean(axis=1)
        return np.tile(np.array([[0.85, 0.2, 0.15]]), (num_faces, 1))

    @staticmethod
    def _shade_face_colors(
        base_colors: np.ndarray,
        face_normals: np.ndarray,
        *,
        ambient: float = 0.35,
        diffuse: float = 0.65,
    ) -> np.ndarray:
        assert base_colors.ndim == 2 and base_colors.shape[1] == 3, f"Bad base colors: {base_colors.shape}"
        assert face_normals.shape == (
            len(base_colors),
            3,
        ), f"Normals shape {face_normals.shape} != colors {base_colors.shape}"
        assert 0.0 <= ambient <= 1.0, f"ambient out of range: {ambient}"
        assert 0.0 <= diffuse <= 1.0, f"diffuse out of range: {diffuse}"
        assert ambient + diffuse > 0, "ambient + diffuse must be > 0"

        # Key light from above-front, soft fill from the opposite side.
        key = np.array([0.45, 0.25, 0.85], dtype=np.float64)
        key /= np.linalg.norm(key)
        fill = np.array([-0.35, -0.15, 0.4], dtype=np.float64)
        fill /= np.linalg.norm(fill)

        normals = face_normals.astype(np.float64)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        assert np.all(norms > 0), "Face normals contain zero-length vectors"
        normals = normals / norms

        key_term = np.clip(normals @ key, 0.0, 1.0)
        fill_term = np.clip(normals @ fill, 0.0, 1.0)
        intensity = ambient + diffuse * (0.75 * key_term + 0.25 * fill_term)
        shaded = np.clip(base_colors * intensity[:, None], 0.0, 1.0)
        return shaded

    @staticmethod
    def _draw_rgb_axes(ax, origin: np.ndarray, length: float) -> None:
        assert origin.shape == (3,), f"origin must be shape (3,), got {origin.shape}"
        assert length > 0, f"axis length must be > 0, got {length}"
        o = origin
        ax.plot([o[0], o[0] + length], [o[1], o[1]], [o[2], o[2]], color="#e74c3c", linewidth=2.0)  # X
        ax.plot([o[0], o[0]], [o[1], o[1] + length], [o[2], o[2]], color="#27ae60", linewidth=2.0)  # Y
        ax.plot([o[0], o[0]], [o[1], o[1]], [o[2], o[2] + length], color="#2980b9", linewidth=2.0)  # Z

    @staticmethod
    def _draw_xy_grid(ax, center: np.ndarray, half_extent: float, z: float, n_cells: int = 4) -> None:
        assert center.shape == (3,), f"center must be shape (3,), got {center.shape}"
        assert half_extent > 0, f"half_extent must be > 0, got {half_extent}"
        assert n_cells >= 1, f"n_cells must be >= 1, got {n_cells}"
        xs = np.linspace(center[0] - half_extent, center[0] + half_extent, n_cells + 1)
        ys = np.linspace(center[1] - half_extent, center[1] + half_extent, n_cells + 1)
        for x in xs:
            ax.plot([x, x], [ys[0], ys[-1]], [z, z], color="#cccccc", linewidth=0.8, alpha=0.9)
        for y in ys:
            ax.plot([xs[0], xs[-1]], [y, y], [z, z], color="#cccccc", linewidth=0.8, alpha=0.9)
        # RGB axis lines along the grid plane for X/Y, plus Z up from center.
        MeshUtils._draw_rgb_axes(ax, np.array([center[0], center[1], z], dtype=np.float64), half_extent)

    @staticmethod
    def save_orbit_gif(
        glb_path: str | Path,
        output_path: str | Path | None = None,
        *,
        num_frames: int = 90,
        resolution: tuple[int, int] = (512, 512),
        elevation_deg: float = 25.0,
        duration_ms: int = 220,
        zoom: float = 0.72,
    ) -> Path:
        """Render a 360-degree yaw orbit of a GLB and save it as a GIF.

        Uses trimesh for loading and matplotlib for headless rendering (no display
        server required). Textures are not shown; vertex/face colors are used when
        present, otherwise a flat accent color. Faces are Lambertian-shaded with a
        fixed key+fill light. A simple XY grid with RGB XYZ axes is drawn under the
        mesh.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from PIL import Image

        glb_path = Path(glb_path)
        assert glb_path.is_file(), f"GLB not found: {glb_path}"
        assert num_frames >= 2, f"num_frames must be >= 2, got {num_frames}"
        assert resolution[0] > 0 and resolution[1] > 0, f"Invalid resolution: {resolution}"
        assert duration_ms > 0, f"duration_ms must be > 0, got {duration_ms}"
        assert 0.1 <= zoom <= 2.0, f"zoom must be in [0.1, 2.0], got {zoom}"

        if output_path is None:
            output_path = glb_path.with_name(f"{glb_path.stem}__orbit.gif")
        else:
            output_path = Path(output_path)

        mesh = MeshUtils._scene_to_triangle_mesh(glb_path)
        triangles = mesh.vertices[mesh.faces]
        assert triangles.ndim == 3 and triangles.shape[1:] == (3, 3), f"Bad triangle array: {triangles.shape}"

        base_colors = MeshUtils._base_face_colors(mesh, len(triangles))
        facecolors = MeshUtils._shade_face_colors(base_colors, np.asarray(mesh.face_normals))

        bounds = mesh.bounds
        center = mesh.centroid
        radius = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.5)
        assert radius > 0, f"Degenerate mesh bounds: {bounds}"
        lim = radius * zoom
        grid_z = float(bounds[0, 2])
        grid_half = lim

        width_px, height_px = resolution
        dpi = 100
        fig_w = width_px / dpi
        fig_h = height_px / dpi

        frames: list[Image.Image] = []
        for i in range(num_frames):
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            MeshUtils._draw_xy_grid(ax, center, grid_half, grid_z, n_cells=4)
            collection = Poly3DCollection(
                triangles,
                facecolors=facecolors,
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
            )
            ax.add_collection3d(collection)
            ax.set_xlim(center[0] - lim, center[0] + lim)
            ax.set_ylim(center[1] - lim, center[1] + lim)
            ax.set_zlim(center[2] - lim, center[2] + lim)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elevation_deg, azim=360.0 * i / num_frames)
            ax.set_axis_off()
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            fig.tight_layout(pad=0)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB").resize(resolution, Image.Resampling.BILINEAR))

        assert len(frames) == num_frames, f"Expected {num_frames} frames, got {len(frames)}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        assert output_path.is_file() and output_path.stat().st_size > 0, f"Failed to write GIF: {output_path}"
        return output_path


class TrackingVisualizer:
    """Viser scene for pose tracking. Construct as soon as the scaled mesh exists; call `set_poses` later."""

    def __init__(
        self,
        glb_path: str | Path,
        rgb_frames: np.ndarray,
        depth_m_frames: np.ndarray,
        K: np.ndarray,
        mesh_position: np.ndarray,
    ) -> None:
        import viser

        from r2st.geometry import depth_rgb_to_pointcloud

        glb_path = Path(glb_path)
        assert glb_path.is_file(), f"GLB not found at {glb_path}"
        glb_data = glb_path.read_bytes()
        assert len(glb_data) > 0, f"GLB file is empty: {glb_path}"
        assert rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3, f"rgb_frames must be NxHxWx3, got {rgb_frames.shape}"
        assert rgb_frames.dtype == np.uint8, f"rgb_frames dtype must be uint8, got {rgb_frames.dtype}"
        self._num_frames = int(rgb_frames.shape[0])
        assert self._num_frames >= 1, "rgb_frames is empty"
        assert depth_m_frames.ndim == 3, f"depth_m_frames must be NxHxW, got {depth_m_frames.shape}"
        assert (
            depth_m_frames.shape[0] == self._num_frames
        ), f"depth_m_frames has {depth_m_frames.shape[0]} frames, rgb has {self._num_frames}"
        assert (
            depth_m_frames.shape[1:] == rgb_frames.shape[1:3]
        ), f"depth {depth_m_frames.shape[1:]} != rgb {rgb_frames.shape[1:3]}"
        assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
        mesh_position = np.asarray(mesh_position, dtype=np.float64).reshape(3)

        H, W = int(rgb_frames.shape[1]), int(rgb_frames.shape[2])
        fy = float(K[1, 1])
        assert fy > 0, f"K[1,1] (fy) must be > 0, got {fy}"
        fov = float(2.0 * np.arctan(H / (2.0 * fy)))
        aspect = W / H
        wxyz0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._rgb_frames = rgb_frames
        self._depth_m_frames = depth_m_frames
        self._K = K
        self._poses_cam: np.ndarray | None = None

        self._server = viser.ViserServer()
        self._server.scene.set_up_direction("-y")
        self._server.scene.world_axes.visible = True
        self._server.scene.world_axes.scale = 0.15
        MeshUtils.add_xy_grid(self._server)

        self._mesh_handle = self._server.scene.add_glb(
            name="/object", glb_data=glb_data, wxyz=wxyz0, position=mesh_position
        )
        self._axes_handle = self._server.scene.add_frame(
            "/object_axes",
            axes_length=0.06,
            axes_radius=0.004,
            origin_radius=0.008,
            wxyz=wxyz0,
            position=mesh_position,
        )
        self._frustum = self._server.scene.add_camera_frustum(
            "/camera",
            fov=fov,
            aspect=aspect,
            scale=0.12,
            line_width=1.5,
            image=rgb_frames[0],
            format="jpeg",
        )
        pts0, colors0 = depth_rgb_to_pointcloud(depth_m_frames[0], rgb_frames[0], K)
        self._pcd_handle = self._server.scene.add_point_cloud(
            "/scene_pcd",
            points=pts0,
            colors=colors0,
            point_size=0.002,
            point_shape="circle",
        )
        self._server.initial_camera.up = (0.0, -1.0, 0.0)
        self._server.initial_camera.look_at = mesh_position
        self._server.initial_camera.position = mesh_position + np.array([-0.25, -0.2, -0.45], dtype=np.float64)
        self._gui_image = self._server.gui.add_image(rgb_frames[0], label="RGB", format="jpeg")
        self._pose_md = self._server.gui.add_markdown(
            f"scaled mesh at `[{mesh_position[0]:.4f}, {mesh_position[1]:.4f}, {mesh_position[2]:.4f}]` (identity rot)"
        )
        print(f"[info] Viser tracking view at http://{self._server.get_host()}:{self._server.get_port()}")

    def set_poses(self, poses_cam: np.ndarray) -> None:
        assert self._poses_cam is None, "set_poses already called"
        poses_cam = np.asarray(poses_cam, dtype=np.float64)
        assert poses_cam.ndim == 3 and poses_cam.shape[1:] == (4, 4), f"poses_cam must be Nx4x4, got {poses_cam.shape}"
        assert (
            poses_cam.shape[0] == self._num_frames
        ), f"poses have {poses_cam.shape[0]} frames, rgb has {self._num_frames}"
        self._poses_cam = poses_cam
        if self._num_frames >= 2:
            traj = poses_cam[:, :3, 3]
            self._server.scene.add_line_segments(
                "/trajectory",
                points=np.stack([traj[:-1], traj[1:]], axis=1),
                colors=(70, 140, 255),
                line_width=2.0,
            )
        self._apply_timestep(0)
        if self._num_frames > 1:
            slider = self._server.gui.add_slider("timestep", min=0, max=self._num_frames - 1, step=1, initial_value=0)

            @slider.on_update
            def _on_timestep(_) -> None:
                self._apply_timestep(int(slider.value))

    def _apply_timestep(self, t: int) -> None:
        from r2st.geometry import depth_rgb_to_pointcloud

        assert self._poses_cam is not None, "set_poses must be called before applying timesteps"
        assert 0 <= t < self._num_frames, f"timestep {t} out of range [0, {self._num_frames})"
        wxyz, position = MeshUtils._pose_mat_to_wxyz_position(self._poses_cam[t])
        self._mesh_handle.wxyz = wxyz
        self._mesh_handle.position = position
        self._axes_handle.wxyz = wxyz
        self._axes_handle.position = position
        self._frustum.image = self._rgb_frames[t]
        self._gui_image.image = self._rgb_frames[t]
        pts, colors = depth_rgb_to_pointcloud(self._depth_m_frames[t], self._rgb_frames[t], self._K)
        self._pcd_handle.points = pts
        self._pcd_handle.colors = colors
        self._pose_md.content = (
            f"**t = {t} / {self._num_frames - 1}**\n\n"
            f"translation (cam): `[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]`"
        )

    def wait(self) -> None:
        self._server.sleep_forever()
