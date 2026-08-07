"""Shared image and mesh visualization utilities."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


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


class MeshUtils:
    """Helpers for inspecting and exporting mesh assets."""

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
        server.scene.add_glb(name="/mesh", glb_data=glb_data)
        print(f"[info] Viser serving {glb_path} at http://{server.get_host()}:{server.get_port()}")
        server.sleep_forever()

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
        assert face_normals.shape == (len(base_colors), 3), (
            f"Normals shape {face_normals.shape} != colors {base_colors.shape}"
        )
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
