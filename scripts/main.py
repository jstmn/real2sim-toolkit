import dataclasses
import pathlib

import cv2
import numpy as np
import tyro

from r2st.core import GroundedSAMPredictor
from r2st.openai import list_objects_in_image

"""
# Example usage:
uv run python scripts/main.py --image data/red_T_block_1.png
"""


@dataclasses.dataclass
class Args:
    image: pathlib.Path
    """Path to input image (JPG/PNG) of the T block."""

    output_dir: pathlib.Path = pathlib.Path("data/meshyai")
    """Directory to save generated mesh assets."""

    visualize: bool = False
    """If set, start a viser server and load the generated GLB."""

    gif: bool = True
    """If set, render a 360-degree orbit GIF of the generated GLB."""


def _pick_target_object(objects: list[str]) -> str:
    assert len(objects) >= 1, f"Expected at least one object from VLM, got {objects}"
    if len(objects) > 1:
        print(f"[warning] Expected 1 object, got {len(objects)}: {objects}")
        print(f"[warning] Picking first object: {objects[0]}")
    return objects[0]


def _get_sam_mask(image_bgr: np.ndarray, object_name: str) -> np.ndarray:
    predictor = GroundedSAMPredictor()
    assert predictor._sam_predictor is not None, "GroundedSAM predictor not loaded"
    assert predictor._bert_model is not None, "GroundedSAM bert model not loaded"
    masks = predictor.get_sam_mask(image_bgr, object_name)
    import torch

    assert isinstance(masks, torch.Tensor), f"Expected torch.Tensor masks, got {type(masks)}"
    mask = masks[0, 0].cpu().numpy()
    mask = mask.astype(bool)
    assert mask.shape[:2] == image_bgr.shape[:2], f"Mask shape {mask.shape[:2]} != image {image_bgr.shape[:2]}"
    return mask


def _generate_mesh_with_meshy(image_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    from r2st.meshy import MESHY_API_KEY, MeshyAPI

    assert MESHY_API_KEY is not None and len(MESHY_API_KEY) > 0, "MESHY_API_KEY is not set"
    api = MeshyAPI(MESHY_API_KEY)
    result_path = pathlib.Path(api.image_to_3d(image_path=image_path, output_dir=output_dir, enable_pbr=True))
    assert result_path.exists(), f"MeshyAPI did not create result at {result_path}"
    for cand in output_dir.rglob("*.glb"):
        return cand
    assert result_path.exists(), f"GLB not found in {output_dir}"
    return result_path


def _crop_to_mask(image_bgr: np.ndarray, mask: np.ndarray, pad: int = 8) -> np.ndarray:
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    assert mask.shape == image_bgr.shape[:2], f"Mask shape {mask.shape} != image {image_bgr.shape[:2]}"
    assert mask.any(), "Cannot crop empty mask"
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    assert y1 > y0 and x1 > x0, f"Invalid crop box: ({y0}:{y1}, {x0}:{x1})"
    return image_bgr[y0:y1, x0:x1]


def _visualize_glb(glb_path: pathlib.Path) -> None:
    import viser

    assert glb_path.exists(), f"GLB not found at {glb_path}"
    assert glb_path.is_file(), f"GLB path is not a file: {glb_path}"
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


def main(args: Args) -> None:
    assert args.image.exists(), f"Image file '{args.image}' not found"
    assert args.image.is_file(), f"Image path '{args.image}' is not a file"
    print(f"[info] Querying VLM for objects in {args.image} ...")
    objects = list_objects_in_image(str(args.image))
    print(f"[info] VLM objects: {' . '.join(objects)}")
    assert len(objects) >= 1, f"Expected at least one object, got {objects}"
    target = _pick_target_object(objects)
    print(f"[info] Target object for mesh: '{target}'")
    image_bgr = cv2.imread(str(args.image))
    assert image_bgr is not None, f"Failed to load image '{args.image}' with cv2"
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
    print(f"[info] Image shape: {image_bgr.shape}")
    mask = _get_sam_mask(image_bgr, target)
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    assert mask.shape[:2] == image_bgr.shape[:2], f"Mask shape {mask.shape[:2]} != image {image_bgr.shape[:2]}"
    assert mask.sum() > 0, "Segmentation mask is empty"
    print(f"[info] Mask pixels: {int(mask.sum())} / {mask.size}")
    target_slug = target.replace(" ", "_")
    asset_dir = args.output_dir / target_slug
    asset_dir.mkdir(parents=True, exist_ok=True)
    masked = image_bgr.copy()
    masked[np.logical_not(mask)] = 0
    masked_cropped = _crop_to_mask(masked, mask)
    demo = image_bgr.copy().astype(np.float32)
    demo[np.logical_not(mask)] *= 0.25
    demo = demo.astype(np.uint8)
    masked_path = asset_dir / f"{target_slug}__masked.png"
    masked_cropped_path = asset_dir / f"{target_slug}__masked_cropped.png"
    demo_path = asset_dir / f"{target_slug}__demo.png"
    cv2.imwrite(str(masked_path), masked)
    cv2.imwrite(str(masked_cropped_path), masked_cropped)
    cv2.imwrite(str(demo_path), demo)
    print(f"[info] Saved masked image to {masked_path}")
    print(f"[info] Saved masked cropped image to {masked_cropped_path} ({masked_cropped.shape[1]}x{masked_cropped.shape[0]})")
    print(f"[info] Saved demo overlay to {demo_path}")
    glb_path = asset_dir / f"{target_slug}_glb.glb"
    meshy_result = _generate_mesh_with_meshy(masked_cropped_path, asset_dir)
    assert meshy_result.exists(), f"Mesh file not created at {meshy_result}"
    if meshy_result != glb_path:
        import shutil

        shutil.copy2(str(meshy_result), str(glb_path))
    assert glb_path.exists(), f"Mesh file not created at {glb_path}"
    print("✅ Mesh generated successfully!")
    print(f"Objects: {' . '.join(objects)}")
    print(f"Target: {target}")
    print(f"Mask: {mask.shape}, {int(mask.sum())} foreground pixels")
    print(f"Mesh: {glb_path} ({glb_path.stat().st_size} bytes)")
    if args.gif:
        from r2st.utils import MeshUtils

        gif_path = asset_dir / f"{target_slug}__orbit.gif"
        gif_path = MeshUtils.save_orbit_gif(glb_path, gif_path)
        print(f"[info] Saved orbit GIF to {gif_path}")
    if args.visualize:
        _visualize_glb(glb_path)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
