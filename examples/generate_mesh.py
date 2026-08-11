import dataclasses
import pathlib
import shutil

import cv2
import numpy as np
import tyro

from r2st.core import GroundedSAMPredictor
from r2st.openai import list_objects_in_image
from r2st.types import CameraImage, ObjectAssets
from r2st.utils import ImageUtils, MeshUtils

"""
# Example usage (single image):
uv run python examples/generate_mesh.py --images data/red_T_block_1.png

# Example usage (multiple camera views of the same object, 1-4 images):
uv run python examples/generate_mesh.py --images data/red_T_block_1.png data/red_T_block_2.png
"""


@dataclasses.dataclass
class Args:
    images: list[pathlib.Path]
    """Paths to input images (JPG/PNG) of the object, e.g. from different camera views. Meshy's
    multi-image-to-3d endpoint accepts 1-4 images."""

    output_dir: pathlib.Path = pathlib.Path("data/meshyai")
    """Directory to save generated mesh assets."""

    visualize: bool = False
    """If set, start a viser server and load the generated GLB."""

    gif: bool = True
    """If set, render a 360-degree orbit GIF of the generated GLB."""


def _pick_target_object(objects: list[str]) -> str:
    assert (
        len(objects) == 1
    ), f"Expected exactly 1 object per camera (cross-view merging not implemented yet), got {len(objects)}: {objects}"
    return objects[0]


def _load_camera_image(image_path: pathlib.Path, predictor: GroundedSAMPredictor) -> tuple[CameraImage, str]:
    """Detect the single target object in `image_path` and segment it.

    Asserts exactly one object per camera. Cross-view object matching/merging is not
    implemented yet — a separate VLM call will handle that later.
    """
    print(f"[info] Querying VLM for objects in {image_path} ...")
    objects = list_objects_in_image(str(image_path))
    print(f"[info] VLM objects: {' . '.join(objects)}")
    target = _pick_target_object(objects)
    print(f"[info] Target object for '{image_path.name}': '{target}'")
    image_bgr = cv2.imread(str(image_path))
    assert image_bgr is not None, f"Failed to load image '{image_path}' with cv2"
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
    print(f"[info] Image shape: {image_bgr.shape}")
    mask = ImageUtils.get_sam_mask(predictor, image_bgr, target)
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    assert mask.sum() > 0, f"Segmentation mask is empty for '{image_path}'"
    print(f"[info] Mask pixels: {int(mask.sum())} / {mask.size}")
    return CameraImage(camera_name=image_path.stem, image=image_bgr, mask=mask), target


def _save_masked_crop(camera_image: CameraImage, target_slug: str, asset_dir: pathlib.Path) -> pathlib.Path:
    prefix = f"{target_slug}__{camera_image.camera_name}"
    masked = camera_image.image.copy()
    masked[np.logical_not(camera_image.mask)] = 0
    masked_cropped = ImageUtils.crop_to_mask(masked, camera_image.mask)
    demo = camera_image.image.copy().astype(np.float32)
    demo[np.logical_not(camera_image.mask)] *= 0.25
    demo = demo.astype(np.uint8)
    masked_path = asset_dir / f"{prefix}__masked.png"
    masked_cropped_path = asset_dir / f"{prefix}__masked_cropped.png"
    demo_path = asset_dir / f"{prefix}__demo.png"
    cv2.imwrite(str(masked_path), masked)
    cv2.imwrite(str(masked_cropped_path), masked_cropped)
    cv2.imwrite(str(demo_path), demo)
    print(f"[info] Saved masked image to {masked_path}")
    print(
        f"[info] Saved masked cropped image to {masked_cropped_path} "
        f"({masked_cropped.shape[1]}x{masked_cropped.shape[0]})"
    )
    print(f"[info] Saved demo overlay to {demo_path}")
    return masked_cropped_path


def main(args: Args) -> None:
    assert len(args.images) >= 1, "At least one --images path is required"
    assert len(args.images) <= 4, f"Meshy multi-image-to-3d accepts at most 4 images, got {len(args.images)}"
    for image_path in args.images:
        assert image_path.exists(), f"Image file '{image_path}' not found"
        assert image_path.is_file(), f"Image path '{image_path}' is not a file"

    predictor = GroundedSAMPredictor()
    loaded = [_load_camera_image(image_path, predictor) for image_path in args.images]
    camera_images = [camera_image for camera_image, _ in loaded]
    target = loaded[0][1]
    object_assets = ObjectAssets(object_name=target, camera_images=camera_images)
    target_slug = target.replace(" ", "_")
    asset_dir = args.output_dir / target_slug
    asset_dir.mkdir(parents=True, exist_ok=True)

    masked_cropped_paths = [_save_masked_crop(ci, target_slug, asset_dir) for ci in camera_images]

    glb_path = asset_dir / f"{target_slug}_glb.glb"
    meshy_result = MeshUtils.generate_with_meshy(masked_cropped_paths, asset_dir)
    assert meshy_result.exists(), f"Mesh file not created at {meshy_result}"
    if meshy_result != glb_path:
        shutil.copy2(str(meshy_result), str(glb_path))
    assert glb_path.exists(), f"Mesh file not created at {glb_path}"
    object_assets.glb_filepath = glb_path
    print("✅ Mesh generated successfully!")
    print(f"Target: {target}")
    print(f"Camera views used: {[ci.camera_name for ci in camera_images]}")
    print(f"Mesh: {glb_path} ({glb_path.stat().st_size} bytes)")
    if args.gif:
        gif_path = asset_dir / f"{target_slug}__orbit.gif"
        gif_path = MeshUtils.save_orbit_gif(glb_path, gif_path)
        print(f"[info] Saved orbit GIF to {gif_path}")
    if args.visualize:
        MeshUtils.visualize_glb(glb_path)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
