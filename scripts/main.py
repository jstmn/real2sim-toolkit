import dataclasses
import pathlib

import cv2
import numpy as np
import tyro

from r2st.core import GroundedSAMPredictor
from r2st.openai import list_objects_in_image


@dataclasses.dataclass
class Args:
    image: pathlib.Path
    """Path to input image (JPG/PNG) of the T block."""

    output_dir: pathlib.Path = pathlib.Path("data/meshyai")
    """Directory to save generated mesh assets."""


def _pick_target_object(objects: list[str]) -> str:
    assert len(objects) >= 1, f"Expected at least one object from VLM, got {objects}"
    for obj in objects:
        low = obj.lower()
        if "red" in low:
            return obj
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
    from mpcm.real2sim.meshy import MESHY_API_KEY, MeshyAPI

    assert MESHY_API_KEY is not None and len(MESHY_API_KEY) > 0, "MESHY_API_KEY is not set"
    api = MeshyAPI(MESHY_API_KEY)
    result_path = pathlib.Path(
        api.image_to_3d(image_path=image_path, output_dir=output_dir, enable_pbr=True, ai_model="meshy-4")
    )
    assert result_path.exists(), f"MeshyAPI did not create result at {result_path}"
    for cand in output_dir.rglob("*.glb"):
        return cand
    assert result_path.exists(), f"GLB not found in {output_dir}"
    return result_path


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
    demo = image_bgr.copy().astype(np.float32)
    demo[np.logical_not(mask)] *= 0.25
    demo = demo.astype(np.uint8)
    masked_path = asset_dir / f"{target_slug}__masked.png"
    demo_path = asset_dir / f"{target_slug}__demo.png"
    cv2.imwrite(str(masked_path), masked)
    cv2.imwrite(str(demo_path), demo)
    print(f"[info] Saved masked image to {masked_path}")
    print(f"[info] Saved demo overlay to {demo_path}")
    output_dir_for_mesh = args.output_dir / target
    output_dir_for_mesh.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir_for_mesh / f"{target}_glb.glb"
    meshy_result = _generate_mesh_with_meshy(masked_path, output_dir_for_mesh)
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
