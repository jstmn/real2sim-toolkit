import dataclasses
import pathlib

import cv2
import numpy as np
import tyro

from r2st.core import SAM3Predictor
from r2st.utils import ImageUtils

"""
# Example usage:
uv run python examples/generate_mask.py --image data/red_T_block_1.png --object-description "red T block"
"""


@dataclasses.dataclass
class Args:
    image: pathlib.Path
    """Path to an input image (JPG/PNG)."""

    object_description: str
    """Language description of the object to segment, e.g. 'red T block'."""

    output_dir: pathlib.Path = pathlib.Path("data/masks")
    """Directory to save the mask overlay and bool npy."""


def main(args: Args) -> None:
    assert args.image.exists(), f"Image file '{args.image}' not found"
    assert args.image.is_file(), f"Image path '{args.image}' is not a file"
    assert len(args.object_description) > 0, "object_description must not be empty"

    image_bgr = cv2.imread(str(args.image))
    assert image_bgr is not None, f"Failed to load image '{args.image}' with cv2"
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"Image must be HxWx3, got {image_bgr.shape}"
    print(f"[info] Image shape: {image_bgr.shape}")

    predictor = SAM3Predictor()
    print(f"[info] Segmenting '{args.object_description}' ...")
    masks, scores, phrases = ImageUtils.get_sam_masks_ranked(predictor, image_bgr, args.object_description)
    for rank in range(masks.shape[0]):
        print(
            f"[info]   rank={rank} score={scores[rank]:.4f} phrase={phrases[rank]!r} "
            f"pixels={int(masks[rank].sum())}"
        )
    mask = masks[0]
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    assert mask.any(), f"Segmentation mask is empty for '{args.object_description}'"
    print(f"[info] Best mask pixels: {int(mask.sum())} / {mask.size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_slug = args.object_description.replace(" ", "_")
    prefix = f"{args.image.stem}__{object_slug}"
    ImageUtils.save_masked_debug(image_bgr, mask, args.output_dir, prefix)
    mask_path = args.output_dir / f"{prefix}__mask.npy"
    np.save(mask_path, mask)
    print(f"[info] Saved bool mask to {mask_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
