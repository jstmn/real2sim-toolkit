"""2D bounding-box tracking via Cutie (video object segmentation).

FoundationPose's frame-to-frame ``track_one`` only refines a pose from the
previous frame's estimate -- it has no way to recover the object's image-plane
location if that prior drifts (e.g. under fast motion). Cutie is a cheap,
independent 2D tracker: re-projecting its bbox center each frame re-anchors
FoundationPose's (x, y) translation before refinement, which is the same
strategy used by FoundationPose++
(https://github.com/lidingsheng/FoundationPose-plus-plus, ``src/VOT.py``).

Expects a Cutie checkout at ``src/r2st/Cutie`` (same layout as the
FoundationPose vendor tree). Lazy-imports Cutie/torch so the rest of the
package can load without those heavy deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

CUTIE_DIR = Path(__file__).resolve().parent / "Cutie"


def _ensure_cutie_on_path() -> None:
    assert CUTIE_DIR.is_dir(), f"Cutie not found at {CUTIE_DIR}. Vendor or symlink the Cutie repo there."
    path_str = str(CUTIE_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_cutie_deps():
    _ensure_cutie_on_path()
    from cutie.inference.inference_core import InferenceCore
    from cutie.utils.get_default_model import get_default_model

    return InferenceCore, get_default_model


def _mask_to_bbox_xywh(mask: np.ndarray, erosion_size: int) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) around mask's nonzero pixels, or None if the mask is empty."""
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    rows = np.any(eroded, axis=1)
    cols = np.any(eroded, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


class CutieTracker:
    """Wraps Cutie's InferenceCore to track a single object's 2D bbox across frames."""

    def __init__(self, segment_threshold: float = 0.1, erosion_size: int = 5):
        assert 0 < segment_threshold < 1, f"segment_threshold must be in (0, 1), got {segment_threshold}"
        assert erosion_size >= 1, f"erosion_size must be >= 1, got {erosion_size}"
        InferenceCore, get_default_model = _load_cutie_deps()
        self.segment_threshold = segment_threshold
        self.erosion_size = erosion_size
        self._model = get_default_model()
        self._processor = InferenceCore(self._model, cfg=self._model.cfg)
        self._processor.max_internal_size = -1

    def initialize(self, color_rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """Seed the tracker with the first frame's object mask; return its bbox as (x, y, w, h)."""
        import torch
        from torchvision.transforms.functional import to_tensor

        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"
        assert mask.shape == color_rgb.shape[:2], f"mask shape {mask.shape} != color {color_rgb.shape[:2]}"

        mask_u8 = mask.astype(np.uint8)
        with torch.no_grad():
            frame_tensor = to_tensor(color_rgb).cuda().float()
            mask_tensor = torch.from_numpy(mask_u8).cuda()
            objects = [1] if mask_u8.any() else []
            output_prob = self._processor.step(frame_tensor, mask_tensor, objects=objects)
            mask_np = self._processor.output_prob_to_mask(output_prob, segment_threshold=self.segment_threshold)
            mask_np = mask_np.cpu().numpy()
        torch.cuda.empty_cache()
        return _mask_to_bbox_xywh(mask_np, self.erosion_size)

    def track(self, color_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Track into a new frame; return the object's bbox as (x, y, w, h), or None if lost."""
        import torch
        from torchvision.transforms.functional import to_tensor

        assert color_rgb.ndim == 3 and color_rgb.shape[2] == 3, f"color_rgb must be HxWx3, got {color_rgb.shape}"

        with torch.no_grad():
            frame_tensor = to_tensor(color_rgb).cuda().float()
            output_prob = self._processor.step(frame_tensor)
            mask_np = self._processor.output_prob_to_mask(output_prob, segment_threshold=self.segment_threshold)
            mask_np = mask_np.cpu().numpy()
        torch.cuda.empty_cache()
        return _mask_to_bbox_xywh(mask_np, self.erosion_size)
