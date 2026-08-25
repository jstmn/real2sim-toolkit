"""Real2Sim core pipeline — toolkit port of real2sim_core.

Implements the same public helpers as the original (intrinsics, depth alignment,
pose transforms, rendering stubs) but without hard ROS/Sapien dependencies.
Heavy deps (torch, sapien, SAM 3) are imported lazily so pure geometry tests
run with only numpy.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage

from r2st.geometry import (
    camera_extrinsic_to_maniskill_pose,
    mat_to_sapien_pose_tuple,
    project_axes_to_image,
    realsense_to_maniskill_basis_matrix,
    reproject_depth_to_color_frame,
    scale_intrinsics,
    transform_pose_cam_to_world,
)

# Re-export geometry helpers so tests can import from core as in original
__all__ = [
    "camera_extrinsic_to_maniskill_pose",
    "mat_to_sapien_pose_tuple",
    "project_axes_to_image",
    "realsense_to_maniskill_basis_matrix",
    "reproject_depth_to_color_frame",
    "scale_intrinsics",
    "transform_pose_cam_to_world",
]

# SAM 3 interactive mask prompt is 4 * (image_size / backbone_stride) = 4 * (1008 / 14) = 288.
SAM_MASK_INPUT_HW = 288
MESHY_ASSET_DIR = Path("data/meshyai")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def save_mask_image(
    image: np.ndarray, output_dir: str, mask_list: torch.Tensor, box_list: torch.Tensor, label_list: list[str]
):
    def show_mask(mask_, ax_, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask_.shape[-2:]
        mask_image = mask_.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax_.imshow(mask_image)

    def show_box(box_, ax_, label_):
        x0, y0 = box_[0], box_[1]
        w, h = box_[2] - box_[0], box_[3] - box_[1]
        ax_.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
        ax_.text(x0, y0, label_)

    plt.figure(figsize=(10, 10))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    for mask in mask_list:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(box_list, label_list):
        show_box(box.numpy(), plt.gca(), label)
    now = datetime.now(UTC).strftime("%m:%d_%H:%M:%S")
    plt.axis("off")
    save_filepath = os.path.join(output_dir, f"sam3_output__{now}.jpg")
    plt.savefig(save_filepath, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved SAM 3 output to '{save_filepath}'")


def save_mask_image_2(img: np.ndarray, mask: np.ndarray | torch.Tensor, save_filepath: str, text: str):
    assert (
        len(mask.shape) == 2 and mask.shape[0:2] == img.shape[0:2]
    ), f"Mask shape {mask.shape} does not match image shape {img.shape}"
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    assert mask.dtype == bool, f"Mask dtype {mask.dtype} is not bool"
    img_out = img.copy().astype(np.float32)
    img_out[np.logical_not(mask)] *= 0.5
    img_out = img_out.astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    margin = 20
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = margin, margin + text_size[1]
    cv2.putText(img_out, text, (text_x, text_y), font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
    cv2.putText(img_out, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.imwrite(save_filepath, img_out)


def get_camera_extrinsic(camera_name: str, extrinsics: dict) -> np.ndarray:
    assert camera_name in extrinsics, f"Camera '{camera_name}' not found. Available: {list(extrinsics.keys())}"
    val = extrinsics[camera_name]
    if isinstance(val, dict) or hasattr(val, "matrix"):
        return np.asarray(val.matrix if hasattr(val, "matrix") else val["matrix"], dtype=np.float32)
    return np.asarray(val, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pointcloud helpers (thin wrappers that degrade gracefully without open3d)
# ---------------------------------------------------------------------------


def bbox_xyxy_from_mask(mask: np.ndarray) -> np.ndarray:
    """Axis-aligned XYXY box around nonzero pixels. ``x1``/``y1`` are exclusive."""
    assert mask.ndim == 2, f"mask must be 2D, got {mask.shape}"
    assert mask.dtype == bool, f"mask dtype must be bool, got {mask.dtype}"
    assert mask.any(), "Cannot compute a bbox from an empty mask"
    ys, xs = np.nonzero(mask)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    box = np.array([x0, y0, x1, y1], dtype=np.float64)
    assert box[2] > box[0] and box[3] > box[1], f"Degenerate bbox {box.tolist()}"
    return box


def binary_mask_to_sam_mask_input(mask: np.ndarray) -> np.ndarray:
    """Resize a full-res bool mask to SAM 3's dense prompt: ``(1, 288, 288)`` float32."""
    assert mask.ndim == 2, f"mask must be 2D, got {mask.shape}"
    assert mask.dtype == bool, f"mask dtype must be bool, got {mask.dtype}"
    assert mask.any(), "Cannot convert an empty mask to SAM mask_input"
    resized = cv2.resize(
        mask.astype(np.float32),
        (SAM_MASK_INPUT_HW, SAM_MASK_INPUT_HW),
        interpolation=cv2.INTER_LINEAR,
    )
    assert resized.shape == (
        SAM_MASK_INPUT_HW,
        SAM_MASK_INPUT_HW,
    ), f"resized mask_input {resized.shape} != ({SAM_MASK_INPUT_HW}, {SAM_MASK_INPUT_HW})"
    return resized[None, :, :].astype(np.float32)


class SAM3Predictor:
    """SAM 3 text-prompted segmentation plus interactive mask/box propagation."""

    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float = 0.3,
        debug_output_dir: str | None = None,
    ):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        from termcolor import colored

        self._debug_output_dir = debug_output_dir
        if self._debug_output_dir is not None:
            os.makedirs(self._debug_output_dir, exist_ok=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ("cpu", "cuda"), f"device must be 'cpu' or 'cuda', got {device!r}"
        assert 0.0 <= confidence_threshold <= 1.0, f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
        self._device = device
        self._confidence_threshold = confidence_threshold
        model = build_sam3_image_model(device=self._device, enable_inst_interactivity=True)
        self._processor = Sam3Processor(model, device=self._device, confidence_threshold=self._confidence_threshold)
        self._model = model
        assert self._processor is not None, "SAM 3 processor not loaded"
        assert self._model.inst_interactive_predictor is not None, "SAM 3 interactive predictor not loaded"
        if torch.device(self._device).type == "cuda":
            sam_where = f"GPU ({self._device})"
        else:
            sam_where = "CPU"
        print(colored(f"[info] SAM 3 is using {sam_where}", "yellow"))

    def get_ranked_sam_masks(self, image: np.ndarray, object_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return all SAM 3 masks for `object_name`, sorted by confidence descending.

        Returns `(N, H, W)` bool, `(N,)` scores, and a phrase list (the text prompt, once per mask).
        """
        assert self._processor is not None, "SAM 3 predictor not loaded"
        assert image.ndim == 3 and image.shape[2] == 3, f"Image must be HxWx3, got {image.shape}"
        assert len(object_name) > 0, "object_name must not be empty"
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        state = self._processor.set_image(image_pil)
        output = self._processor.set_text_prompt(state=state, prompt=object_name)
        masks_t = output["masks"]
        scores_t = output["scores"]
        boxes_t = output["boxes"]
        assert torch.is_tensor(masks_t), f"Expected tensor masks, got {type(masks_t)}"
        assert torch.is_tensor(scores_t), f"Expected tensor scores, got {type(scores_t)}"
        if masks_t.ndim == 4:
            assert masks_t.shape[1] == 1, f"Expected (N, 1, H, W) masks, got {tuple(masks_t.shape)}"
            masks_t = masks_t[:, 0]
        assert masks_t.ndim == 3, f"Expected (N, H, W) masks, got {tuple(masks_t.shape)}"
        assert masks_t.shape[0] >= 1, f"SAM 3 returned no masks for {object_name!r}"
        assert masks_t.shape[1:] == (height, width), f"Mask size {tuple(masks_t.shape[1:])} != image {(height, width)}"
        assert scores_t.shape == (masks_t.shape[0],), f"scores {tuple(scores_t.shape)} != n_masks {masks_t.shape[0]}"
        order = torch.argsort(scores_t, descending=True)
        masks_np = masks_t[order].detach().float().cpu().numpy().astype(bool)
        scores_np = scores_t[order].detach().float().cpu().numpy().astype(np.float64)
        phrases_sorted = [object_name] * int(masks_np.shape[0])
        if self._debug_output_dir is not None:
            print(f"Saving SAM 3 output to '{self._debug_output_dir}'")
            save_mask_image(
                image_rgb,
                self._debug_output_dir,
                masks_t[order].detach().float(),
                boxes_t[order].detach().float().cpu(),
                phrases_sorted,
            )
        return masks_np, scores_np, phrases_sorted

    def propagate_from_mask(
        self,
        image_bgr: np.ndarray,
        mask_input: np.ndarray,
        box_xyxy: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """SAM 3-decode one mask on ``image_bgr`` from a previous mask prompt.

        ``mask_input`` is SAM 3's dense prompt ``(1, 288, 288)``. ``box_xyxy`` is the
        previous binary mask's XYXY box in pixel coordinates.
        """
        assert self._model is not None, "SAM 3 model not loaded"
        assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, f"image_bgr must be HxWx3, got {image_bgr.shape}"
        assert mask_input.shape == (
            1,
            SAM_MASK_INPUT_HW,
            SAM_MASK_INPUT_HW,
        ), f"mask_input must be (1, {SAM_MASK_INPUT_HW}, {SAM_MASK_INPUT_HW}), got {mask_input.shape}"
        assert np.isfinite(mask_input).all(), "mask_input contains non-finite values"
        box_xyxy = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
        assert np.isfinite(box_xyxy).all(), f"box_xyxy contains non-finite values: {box_xyxy}"
        assert box_xyxy[2] > box_xyxy[0] and box_xyxy[3] > box_xyxy[1], f"Degenerate box {box_xyxy.tolist()}"
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        state = self._processor.set_image(image_pil)
        masks, ious, low_res = self._model.predict_inst(
            state,
            point_coords=None,
            point_labels=None,
            box=box_xyxy.astype(np.float32),
            mask_input=mask_input.astype(np.float32),
            multimask_output=False,
        )
        if masks.ndim == 2:
            masks = masks[None, ...]
            ious = np.atleast_1d(ious)
            low_res = low_res[None, ...] if low_res.ndim == 2 else low_res
        assert masks.ndim == 3 and masks.shape[0] == 1, f"Expected (1, H, W) masks, got {masks.shape}"
        assert masks.shape[1:] == image_bgr.shape[:2], f"Mask {masks.shape[1:]} != image {image_bgr.shape[:2]}"
        assert ious.shape == (1,), f"Expected one IoU, got {ious.shape}"
        assert low_res.shape == (
            1,
            SAM_MASK_INPUT_HW,
            SAM_MASK_INPUT_HW,
        ), f"low_res must be (1, {SAM_MASK_INPUT_HW}, {SAM_MASK_INPUT_HW}), got {low_res.shape}"
        mask = masks[0].astype(bool)
        assert mask.any(), "SAM 3 propagation produced an empty mask"
        return mask, float(ious[0]), low_res.astype(np.float32)

    def get_sam_mask(self, image: np.ndarray, object_name: str) -> torch.Tensor:
        masks_np, _, _ = self.get_ranked_sam_masks(image, object_name)
        return torch.from_numpy(masks_np[:, None, ...])
