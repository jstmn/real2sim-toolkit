"""Real2Sim core pipeline — toolkit port of real2sim_core.

Implements the same public helpers as the original (intrinsics, depth alignment,
pose transforms, rendering stubs) but without hard ROS/Sapien dependencies.
Heavy deps (torch, sapien, grounded-sam) are imported lazily so pure geometry
tests run with only numpy.
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Make groundingdino importable as top-level (internal code does `from groundingdino.util ...`)
sys.path.insert(0, str(Path(__file__).parent / "GroundingDINO"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage

from r2st.geometry import (
    align_depth_to_color,
    align_ros_depth_to_color,
    camera_extrinsic_to_maniskill_pose,
    mat_to_sapien_pose_tuple,
    project_axes_to_image,
    realsense_to_maniskill_basis_matrix,
    scale_intrinsics,
    transform_pose_cam_to_world,
)

# Re-export geometry helpers so tests can import from core as in original
__all__ = [
    "align_depth_to_color",
    "align_ros_depth_to_color",
    "camera_extrinsic_to_maniskill_pose",
    "mat_to_sapien_pose_tuple",
    "project_axes_to_image",
    "realsense_to_maniskill_basis_matrix",
    "scale_intrinsics",
    "transform_pose_cam_to_world",
]

_R2ST_DIR = Path(__file__).resolve().parent
GROUNDING_DINO_CONFIG = _R2ST_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = _R2ST_DIR / "models" / "groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = _R2ST_DIR / "models" / "sam_vit_h_4b8939.pth"
SAM_VERSION = "vit_h"
MESHY_ASSET_DIR = Path("data/meshyai")

assert GROUNDING_DINO_CONFIG.is_file(), f"GroundingDINO config not found: {GROUNDING_DINO_CONFIG}"
assert GROUNDING_DINO_CHECKPOINT.is_file(), f"GroundingDINO checkpoint not found: {GROUNDING_DINO_CHECKPOINT}"
assert SAM_CHECKPOINT.is_file(), f"SAM checkpoint not found: {SAM_CHECKPOINT}"


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
    now = datetime.now().strftime("%m:%d_%H:%M:%S")
    plt.axis("off")
    save_filepath = os.path.join(output_dir, f"grounded_sam_output__{now}.jpg")
    plt.savefig(save_filepath, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved grounded SAM output to '{save_filepath}'")


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


class GroundedSAMPredictor:
    """GroundedSAM predictor with _sam_predictor for full pipeline."""

    def __init__(
        self,
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        debug_output_dir: str | None = None,
    ):
        self._debug_output_dir = debug_output_dir
        if self._debug_output_dir is not None:
            os.makedirs(self._debug_output_dir, exist_ok=True)
        from r2st.segment_anything.segment_anything import (
            SamPredictor,
            sam_model_registry,
        )

        self._device = device or "cpu"
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._bert_model = self._load_bert_model(
            str(GROUNDING_DINO_CONFIG), str(GROUNDING_DINO_CHECKPOINT), None, self._device
        )
        self._sam_predictor = SamPredictor(
            sam_model_registry[SAM_VERSION](checkpoint=str(SAM_CHECKPOINT)).to(self._device)
        )
        assert self._bert_model is not None, "GroundedSAM bert model not loaded"
        assert self._sam_predictor is not None, "GroundedSAM predictor not loaded"

    @staticmethod
    def _load_bert_model(model_config_path: str, model_checkpoint_path: str, bert_base_uncased_path, device: str):
        # Third-party FutureWarnings from transformers / huggingface_hub on current torch.
        warnings.filterwarnings(
            "ignore",
            message=r".*_register_pytree_node.*is deprecated.*",
            category=FutureWarning,
            module=r"transformers(\..*)?",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*resume_download.*is deprecated.*",
            category=FutureWarning,
            module=r"huggingface_hub(\..*)?",
        )

        from r2st.GroundingDINO.groundingdino.models import build_model
        from r2st.GroundingDINO.groundingdino.util.slconfig import SLConfig
        from r2st.GroundingDINO.groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(model_config_path)
        args.device = device
        args.bert_base_uncased_path = bert_base_uncased_path
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    @staticmethod
    def _get_grounding_output(
        model,
        image: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        with_logits: bool = True,
        device: str = "cpu",
    ):
        import torch

        assert getattr(model, "tokenizer", None) is not None, "GroundingDINO model has no tokenizer"
        import r2st.GroundingDINO.groundingdino.datasets.transforms as T
        from r2st.GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap

        def load_image(image_: np.ndarray):
            image_pil = PILImage.fromarray(image_).convert("RGB")
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            img, _ = transform(image_pil, None)
            return img

        image_t = load_image(image).to(device)
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image_t = image_t.to(device)
        with torch.no_grad():
            outputs = model(image_t[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        return boxes_filt, pred_phrases

    def get_sam_mask(self, image: np.ndarray, object_name: str) -> torch.Tensor:
        assert self._sam_predictor is not None and self._bert_model is not None, "GroundedSAM not initialized"
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert isinstance(image_rgb, np.ndarray)
        self._sam_predictor.set_image(image_rgb)
        boxes_filt, pred_phrases = self._get_grounding_output(
            self._bert_model, image_rgb, object_name, self._box_threshold, self._text_threshold, device=self._device
        )
        img_size = image_rgb.shape[:2]
        W, H = img_size[1], img_size[0]
        assert H < W, f"Image height ({H}) should be less than width ({W})"
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(boxes_filt, img_size).to(self._device)
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed_boxes.to(self._device), multimask_output=False
        )
        if self._debug_output_dir is not None:
            print(f"Saving grounded SAM output to '{self._debug_output_dir}'")
            save_mask_image(image_rgb, self._debug_output_dir, masks, boxes_filt, pred_phrases)
        return masks
