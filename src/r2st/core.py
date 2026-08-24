"""Real2Sim core pipeline — toolkit port of real2sim_core.

Implements the same public helpers as the original (intrinsics, depth alignment,
pose transforms, rendering stubs) but without hard ROS/Sapien dependencies.
Heavy deps (torch, sapien, GroundingDINO, SAM 2) are imported lazily so pure
geometry tests run with only numpy.
"""

import os
import sys
import warnings
from datetime import UTC, datetime
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

_R2ST_DIR = Path(__file__).resolve().parent
GROUNDING_DINO_CONFIG = _R2ST_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = _R2ST_DIR / "models" / "groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = _R2ST_DIR / "models" / "sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_MASK_INPUT_HW = 256
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
    save_filepath = os.path.join(output_dir, f"sam2_output__{now}.jpg")
    plt.savefig(save_filepath, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved SAM 2 output to '{save_filepath}'")


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
    """Resize a full-res bool mask to SAM 2's dense prompt: ``(1, 256, 256)`` float32."""
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


class SAM2Predictor:
    """GroundingDINO text boxes + SAM 2 masks, with mask/box propagation."""

    def __init__(
        self,
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        debug_output_dir: str | None = None,
    ):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from termcolor import colored

        self._debug_output_dir = debug_output_dir
        if self._debug_output_dir is not None:
            os.makedirs(self._debug_output_dir, exist_ok=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ("cpu", "cuda"), f"device must be 'cpu' or 'cuda', got {device!r}"
        assert GROUNDING_DINO_CONFIG.is_file(), f"GroundingDINO config not found: {GROUNDING_DINO_CONFIG}"
        assert GROUNDING_DINO_CHECKPOINT.is_file(), f"GroundingDINO checkpoint not found: {GROUNDING_DINO_CHECKPOINT}"
        assert SAM2_CHECKPOINT.is_file(), f"SAM 2 checkpoint not found: {SAM2_CHECKPOINT}"
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._bert_model = self._load_bert_model(
            str(GROUNDING_DINO_CONFIG), str(GROUNDING_DINO_CHECKPOINT), None, self._device
        )
        sam2_model = build_sam2(SAM2_MODEL_CFG, str(SAM2_CHECKPOINT), device=self._device)
        self._sam_predictor = SAM2ImagePredictor(sam2_model)
        assert self._bert_model is not None, "GroundingDINO model not loaded"
        assert self._sam_predictor is not None, "SAM 2 predictor not loaded"
        if torch.device(self._device).type == "cuda":
            sam_where = f"GPU ({self._device})"
        else:
            sam_where = "CPU"
        print(colored(f"[info] SAM 2 is using {sam_where}", "yellow"))

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
        box_scores = logits_filt.max(dim=1)[0]
        return boxes_filt, pred_phrases, box_scores

    def get_ranked_sam_masks(self, image: np.ndarray, object_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return all SAM 2 masks for `object_name`, sorted by confidence descending.

        GroundingDINO boxes each produce three SAM 2 hypotheses (`multimask_output=True`).
        Score is `box_logit * sam_iou`. Returns `(N, H, W)` bool, `(N,)` scores, phrases.
        """
        assert self._sam_predictor is not None and self._bert_model is not None, "SAM 2 predictor not initialized"
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert isinstance(image_rgb, np.ndarray)
        self._sam_predictor.set_image(image_rgb)
        boxes_filt, pred_phrases, box_scores = self._get_grounding_output(
            self._bert_model, image_rgb, object_name, self._box_threshold, self._text_threshold, device=self._device
        )
        assert boxes_filt.size(0) > 0, f"GroundingDINO found no boxes for {object_name!r}"
        assert box_scores.shape == (
            boxes_filt.size(0),
        ), f"box_scores {box_scores.shape} != n_boxes {boxes_filt.size(0)}"
        img_size = image_rgb.shape[:2]
        W, H = img_size[1], img_size[0]
        assert H < W, f"Image height ({H}) should be less than width ({W})"
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        mask_rows = []
        score_rows = []
        phrase_rows = []
        for box_i in range(boxes_filt.size(0)):
            box_xyxy = boxes_filt[box_i].numpy().astype(np.float32)
            masks_i, iou_i, _ = self._sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy,
                multimask_output=True,
                normalize_coords=True,
            )
            if masks_i.ndim == 2:
                masks_i = masks_i[None, ...]
                iou_i = np.atleast_1d(iou_i)
            assert masks_i.ndim == 3, f"Expected (n_hyp, H, W) masks, got {masks_i.shape}"
            n_hyp, mh, mw = masks_i.shape
            assert (mh, mw) == (H, W), f"Mask size {(mh, mw)} != image {(H, W)}"
            assert iou_i.shape == (n_hyp,), f"iou_preds {iou_i.shape} != n_hyp {n_hyp}"
            combined_i = box_scores[box_i].item() * iou_i
            mask_rows.append(masks_i.astype(bool))
            score_rows.append(combined_i.astype(np.float64))
            phrase_rows.extend([pred_phrases[box_i]] * n_hyp)
        masks_np = np.concatenate(mask_rows, axis=0)
        scores_np = np.concatenate(score_rows, axis=0)
        assert masks_np.shape[0] == scores_np.shape[0] == len(phrase_rows)
        order = np.argsort(-scores_np)
        masks_np = masks_np[order]
        scores_np = scores_np[order]
        phrases_sorted = [phrase_rows[int(i)] for i in order]
        if self._debug_output_dir is not None:
            print(f"Saving SAM 2 output to '{self._debug_output_dir}'")
            save_mask_image(
                image_rgb,
                self._debug_output_dir,
                torch.from_numpy(masks_np.astype(np.uint8)),
                boxes_filt,
                pred_phrases,
            )
        return masks_np, scores_np, phrases_sorted

    def propagate_from_mask(
        self,
        image_bgr: np.ndarray,
        mask_input: np.ndarray,
        box_xyxy: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """SAM 2-decode one mask on ``image_bgr`` from a previous mask prompt.

        ``mask_input`` is SAM 2's dense prompt ``(1, 256, 256)``. ``box_xyxy`` is the
        previous binary mask's XYXY box in pixel coordinates. GroundingDINO is not used.
        """
        assert self._sam_predictor is not None, "SAM 2 predictor not loaded"
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
        self._sam_predictor.set_image(image_rgb)
        masks, ious, low_res = self._sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy.astype(np.float32),
            mask_input=mask_input.astype(np.float32),
            multimask_output=False,
            normalize_coords=True,
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
        assert mask.any(), "SAM 2 propagation produced an empty mask"
        return mask, float(ious[0]), low_res.astype(np.float32)

    def get_sam_mask(self, image: np.ndarray, object_name: str) -> torch.Tensor:
        masks_np, _, _ = self.get_ranked_sam_masks(image, object_name)
        return torch.from_numpy(masks_np[:, None, ...])
