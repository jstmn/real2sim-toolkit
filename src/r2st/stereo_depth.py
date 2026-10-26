"""Learned stereo depth via S2M2 (https://github.com/junhong-3dv/s2m2, ICCV 2025).

Expects the S2M2 package vendored at ``src/r2st/S2M2/s2m2`` (upstream ``src/s2m2`` at commit
cbb5d169981e3086be1f959db5a8a0f139adcb17) and checkpoints in ``src/r2st/S2M2/weights/pretrain_weights``:

    uv run hf download minimok/s2m2 CH256NTR3.pth --local-dir src/r2st/S2M2/weights/pretrain_weights

Inputs must be a rectified pinhole stereo pair (vertical disparity < 2 px), e.g. the ZED SDK's rectified left/right
images. S2M2 is licensed CC BY-NC 4.0 (non-commercial research and education only).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

S2M2_DIR = Path(__file__).resolve().parent / "S2M2"
S2M2_WEIGHTS_DIR = S2M2_DIR / "weights" / "pretrain_weights"

# model_type -> (feature_channels, num_transformer). Checkpoint is CH{feature_channels}NTR{num_transformer}.pth
S2M2_MODEL_CONFIGS = {
    "S": (128, 1),
    "M": (192, 2),
    "L": (256, 3),
    "XL": (384, 3),
}

# S2M2 runs at 1/32 resolution internally; inputs are padded up to a multiple of this
_PAD_FACTOR = 32

# Parameters in the released checkpoints that the released S2M2 module no longer defines (the attention refiner and
# 4x upsampler that this release replaced with a UNet)
_UNUSED_CHECKPOINT_PREFIXES = ("feature_flow_attn.", "upsample_mask_4x.", "disp_init.layer_norm1.")


def _ensure_s2m2_on_path() -> None:
    assert (S2M2_DIR / "s2m2").is_dir(), f"S2M2 not found at {S2M2_DIR / 's2m2'}. Vendor the S2M2 package there."
    path_str = str(S2M2_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_s2m2_deps():
    _ensure_s2m2_on_path()
    from s2m2.core.model.s2m2 import S2M2
    from s2m2.core.utils.image_utils import image_crop, image_pad

    return S2M2, image_pad, image_crop


@dataclass
class StereoResult:
    """S2M2 outputs for one stereo pair, all ``(H, W)`` float32 on the left image's pixel grid."""

    disparity_px: np.ndarray
    """Left-to-right disparity in pixels."""

    occlusion: np.ndarray
    """Occlusion probability; 0 means occluded in the right image."""

    confidence: np.ndarray
    """Confidence probability (trained as 1 where the disparity error is < 4 px)."""


class S2M2StereoDepth:
    """Runs S2M2 on rectified stereo pairs and converts disparity to metric depth."""

    def __init__(self, model_type: str, refine_iter: int, allow_negative: bool, device: str):
        assert (
            model_type in S2M2_MODEL_CONFIGS
        ), f"Unknown model_type {model_type!r}. Available: {sorted(S2M2_MODEL_CONFIGS)}"
        assert isinstance(refine_iter, int) and refine_iter >= 1, f"refine_iter must be int >= 1, got {refine_iter}"
        assert isinstance(allow_negative, bool), f"allow_negative must be bool, got {type(allow_negative)}"
        S2M2, self._image_pad, self._image_crop = _load_s2m2_deps()

        feature_channels, num_transformer = S2M2_MODEL_CONFIGS[model_type]
        ckpt_path = S2M2_WEIGHTS_DIR / f"CH{feature_channels}NTR{num_transformer}.pth"
        assert ckpt_path.is_file(), (
            f"S2M2 checkpoint not found: {ckpt_path}. Download it with "
            f"`uv run hf download minimok/s2m2 {ckpt_path.name} --local-dir {S2M2_WEIGHTS_DIR}`"
        )
        self.device = torch.device(device)
        model = S2M2(
            feature_channels=feature_channels,
            dim_expansion=1,
            num_transformer=num_transformer,
            use_positivity=not allow_negative,
            refine_iter=refine_iter,
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Unlike upstream's my_load_state_dict (which silently skips mismatched/missing parameters), every model
        # parameter must load. Only the known leftovers from the pre-release architecture may be left over.
        result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        assert not result.missing_keys, f"{ckpt_path.name} is missing parameters: {result.missing_keys}"
        unknown = [k for k in result.unexpected_keys if not k.startswith(_UNUSED_CHECKPOINT_PREFIXES)]
        assert not unknown, f"{ckpt_path.name} has unexpected parameters: {unknown}"
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def predict(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> StereoResult:
        """Disparity, occlusion and confidence for one rectified ``(H, W, 3)`` uint8 RGB pair."""
        for name, img in (("left_rgb", left_rgb), ("right_rgb", right_rgb)):
            assert isinstance(img, np.ndarray), f"{name} must be ndarray, got {type(img)}"
            assert img.ndim == 3 and img.shape[2] == 3, f"{name} must be HxWx3, got {img.shape}"
            assert img.dtype == np.uint8, f"{name} must be uint8, got {img.dtype}"
        assert left_rgb.shape == right_rgb.shape, f"left_rgb {left_rgb.shape} != right_rgb {right_rgb.shape}"

        H, W = left_rgb.shape[:2]
        # S2M2 normalizes inputs from [0, 255] itself
        left = torch.from_numpy(left_rgb).permute(2, 0, 1)[None].float().to(self.device)
        right = torch.from_numpy(right_rgb).permute(2, 0, 1)[None].float().to(self.device)
        left = self._image_pad(left, _PAD_FACTOR)
        right = self._image_pad(right, _PAD_FACTOR)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            disp, occ, conf = self.model(left, right)
        return StereoResult(
            disparity_px=self._crop_to_numpy(disp, (H, W)),
            occlusion=self._crop_to_numpy(occ, (H, W)),
            confidence=self._crop_to_numpy(conf, (H, W)),
        )

    def _crop_to_numpy(self, t: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
        """Undo the ``_PAD_FACTOR`` padding on a ``(1, 1, H', W')`` model output."""
        out = self._image_crop(t, hw).squeeze().float().cpu().numpy()
        assert out.shape == hw, f"S2M2 output cropped to {out.shape}, expected {hw}"
        return out

    def depth_m(self, left_rgb: np.ndarray, right_rgb: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
        """``(H, W)`` float32 depth in meters on the left image's grid; 0 where disparity is not positive."""
        return disparity_to_depth_m(self.predict(left_rgb, right_rgb).disparity_px, fx, baseline_m)


def disparity_to_depth_m(disparity_px: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """``depth = fx * baseline / disparity``. Pixels with disparity <= 0 get depth 0 (no return), matching the h5
    depth convention."""
    assert disparity_px.ndim == 2, f"disparity_px must be HxW, got {disparity_px.shape}"
    assert fx > 0, f"fx must be > 0, got {fx}"
    assert baseline_m > 0, f"baseline_m must be > 0, got {baseline_m}"
    valid = disparity_px > 0
    depth = np.zeros(disparity_px.shape, dtype=np.float32)
    depth[valid] = (fx * baseline_m / disparity_px[valid]).astype(np.float32)
    return depth
