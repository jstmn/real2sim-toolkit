"""Learned stereo depth via S2M2 (https://github.com/junhong-3dv/s2m2, ICCV 2025).

Expects the S2M2 package vendored at ``src/r2st/S2M2/s2m2`` (upstream ``src/s2m2`` at commit
cbb5d169981e3086be1f959db5a8a0f139adcb17) and checkpoints in ``src/r2st/S2M2/weights/pretrain_weights``:

    uv run hf download minimok/s2m2 CH256NTR3.pth --local-dir src/r2st/S2M2/weights/pretrain_weights

Inputs must be a rectified pinhole stereo pair (vertical disparity < 2 px), e.g. the ZED SDK's rectified left/right
images. S2M2 is licensed CC BY-NC 4.0 (non-commercial research and education only).

Two backends with the same interface: ``S2M2StereoDepth`` runs the PyTorch model under fp16 autocast, and
``S2M2TensorRTStereoDepth`` runs a TensorRT engine of it built by ``build_s2m2_tensorrt_engine``
(``examples/build_s2m2_tensorrt_engine.py``). An engine has a fixed image size and runs only with the TensorRT version
and GPU it was built with; its file name records all of them (``s2m2_tensorrt_engine_path``).
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

S2M2_DIR = Path(__file__).resolve().parent / "S2M2"
S2M2_WEIGHTS_DIR = S2M2_DIR / "weights" / "pretrain_weights"
S2M2_TENSORRT_DIR = S2M2_DIR / "weights" / "tensorrt"

# model_type -> (feature_channels, num_transformer). Checkpoint is CH{feature_channels}NTR{num_transformer}.pth
S2M2_MODEL_CONFIGS = {
    "S": (128, 1),
    "M": (192, 2),
    "L": (256, 3),
    "XL": (384, 3),
}

# S2M2 runs at 1/32 resolution internally; inputs are padded up to a multiple of this
_PAD_FACTOR = 32

# Hex digits of a checkpoint's SHA-256 kept in its id (see s2m2_checkpoint_id)
_CHECKPOINT_SHA_PREFIX_LEN = 12

# Parameters in the released checkpoints that the released S2M2 module no longer defines (the attention refiner and
# 4x upsampler that this release replaced with a UNet)
_UNUSED_CHECKPOINT_PREFIXES = ("feature_flow_attn.", "upsample_mask_4x.", "disp_init.layer_norm1.")

# The engine's input and output tensors, in order
_TENSORRT_INPUTS = ("left", "right")
_TENSORRT_OUTPUTS = ("disparity", "occlusion", "confidence")

# ONNX opset of the engine's export. Opset 20 has a Gelu op; 18 spells GELU out with Erf.
_ONNX_OPSET = 20

# TensorRT builder optimization level (0-5). At the default 3, TensorRT 10.16 and 11.4 pick a ~15x slower tactic for
# S2M2's 1x1 convolutions on small feature maps (1.1 ms instead of 0.06 ms on an RTX 4090), which makes the engine
# slower than PyTorch.
_TENSORRT_OPTIMIZATION_LEVEL = 5


def s2m2_checkpoint_path(model_type: str) -> Path:
    """Path of the `model_type` checkpoint, `CH{feature_channels}NTR{num_transformer}.pth`, asserting it is downloaded."""
    assert (
        model_type in S2M2_MODEL_CONFIGS
    ), f"Unknown model_type {model_type!r}. Available: {sorted(S2M2_MODEL_CONFIGS)}"
    feature_channels, num_transformer = S2M2_MODEL_CONFIGS[model_type]
    ckpt_path = S2M2_WEIGHTS_DIR / f"CH{feature_channels}NTR{num_transformer}.pth"
    assert ckpt_path.is_file(), (
        f"S2M2 checkpoint not found: {ckpt_path}. Download it with "
        f"`uv run hf download minimok/s2m2 {ckpt_path.name} --local-dir {S2M2_WEIGHTS_DIR}`"
    )
    return ckpt_path


def s2m2_checkpoint_id(model_type: str) -> str:
    """`<checkpoint stem>-<sha256 prefix>`, e.g. `CH256NTR3-770e82227258`: identifies the exact weights, so outputs
    cached under it go stale if the checkpoint file changes."""
    import hashlib

    ckpt_path = s2m2_checkpoint_path(model_type)
    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            sha256.update(chunk)
    return f"{ckpt_path.stem}-{sha256.hexdigest()[:_CHECKPOINT_SHA_PREFIX_LEN]}"


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


def _check_s2m2_args(model_type: str, refine_iter: int, allow_negative: bool) -> None:
    assert (
        model_type in S2M2_MODEL_CONFIGS
    ), f"Unknown model_type {model_type!r}. Available: {sorted(S2M2_MODEL_CONFIGS)}"
    assert isinstance(refine_iter, int) and refine_iter >= 1, f"refine_iter must be int >= 1, got {refine_iter}"
    assert isinstance(allow_negative, bool), f"allow_negative must be bool, got {type(allow_negative)}"


class _S2M2StereoDepthBase:
    """Pads, runs and crops one stereo pair; subclasses run the padded pair (``_run``)."""

    device: torch.device

    def __init__(self):
        _, self._image_pad, self._image_crop = _load_s2m2_deps()

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Disparity, occlusion and confidence, each ``(1, 1, H', W')``, of a padded ``(1, 3, H', W')`` float pair in
        [0, 255] on ``self.device``."""
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> StereoResult:
        """Disparity, occlusion and confidence for one rectified ``(H, W, 3)`` uint8 RGB pair."""
        for name, img in (("left_rgb", left_rgb), ("right_rgb", right_rgb)):
            assert isinstance(img, np.ndarray), f"{name} must be ndarray, got {type(img)}"
            assert img.ndim == 3 and img.shape[2] == 3, f"{name} must be HxWx3, got {img.shape}"
            assert img.dtype == np.uint8, f"{name} must be uint8, got {img.dtype}"
        assert left_rgb.shape == right_rgb.shape, f"left_rgb {left_rgb.shape} != right_rgb {right_rgb.shape}"

        H, W = left_rgb.shape[:2]
        # S2M2 normalizes inputs from [0, 255] itself. Copied to the GPU as uint8, a quarter of the bytes of float32.
        left = torch.from_numpy(left_rgb).to(self.device).permute(2, 0, 1)[None].float()
        right = torch.from_numpy(right_rgb).to(self.device).permute(2, 0, 1)[None].float()
        disp, occ, conf = self._run(self._image_pad(left, _PAD_FACTOR), self._image_pad(right, _PAD_FACTOR))
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

    def disparity_sequence(self, left_rgb: np.ndarray, right_rgb: np.ndarray, desc: str) -> np.ndarray:
        """Disparity in pixels for every frame of ``(T, H, W, 3)`` uint8 left/right sequences → ``(T, H, W)``
        float32."""
        from tqdm import tqdm

        assert left_rgb.ndim == 4, f"left_rgb must be TxHxWx3, got {left_rgb.shape}"
        assert left_rgb.shape == right_rgb.shape, f"left_rgb {left_rgb.shape} != right_rgb {right_rgb.shape}"
        disparity = np.zeros(left_rgb.shape[:3], dtype=np.float32)
        for t in tqdm(range(len(left_rgb)), desc=desc):
            disparity[t] = self.predict(left_rgb[t], right_rgb[t]).disparity_px
        return disparity


class S2M2StereoDepth(_S2M2StereoDepthBase):
    """Runs S2M2 (PyTorch, fp16 autocast on CUDA) on rectified stereo pairs and converts disparity to metric depth."""

    def __init__(self, model_type: str, refine_iter: int, allow_negative: bool, device: str):
        _check_s2m2_args(model_type, refine_iter, allow_negative)
        super().__init__()
        S2M2, _, _ = _load_s2m2_deps()

        feature_channels, num_transformer = S2M2_MODEL_CONFIGS[model_type]
        ckpt_path = s2m2_checkpoint_path(model_type)
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

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            return self.model(left, right)


def _padded_hw(image_hw: tuple[int, int]) -> tuple[int, int]:
    """``image_hw`` rounded up to multiples of ``_PAD_FACTOR``, the size S2M2 runs at."""
    H, W = image_hw
    return -(-H // _PAD_FACTOR) * _PAD_FACTOR, -(-W // _PAD_FACTOR) * _PAD_FACTOR


def s2m2_tensorrt_engine_path(
    model_type: str, refine_iter: int, allow_negative: bool, image_hw: tuple[int, int], device: str
) -> Path:
    """``S2M2_TENSORRT_DIR/<checkpoint_id>__refine_iter=<n>__allow_negative=<b>__<W>x<H>__trt<version>__<gpu>.engine``.
    ``refine_iter`` and ``allow_negative`` are compiled into the engine, ``<W>x<H>`` is the (unpadded) image size it
    takes, and ``<version>`` and ``<gpu>`` are the TensorRT version and the GPU it was built with, the only ones it runs
    with. ``checkpoint_id`` is ``s2m2_checkpoint_id``, so swapping the weights never reuses a stale engine."""
    import tensorrt as trt

    _check_s2m2_args(model_type, refine_iter, allow_negative)
    H, W = image_hw
    assert isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0, f"image_hw must be 2 ints > 0: {image_hw}"
    gpu = torch.cuda.get_device_name(torch.device(device)).replace(" ", "_")
    name = (
        f"{s2m2_checkpoint_id(model_type)}__refine_iter={refine_iter}__allow_negative={allow_negative}__{W}x{H}"
        f"__trt{trt.__version__}__{gpu}.engine"
    )
    return S2M2_TENSORRT_DIR / name


def build_s2m2_tensorrt_engine(
    model_type: str, refine_iter: int, allow_negative: bool, image_hw: tuple[int, int], device: str
) -> Path:
    """Builds the engine at ``s2m2_tensorrt_engine_path`` and returns its path; it must not exist yet.

    The model is exported to ONNX under the same fp16 autocast that ``S2M2StereoDepth`` runs it with, so the graph
    carries autocast's fp16/fp32 casts, and built as a strongly typed network that keeps exactly those types (TensorRT
    11 has no weakly typed fp16 mode, upstream's ``--fp16`` build). Takes several minutes, and for XL ~2 GB of disk for
    the intermediate ONNX, in a temporary directory in ``S2M2_TENSORRT_DIR``."""
    import tensorrt as trt

    engine_path = s2m2_tensorrt_engine_path(model_type, refine_iter, allow_negative, image_hw, device)
    assert not engine_path.exists(), f"{engine_path} already exists. Delete it to rebuild."
    stereo = S2M2StereoDepth(model_type, refine_iter=refine_iter, allow_negative=allow_negative, device=device)
    assert stereo.device.type == "cuda", f"TensorRT needs a CUDA device, got {device!r}"
    # Two separate tensors: torch.export merges an input passed twice, and the engine would then read only `right`
    example_left, example_right = (
        torch.zeros((1, 3, *_padded_hw(image_hw)), dtype=torch.float32, device=stereo.device) for _ in range(2)
    )

    S2M2_TENSORRT_DIR.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    with tempfile.TemporaryDirectory(dir=S2M2_TENSORRT_DIR) as tmp_dir:
        onnx_path = Path(tmp_dir) / "s2m2.onnx"
        print(f"[info] Exporting S2M2 {model_type} at {example_left.shape[-1]}x{example_left.shape[-2]} to ONNX")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            program = torch.onnx.export(
                stereo.model,
                (example_left, example_right),
                dynamo=True,
                opset_version=_ONNX_OPSET,
                input_names=list(_TENSORRT_INPUTS),
                output_names=list(_TENSORRT_OUTPUTS),
            )
        program.save(str(onnx_path))
        del program, stereo
        torch.cuda.empty_cache()
        parsed = parser.parse_from_file(str(onnx_path))
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        assert parsed, "TensorRT could not parse the S2M2 ONNX export:\n" + "\n".join(errors)
        config = builder.create_builder_config()
        config.builder_optimization_level = _TENSORRT_OPTIMIZATION_LEVEL
        print(
            f"[info] Building the TensorRT {trt.__version__} engine (optimization level {_TENSORRT_OPTIMIZATION_LEVEL})"
        )
        with torch.cuda.device(example_left.device):
            plan = builder.build_serialized_network(network, config)
    assert plan is not None, "TensorRT failed to build the S2M2 engine; see its log above"
    # Write then rename, so an interrupted build never leaves a truncated engine behind
    tmp_path = engine_path.with_suffix(".engine.tmp")
    with open(tmp_path, "wb") as f:
        f.write(plan)
    tmp_path.replace(engine_path)
    print(f"[info] Saved S2M2 TensorRT engine {engine_path} ({plan.nbytes / 1e9:.2f} GB)")
    return engine_path


class S2M2TensorRTStereoDepth(_S2M2StereoDepthBase):
    """Runs a TensorRT engine of S2M2 (``build_s2m2_tensorrt_engine``) on rectified ``image_hw`` stereo pairs."""

    def __init__(self, model_type: str, refine_iter: int, allow_negative: bool, image_hw: tuple[int, int], device: str):
        import tensorrt as trt

        super().__init__()
        engine_path = s2m2_tensorrt_engine_path(model_type, refine_iter, allow_negative, image_hw, device)
        H, W = image_hw
        build_script = Path(__file__).resolve().parents[2] / "examples" / "build_s2m2_tensorrt_engine.py"
        assert engine_path.is_file(), (
            f"S2M2 TensorRT engine not found: {engine_path}. Build it (several minutes) from the env that runs it, "
            f"with `uv run python {build_script} --model-type {model_type} --refine-iter {refine_iter} "
            f"--allow-negative {allow_negative} --height {H} --width {W} --device {device}`."
        )
        self.device = torch.device(device)
        assert self.device.type == "cuda", f"TensorRT needs a CUDA device, got {device!r}"
        self.image_hw = (H, W)
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
            assert self._engine is not None, f"TensorRT could not load {engine_path}; see its log above"
            self._context = self._engine.create_execution_context()
        names = tuple(self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors))
        assert names == _TENSORRT_INPUTS + _TENSORRT_OUTPUTS, f"{engine_path.name} has tensors {names}"
        self._input_shape = (1, 3, *_padded_hw(image_hw))
        for name in names:
            shape = tuple(self._engine.get_tensor_shape(name))
            expected = self._input_shape if name in _TENSORRT_INPUTS else (1, 1, *self._input_shape[2:])
            assert shape == expected, f"{engine_path.name}: {name} is {shape}, expected {expected}"
            assert self._engine.get_tensor_dtype(name) == trt.float32, f"{engine_path.name}: {name} is not float32"
        self._stream = torch.cuda.Stream(self.device)
        # Reused for every pair: predict copies them to the host before the next pair runs
        self._outputs = tuple(
            torch.empty((1, 1, *self._input_shape[2:]), dtype=torch.float32, device=self.device)
            for _ in _TENSORRT_OUTPUTS
        )
        for name, t in zip(_TENSORRT_OUTPUTS, self._outputs):
            self._context.set_tensor_address(name, t.data_ptr())

    def predict(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> StereoResult:
        assert (
            left_rgb.shape[:2] == self.image_hw
        ), f"left_rgb is {left_rgb.shape[:2]}, but the engine takes {self.image_hw} pairs"
        return super().predict(left_rgb, right_rgb)

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for name, t in zip(_TENSORRT_INPUTS, (left, right)):
            assert t.shape == self._input_shape and t.dtype == torch.float32, f"{name}: {t.shape} {t.dtype}"
            assert t.is_contiguous(), f"{name} is not contiguous"
            self._context.set_tensor_address(name, t.data_ptr())
        # On its own stream (on the default stream, TensorRT adds syncs of its own), ordered after the padding and
        # before the crop on torch's current stream. The inputs are recorded on it, so their memory is not reused
        # while it reads them.
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        left.record_stream(self._stream)
        right.record_stream(self._stream)
        with torch.cuda.device(self.device):
            ran = self._context.execute_async_v3(self._stream.cuda_stream)
        assert ran, "TensorRT failed to run the S2M2 engine; see its log above"
        current.wait_stream(self._stream)
        return self._outputs


def s2m2_disparity_cache_path(
    h5_path: Path, camera: str, checkpoint_id: str, refine_iter: int, num_frames: int
) -> Path:
    """``<h5_dir>/<h5_stem>/s2m2_disparity/<camera>__model=<checkpoint_id>__refine_iter=<n>__frames=<T>.npy``: S2M2
    disparity of the first ``num_frames`` frames of ``camera``'s ``left/rgb`` + ``right/rgb``. ``checkpoint_id`` is
    ``s2m2_checkpoint_id``, so the cache is keyed by the exact weights. Disparity, not depth, so the cache does not
    depend on ``fx`` or the baseline."""
    name = f"{camera}__model={checkpoint_id}__refine_iter={refine_iter}__frames={num_frames}.npy"
    return h5_path.parent / h5_path.stem / "s2m2_disparity" / name


def cached_s2m2_depth_m(
    h5_path: Path,
    camera: str,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    fx: float,
    baseline_m: float,
    model_type: str,
    refine_iter: int,
) -> np.ndarray:
    """S2M2 depth ``(T, H, W)`` float32 for ``camera``'s left/right ``(T, H, W, 3)`` sequences read from ``h5_path``.
    The disparity is cached at ``s2m2_disparity_cache_path``; S2M2 is loaded and run only when it is missing."""
    checkpoint_id = s2m2_checkpoint_id(model_type)
    cache_path = s2m2_disparity_cache_path(Path(h5_path), camera, checkpoint_id, refine_iter, len(left_rgb))
    if cache_path.is_file():
        disparity = np.load(cache_path)
        assert disparity.shape == left_rgb.shape[:3], (
            f"Cached S2M2 disparity {cache_path} has shape {disparity.shape}, expected {left_rgb.shape[:3]}. "
            "Delete it to recompute."
        )
        print(f"[info] Reusing cached S2M2 disparity {cache_path}")
    else:
        stereo = S2M2StereoDepth(model_type, refine_iter=refine_iter, allow_negative=False, device="cuda")
        disparity = stereo.disparity_sequence(left_rgb, right_rgb, desc=f"S2M2 {camera}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so an interrupted run never leaves a truncated cache file behind
        tmp_path = cache_path.with_suffix(".npy.tmp")
        with open(tmp_path, "wb") as f:
            np.save(f, disparity)
        tmp_path.replace(cache_path)
        print(f"[info] Saved S2M2 disparity cache {cache_path}")
    return disparity_to_depth_m(disparity, fx, baseline_m)


def disparity_to_depth_m(disparity_px: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """``depth = fx * baseline / disparity``, elementwise for ``(H, W)`` or ``(T, H, W)``. Pixels with disparity <= 0
    get depth 0 (no return), matching the h5 depth convention."""
    assert disparity_px.ndim in (2, 3), f"disparity_px must be HxW or TxHxW, got {disparity_px.shape}"
    assert fx > 0, f"fx must be > 0, got {fx}"
    assert baseline_m > 0, f"baseline_m must be > 0, got {baseline_m}"
    valid = disparity_px > 0
    depth = np.zeros(disparity_px.shape, dtype=np.float32)
    depth[valid] = (fx * baseline_m / disparity_px[valid]).astype(np.float32)
    return depth
