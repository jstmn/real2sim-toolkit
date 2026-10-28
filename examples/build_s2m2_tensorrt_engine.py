import dataclasses
import time

import numpy as np
import torch
import tyro

from r2st.stereo_depth import S2M2TensorRTStereoDepth, build_s2m2_tensorrt_engine

"""
Build a TensorRT engine of S2M2 for one image size (r2st.stereo_depth.build_s2m2_tensorrt_engine), then time it on a
pair. The engine is saved in src/r2st/S2M2/weights/tensorrt/, named by checkpoint, refine_iter, allow_negative,
image size, TensorRT version and GPU (r2st.stereo_depth.s2m2_tensorrt_engine_path), and checked on a pair whose right
image is the left shifted by 8 px. It runs only with the TensorRT
version and GPU it was built with, so build it from the env that runs it: e.g. Mpc-4-Manipulation's tracker runs
S2M2TensorRTStereoDepth in that repo's env, with that env's tensorrt.

# Example usage (the ZED VGA image size, 672x376):
    uv run python examples/build_s2m2_tensorrt_engine.py \
        --model-type XL --refine-iter 3 --allow-negative False --height 376 --width 672 --device cuda:0
"""

# Pairs timed after building, after as many warm-up pairs
_N_TIMED = 20

# Horizontal shift, in pixels, between the left and right images of the pair the new engine is checked on
_CHECK_SHIFT_PX = 8


@dataclasses.dataclass
class Args:
    model_type: str
    """S2M2 model size (S, M, L, XL); its checkpoint must be downloaded."""

    refine_iter: int
    """S2M2 local refinement iterations, compiled into the engine."""

    allow_negative: bool
    """Allow negative disparity (S2M2's use_positivity off), compiled into the engine."""

    height: int
    """Image height in pixels (unpadded), e.g. 376 for ZED VGA."""

    width: int
    """Image width in pixels (unpadded), e.g. 672 for ZED VGA."""

    device: str
    """CUDA device to build and time on, e.g. cuda:0."""


def main(args: Args) -> None:
    image_hw = (args.height, args.width)
    t0 = time.monotonic()
    build_s2m2_tensorrt_engine(args.model_type, args.refine_iter, args.allow_negative, image_hw, args.device)
    print(f"[info] Built in {time.monotonic() - t0:.0f} s")

    stereo = S2M2TensorRTStereoDepth(args.model_type, args.refine_iter, args.allow_negative, image_hw, args.device)
    # A right image that is the left shifted by a constant: the engine must find that disparity
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 256, (args.height, args.width + _CHECK_SHIFT_PX, 3), dtype=np.uint8)
    left = np.ascontiguousarray(texture[:, :-_CHECK_SHIFT_PX])
    right = np.ascontiguousarray(texture[:, _CHECK_SHIFT_PX:])
    interior = stereo.predict(left, right).disparity_px[32:-32, 64:-64]
    median_px = float(np.median(interior))
    assert abs(median_px - _CHECK_SHIFT_PX) < 0.5, (
        f"The engine gives a median disparity of {median_px:.2f} px on a pair shifted by {_CHECK_SHIFT_PX} px; it is "
        "broken. Delete it and investigate."
    )
    print(f"[info] Check passed: median disparity {median_px:.2f} px on a pair shifted by {_CHECK_SHIFT_PX} px")
    for _ in range(_N_TIMED):
        stereo.predict(left, right)
    torch.cuda.synchronize(stereo.device)
    t0 = time.perf_counter()
    for _ in range(_N_TIMED):
        stereo.predict(left, right)
    ms = (time.perf_counter() - t0) / _N_TIMED * 1e3
    print(f"[info] {ms:.1f} ms per {args.width}x{args.height} pair (predict, including host copies)")


if __name__ == "__main__":
    main(tyro.cli(Args))
