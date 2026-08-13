"""Temporary converter: raise_cube_0.h5 ({cam}__color/depth_image) -> merged obs/sensor_data layout.

uv run python examples/convert_raise_cube_h5.py --input data/raise_cube_0.h5
"""

import dataclasses
import pathlib

import h5py
import numpy as np
import tyro

_CAMERAS = ("north", "base", "eih")


@dataclasses.dataclass
class Args:
    input: pathlib.Path
    """Source raise-cube h5 (flat `{cam}__color_image` / `{cam}__depth_image` datasets)."""

    output: pathlib.Path | None = None
    """Destination merged h5. Defaults to <input_stem>_merged.h5 next to the input."""


def main(args: Args) -> None:
    assert args.input.is_file(), f"Input not found: {args.input}"
    output = args.output if args.output is not None else args.input.with_name(f"{args.input.stem}_merged.h5")
    tmp_output = output.with_name(output.name + ".tmp")

    with h5py.File(args.input, "r") as src, h5py.File(tmp_output, "w") as dst:
        for cam in _CAMERAS:
            color_key = f"{cam}__color_image"
            depth_key = f"{cam}__depth_image"
            assert color_key in src, f"{args.input}: missing {color_key}. keys: {list(src.keys())}"
            assert depth_key in src, f"{args.input}: missing {depth_key}. keys: {list(src.keys())}"
            rgb = src[color_key][:]
            depth = src[depth_key][:]
            assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"{color_key} must be NxHxWx3, got {rgb.shape}"
            assert rgb.dtype == np.uint8, f"{color_key} must be uint8, got {rgb.dtype}"
            assert depth.ndim == 3, f"{depth_key} must be NxHxW, got {depth.shape}"
            assert depth.shape[0] == rgb.shape[0], f"{cam}: rgb {rgb.shape[0]} frames vs depth {depth.shape[0]}"
            assert depth.shape[1:] == rgb.shape[1:3], f"{cam}: depth {depth.shape[1:]} != rgb {rgb.shape[1:3]}"
            n = rgb.shape[0]
            assert n > 0, f"{cam}: no frames"
            timestamps_ms = np.arange(n, dtype=np.float64)
            camera_name = f"camera_{cam}"
            group = dst.create_group(f"obs/sensor_data/{camera_name}")
            group.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=4)
            group.create_dataset("depth", data=depth, compression="gzip", compression_opts=4)
            group.create_dataset("timestamp_ms", data=timestamps_ms)
            group.create_dataset("rgb_timestamp_ms", data=timestamps_ms)
            print(f"[info] Wrote obs/sensor_data/{camera_name}: rgb{rgb.shape}, depth{depth.shape}")

    tmp_output.rename(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
