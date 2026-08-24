import pathlib

import pytest

from examples.generate_mask import Args, main


def test_generate_mask_missing_image_raises(tmp_path: pathlib.Path):
    with pytest.raises(AssertionError, match="not found"):
        main(Args(image=tmp_path / "missing.png", object_description="red T block"))


def test_generate_mask_rejects_empty_object_description(tmp_path: pathlib.Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"x")
    with pytest.raises(AssertionError, match="object_description"):
        main(Args(image=image, object_description=""))
