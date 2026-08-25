import pathlib

import pytest

from examples.generate_mesh import Args, main


def test_generate_mesh_missing_image_raises(tmp_path: pathlib.Path):
    with pytest.raises(AssertionError, match="not found"):
        main(Args(images=[tmp_path / "missing.png"], object_description="red block"))


def test_generate_mesh_rejects_empty_object_description(tmp_path: pathlib.Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"x")
    with pytest.raises(AssertionError, match="object_description"):
        main(Args(images=[image], object_description=""))


def test_generate_mesh_requires_exactly_one_description_mode(tmp_path: pathlib.Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"x")
    with pytest.raises(AssertionError, match="exactly one"):
        main(Args(images=[image]))
    with pytest.raises(AssertionError, match="exactly one"):
        main(Args(images=[image], object_description="red block", object_description_from_vlm=True))
