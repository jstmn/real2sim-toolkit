import pathlib

import numpy as np
import pytest

from examples.generate_masks_across_trajectory import (
    Args,
    _propagate_masks,
    _union_top_sam_masks,
    main,
)


def test_generate_masks_across_trajectory_missing_h5_raises(tmp_path: pathlib.Path):
    with pytest.raises(AssertionError, match="not found"):
        main(
            Args(
                h5_path=tmp_path / "missing.h5",
                camera="cam_1",
                object_description="robot arm",
            )
        )


def test_generate_masks_across_trajectory_rejects_empty_object_description(tmp_path: pathlib.Path):
    h5_path = tmp_path / "merged.h5"
    h5_path.write_bytes(b"x")
    with pytest.raises(AssertionError, match="object_description"):
        main(Args(h5_path=h5_path, camera="cam_1", object_description=""))


def test_union_top_sam_masks_ors_masks_above_threshold():
    masks = np.zeros((3, 4, 5), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True
    masks[2, 2, 2] = True
    scores = np.array([0.9, 0.4, 0.2])
    union = _union_top_sam_masks(masks, scores, kmax=3, score_threshold=0.3)
    expected = np.zeros((4, 5), dtype=bool)
    expected[0, 0] = True
    expected[1, 1] = True
    assert np.array_equal(union, expected)


def test_union_top_sam_masks_rejects_none_above_threshold():
    masks = np.zeros((2, 3, 3), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True
    with pytest.raises(AssertionError, match="No SAM 2 masks"):
        _union_top_sam_masks(masks, np.array([0.2, 0.1]), kmax=2, score_threshold=0.3)


class _FakePropagatePredictor:
    def __init__(self):
        self.calls = 0

    def propagate_from_mask(self, image_bgr, mask_input, box_xyxy):
        self.calls += 1
        t = self.calls
        height, width = image_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[2 : 2 + t, 3 : 5 + t] = True
        low_res = np.full((1, 256, 256), float(t), dtype=np.float32)
        return mask, 0.9, low_res


def test_propagate_masks_keeps_seed_and_fills_later_frames():
    rgb = np.zeros((3, 16, 20, 3), dtype=np.uint8)
    seed = np.zeros((16, 20), dtype=bool)
    seed[4:8, 6:11] = True
    predictor = _FakePropagatePredictor()
    out = _propagate_masks(rgb, predictor, seed)
    assert out.shape == (3, 16, 20)
    assert np.array_equal(out[0], seed)
    assert predictor.calls == 2
    expected1 = np.zeros((16, 20), dtype=bool)
    expected1[2:3, 3:6] = True
    expected2 = np.zeros((16, 20), dtype=bool)
    expected2[2:4, 3:7] = True
    assert np.array_equal(out[1], expected1)
    assert np.array_equal(out[2], expected2)


def test_propagate_masks_rejects_empty_seed():
    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(AssertionError, match="empty"):
        _propagate_masks(rgb, _FakePropagatePredictor(), np.zeros((8, 8), dtype=bool))
