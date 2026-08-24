import numpy as np
import pytest


class TestCoreHelpers:
    def test_save_mask_image_2(self, tmp_path):
        from r2st.core import save_mask_image_2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True
        out = tmp_path / "out.jpg"
        save_mask_image_2(img, mask, str(out), "hello")
        assert out.exists()

    def test_save_mask_image_2_bad_shape_raises(self, tmp_path):
        from r2st.core import save_mask_image_2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(AssertionError):
            save_mask_image_2(img, mask, str(tmp_path / "o.jpg"), "hi")

    def test_get_camera_extrinsic(self):
        from r2st.core import get_camera_extrinsic

        extr = {"camera_south": np.eye(4)}
        m = get_camera_extrinsic("camera_south", extr)
        assert m.shape == (4, 4)
        with pytest.raises(AssertionError):
            get_camera_extrinsic("unknown", extr)

    def test_get_camera_extrinsic_object_style(self):
        from r2st.core import get_camera_extrinsic
        from r2st.types import CameraExtrinsics

        ce = CameraExtrinsics(matrix=np.eye(4))
        m = get_camera_extrinsic("camera_south", {"camera_south": ce})
        assert m.shape == (4, 4)


def test_bbox_xyxy_from_mask():
    from r2st.core import bbox_xyxy_from_mask

    mask = np.zeros((20, 30), dtype=bool)
    mask[2:5, 7:10] = True
    box = bbox_xyxy_from_mask(mask)
    assert box.tolist() == [7.0, 2.0, 10.0, 5.0]


def test_bbox_xyxy_from_empty_mask_raises():
    from r2st.core import bbox_xyxy_from_mask

    with pytest.raises(AssertionError, match="empty"):
        bbox_xyxy_from_mask(np.zeros((4, 4), dtype=bool))


def test_binary_mask_to_sam_mask_input_shape():
    from r2st.core import SAM_MASK_INPUT_HW, binary_mask_to_sam_mask_input

    mask = np.zeros((48, 64), dtype=bool)
    mask[10:30, 20:40] = True
    mask_input = binary_mask_to_sam_mask_input(mask)
    assert mask_input.shape == (1, SAM_MASK_INPUT_HW, SAM_MASK_INPUT_HW)
    assert mask_input.dtype == np.float32
    assert float(mask_input.max()) == 1.0
    assert float(mask_input.min()) == 0.0
