import numpy as np
import pytest
import torch

from r2st.stereo_depth import S2M2_WEIGHTS_DIR, S2M2StereoDepth, disparity_to_depth_m


class TestDisparityToDepth:
    def test_depth_from_disparity(self):
        disp = np.array([[10.0, 20.0], [0.0, -1.0]], dtype=np.float32)
        depth = disparity_to_depth_m(disp, fx=500.0, baseline_m=0.12)
        assert depth.dtype == np.float32
        np.testing.assert_allclose(depth, [[6.0, 3.0], [0.0, 0.0]], rtol=1e-6)

    def test_rejects_bad_baseline(self):
        with pytest.raises(AssertionError):
            disparity_to_depth_m(np.ones((2, 2), dtype=np.float32), fx=500.0, baseline_m=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not (S2M2_WEIGHTS_DIR / "CH128NTR1.pth").is_file(), reason="needs S2M2 S checkpoint")
class TestS2M2StereoDepth:
    def test_shifted_pair(self):
        """A right image that is the left shifted by a constant should give roughly that disparity."""
        rng = np.random.default_rng(0)
        shift_px = 8
        texture = (rng.random((376, 672 + shift_px, 3)) * 255).astype(np.uint8)
        # A left pixel at x appears in the right image at x - disparity
        left = np.ascontiguousarray(texture[:, :-shift_px])
        right = np.ascontiguousarray(texture[:, shift_px:])
        result = S2M2StereoDepth("S", refine_iter=3, allow_negative=False, device="cuda").predict(left, right)
        assert result.disparity_px.shape == (376, 672)
        interior = result.disparity_px[32:-32, 64:-64]
        assert abs(float(np.median(interior)) - shift_px) < 0.5, f"median disparity {np.median(interior)}"

    def test_rejects_mismatched_pair(self):
        est = S2M2StereoDepth("S", refine_iter=1, allow_negative=False, device="cuda")
        with pytest.raises(AssertionError):
            est.predict(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 32, 3), np.uint8))
