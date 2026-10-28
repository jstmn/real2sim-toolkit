import numpy as np
import pytest
import torch

from r2st.stereo_depth import (
    S2M2_WEIGHTS_DIR,
    S2M2StereoDepth,
    S2M2TensorRTStereoDepth,
    disparity_to_depth_m,
    s2m2_tensorrt_engine_path,
)


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


class TestCachedS2M2Depth:
    def test_caches_disparity_and_reuses_it(self, tmp_path, monkeypatch):
        from r2st import stereo_depth

        n_constructed = []

        class FakeStereo:
            def __init__(self, model_type, refine_iter, allow_negative, device):
                n_constructed.append((model_type, refine_iter))

            def disparity_sequence(self, left_rgb, right_rgb, desc):
                return np.full(left_rgb.shape[:3], 10.0, dtype=np.float32)

        monkeypatch.setattr(stereo_depth, "S2M2StereoDepth", FakeStereo)
        monkeypatch.setattr(stereo_depth, "s2m2_checkpoint_id", lambda model_type: f"CKPT_{model_type}-0123456789ab")
        h5_path = tmp_path / "traj_0.h5"
        frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)

        depth = stereo_depth.cached_s2m2_depth_m(h5_path, "zed", frames, frames, 500.0, 0.12, "L", 3)
        cache_path = stereo_depth.s2m2_disparity_cache_path(h5_path, "zed", "CKPT_L-0123456789ab", 3, 3)
        assert (
            cache_path
            == tmp_path / "traj_0" / "s2m2_disparity" / "zed__model=CKPT_L-0123456789ab__refine_iter=3__frames=3.npy"
        )
        assert cache_path.is_file() and not cache_path.with_suffix(".npy.tmp").exists()
        np.testing.assert_allclose(depth, 6.0)
        assert n_constructed == [("L", 3)]

        # Second call: read from the cache, no model; depth uses the new fx/baseline
        depth = stereo_depth.cached_s2m2_depth_m(h5_path, "zed", frames, frames, 250.0, 0.12, "L", 3)
        np.testing.assert_allclose(depth, 3.0)
        assert n_constructed == [("L", 3)]

    def test_rejects_cache_with_wrong_shape(self, tmp_path, monkeypatch):
        from r2st import stereo_depth

        monkeypatch.setattr(stereo_depth, "s2m2_checkpoint_id", lambda model_type: f"CKPT_{model_type}-0123456789ab")
        h5_path = tmp_path / "traj_0.h5"
        frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
        cache_path = stereo_depth.s2m2_disparity_cache_path(h5_path, "zed", "CKPT_L-0123456789ab", 3, 3)
        cache_path.parent.mkdir(parents=True)
        np.save(cache_path, np.ones((3, 4, 6), dtype=np.float32))
        with pytest.raises(AssertionError, match="Delete it to recompute"):
            stereo_depth.cached_s2m2_depth_m(h5_path, "zed", frames, frames, 500.0, 0.12, "L", 3)


def test_checkpoint_id_matches_file_sha256(tmp_path, monkeypatch):
    import hashlib

    from r2st import stereo_depth

    monkeypatch.setattr(stereo_depth, "S2M2_WEIGHTS_DIR", tmp_path)
    ckpt = tmp_path / "CH256NTR3.pth"
    ckpt.write_bytes(b"weights v1")
    assert stereo_depth.s2m2_checkpoint_id("L") == f"CH256NTR3-{hashlib.sha256(b'weights v1').hexdigest()[:12]}"
    ckpt.write_bytes(b"weights v2")
    assert stereo_depth.s2m2_checkpoint_id("L") == f"CH256NTR3-{hashlib.sha256(b'weights v2').hexdigest()[:12]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tensorrt_engine_path_names_its_build(monkeypatch):
    pytest.importorskip("tensorrt")
    import tensorrt as trt

    from r2st import stereo_depth

    monkeypatch.setattr(stereo_depth, "s2m2_checkpoint_id", lambda model_type: f"CKPT_{model_type}-0123456789ab")
    path = s2m2_tensorrt_engine_path("XL", 3, False, (376, 672), "cuda")
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    assert path.parent == stereo_depth.S2M2_TENSORRT_DIR
    assert (
        path.name
        == f"CKPT_XL-0123456789ab__refine_iter=3__allow_negative=False__672x376__trt{trt.__version__}__{gpu}.engine"
    )
    assert stereo_depth._padded_hw((376, 672)) == (384, 672)
    with pytest.raises(AssertionError):
        s2m2_tensorrt_engine_path("XL", 0, False, (376, 672), "cuda")


def _xl_vga_engine_built() -> bool:
    """Whether the XL, refine_iter=3, 672x376 engine is built for this env's TensorRT and GPU."""
    import importlib.util

    if not torch.cuda.is_available() or not (S2M2_WEIGHTS_DIR / "CH384NTR3.pth").is_file():
        return False
    if importlib.util.find_spec("tensorrt") is None:
        return False
    return s2m2_tensorrt_engine_path("XL", 3, False, (376, 672), "cuda").is_file()


@pytest.mark.skipif(
    not _xl_vga_engine_built(), reason="needs the XL 672x376 TensorRT engine (build_s2m2_tensorrt_engine)"
)
class TestS2M2TensorRTStereoDepth:
    def test_matches_pytorch_on_shifted_pair(self):
        """The engine is the PyTorch model under the same fp16 autocast, so their disparities nearly agree."""
        rng = np.random.default_rng(0)
        shift_px = 8
        texture = (rng.random((376, 672 + shift_px, 3)) * 255).astype(np.uint8)
        left = np.ascontiguousarray(texture[:, :-shift_px])
        right = np.ascontiguousarray(texture[:, shift_px:])
        disp_trt = S2M2TensorRTStereoDepth("XL", 3, False, (376, 672), "cuda").predict(left, right).disparity_px
        disp_torch = S2M2StereoDepth("XL", refine_iter=3, allow_negative=False, device="cuda").predict(left, right)
        assert disp_trt.shape == (376, 672)
        interior = disp_trt[32:-32, 64:-64]
        assert abs(float(np.median(interior)) - shift_px) < 0.5, f"median disparity {np.median(interior)}"
        err = np.abs(disp_trt - disp_torch.disparity_px)[32:-32, 64:-64]
        assert (
            np.median(err) < 0.05 and np.percentile(err, 99) < 1.0
        ), f"median {np.median(err)}, p99 {np.percentile(err, 99)}"

    def test_rejects_other_image_size(self):
        est = S2M2TensorRTStereoDepth("XL", 3, False, (376, 672), "cuda")
        with pytest.raises(AssertionError, match="the engine takes"):
            est.predict(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640, 3), np.uint8))
