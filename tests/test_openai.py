from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestListObjectsInImage:
    def _mock_openai(self, content="red cup . blue box"):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=content))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def test_numpy_input(self):
        from r2st.openai import list_objects_in_image

        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        mock_client = self._mock_openai("red cup . blue box")
        with patch("r2st.openai.OpenAI", return_value=mock_client):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
                result = list_objects_in_image(img)
        assert result == ["red cup", "blue box"]

    def test_path_string_input(self, tmp_path):
        from PIL import Image

        from r2st.openai import list_objects_in_image

        p = tmp_path / "img.jpg"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(p)
        mock_client = self._mock_openai("green block")
        with patch("r2st.openai.OpenAI", return_value=mock_client):
            result = list_objects_in_image(str(p))
        assert result == ["green block"]

    def test_invalid_numpy_shape_raises(self):
        from r2st.openai import list_objects_in_image

        bad = np.ones((10, 10), dtype=np.uint8)  # missing channel
        with pytest.raises(AssertionError):
            list_objects_in_image(bad)

    def test_invalid_numpy_channels_raises(self):
        from r2st.openai import list_objects_in_image

        bad = np.ones((10, 10, 4), dtype=np.uint8)
        with pytest.raises(AssertionError):
            list_objects_in_image(bad)

    def test_trims_whitespace(self):
        from r2st.openai import list_objects_in_image

        img = np.ones((5, 5, 3), dtype=np.uint8)
        mock_client = self._mock_openai("  red cup  .  blue box  ")
        with patch("r2st.openai.OpenAI", return_value=mock_client):
            result = list_objects_in_image(img)
        assert result == ["red cup", "blue box"]


class TestQueryOpenaiScript:
    def test_script_main_calls_vlm(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch

        from PIL import Image
        from scripts.query_openai import Args, main

        p = tmp_path / "img.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)
        args = Args(image=Path(p))
        with patch("scripts.query_openai.list_objects_in_image", return_value=["red cup"]):
            main(args)  # should not raise, just print

    def test_missing_image_raises(self, tmp_path):
        from pathlib import Path

        from scripts.query_openai import Args, main

        args = Args(image=Path(tmp_path) / "nonexistent.jpg")
        with pytest.raises(AssertionError):
            main(args)


class TestLiveOpenAI:
    @pytest.mark.skipif(
        not __import__("os").getenv("OPENAI_API_KEY")
        or __import__("os").getenv("OPENAI_API_KEY") in ("dummy", "test", ""),
        reason="OPENAI_API_KEY not set or is dummy — skipping live API test",
    )
    def test_red_block_image_contains_red_or_block(self):
        from pathlib import Path

        from r2st.openai import list_objects_in_image

        # Image shows a red T-shaped block on a white table (saved from issue attachment)
        img_path = Path(__file__).parent / "data" / "red_T_block.jpg"
        assert img_path.exists(), f"Live test image not found: {img_path}"

        # Also verify numpy path works — load via PIL to exercise encode_numpy_as_data_url
        import numpy as np
        from PIL import Image

        pil = Image.open(img_path).convert("RGB")
        img_np = np.array(pil)
        assert img_np.shape[2] == 3

        # Call live API (uses path string variant — tests encode_image_as_data_url)
        result = list_objects_in_image(str(img_path))
        assert len(result) >= 1, f"Expected at least one object, got {result}"
        assert any(
            "red" in name.lower() or "block" in name.lower() for name in result
        ), f"Expected 'red' or 'block' in one of {result}"

        # Also exercise numpy variant on same image to ensure both paths agree
        result_np = list_objects_in_image(img_np)
        assert any(
            "red" in name.lower() or "block" in name.lower() for name in result_np
        ), f"Expected 'red' or 'block' in one of {result_np} (numpy path)"
