import base64
import mimetypes
import os
import time
from collections.abc import Sequence
from pathlib import Path

import requests

MESHY_API_BASE = "https://api.meshy.ai/openapi/v1"
MESHY_API_KEY = os.getenv("MESHY_API_KEY", "")
_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S = 600.0
MESHY_MODEL_ID = "latest"


def _image_path_to_data_uri(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    assert mime in ("image/jpeg", "image/png"), f"Unsupported image MIME type {mime!r} for {image_path}"
    data = image_path.read_bytes()
    assert len(data) > 0, f"Image file is empty: {image_path}"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class MeshyAPI:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else MESHY_API_KEY
        assert self.api_key is not None and len(self.api_key) > 0, "MESHY_API_KEY is not set"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _create_task(
        self,
        image_urls: list[str],
        *,
        enable_pbr: bool,
    ) -> str:
        assert 1 <= len(image_urls) <= 4, f"multi-image-to-3d accepts 1-4 images, got {len(image_urls)}"
        payload = {
            "image_urls": image_urls,
            "image_enhancement": True,
            "pose_mode": "",
            "enable_pbr": enable_pbr,
            "target_formats": ["glb"],
            "ai_model": MESHY_MODEL_ID,
            "should_remesh": False,
        }
        response = requests.post(
            f"{MESHY_API_BASE}/multi-image-to-3d",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        assert "result" in body, f"Meshy create response missing 'result': {body}"
        task_id = body["result"]
        assert isinstance(task_id, str) and len(task_id) > 0, f"Invalid Meshy task id: {task_id!r}"
        return task_id

    def _poll_task(self, task_id: str) -> dict:
        from tqdm import tqdm

        deadline = time.monotonic() + _POLL_TIMEOUT_S
        with tqdm(total=100, desc=f"Meshy {task_id[:8]}", unit="%") as pbar:
            last_progress = 0
            while True:
                response = requests.get(
                    f"{MESHY_API_BASE}/multi-image-to-3d/{task_id}",
                    headers=self._headers(),
                    timeout=60,
                )
                response.raise_for_status()
                task = response.json()
                status = task.get("status")
                progress = task.get("progress")
                assert status is not None, f"Meshy task missing status: {task}"
                if isinstance(progress, (int, float)):
                    progress_i = max(0, min(100, int(progress)))
                    if progress_i > last_progress:
                        pbar.update(progress_i - last_progress)
                        last_progress = progress_i
                pbar.set_postfix(status=status)
                if status == "SUCCEEDED":
                    if last_progress < 100:
                        pbar.update(100 - last_progress)
                    return task
                if status == "FAILED":
                    error = task.get("task_error", {})
                    raise RuntimeError(f"Meshy multi-image-to-3d failed for {task_id}: {error}")
                assert time.monotonic() < deadline, f"Meshy task {task_id} timed out after {_POLL_TIMEOUT_S}s"
                time.sleep(_POLL_INTERVAL_S)

    def _download_glb(self, glb_url: str, output_path: Path) -> None:
        from tqdm import tqdm

        with requests.get(glb_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total is not None else None
            chunks: list[bytes] = []
            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading GLB",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    pbar.update(len(chunk))
            data = b"".join(chunks)
        assert len(data) > 0, f"Downloaded GLB is empty from {glb_url}"
        output_path.write_bytes(data)

    def image_to_3d(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        enable_pbr: bool = True,
    ) -> str:
        image_paths = [Path(p) for p in image_paths]
        assert len(image_paths) > 0, "image_paths must not be empty"
        output_dir = Path(output_dir)
        for image_path in image_paths:
            assert image_path.is_file(), f"Image file '{image_path}' not found"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_urls = [_image_path_to_data_uri(image_path) for image_path in image_paths]
        task_id = self._create_task(image_urls, enable_pbr=enable_pbr)
        print(f"[info] Created Meshy image-to-3d task: {task_id}")
        task = self._poll_task(task_id)
        model_urls = task.get("model_urls")
        assert isinstance(model_urls, dict), f"Meshy task missing model_urls: {task}"
        glb_url = model_urls.get("glb")
        assert isinstance(glb_url, str) and len(glb_url) > 0, f"Meshy task missing glb URL: {model_urls}"

        glb_path = output_dir / "model_glb.glb"
        self._download_glb(glb_url, glb_path)
        assert glb_path.is_file() and glb_path.stat().st_size > 0, f"Failed to write GLB to {glb_path}"
        print(f"[info] Saved Meshy GLB to {glb_path} ({glb_path.stat().st_size} bytes)")
        return str(glb_path)
