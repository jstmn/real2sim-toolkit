import base64
import io
import mimetypes
import os

import numpy as np
from openai import OpenAI
from PIL import Image

API_KEY = os.getenv("OPENAI_API_KEY")


def encode_image_as_data_url(path):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_numpy_as_data_url(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 data URL."""
    # Ensure the array is in uint8 format
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def list_objects_in_image(image: np.ndarray | str) -> list[str]:
    if isinstance(image, np.ndarray):
        assert len(image.shape) == 3 and image.shape[2] == 3, "Image array must be HxWx3 (RGB)"
        data_url = encode_numpy_as_data_url(image)
    else:
        data_url = encode_image_as_data_url(image)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY is not set"
    client = OpenAI(api_key=api_key)
    prompt = """
        Identify only the main solid objects on the white table as discrete physical items that can be picked up or manipulated.
        For each object, provide only the main color and the overall object type (e.g., "red cup", "blue box").
        Do NOT describe patterns, decorations, textures, or sub-components of objects.
        Do NOT segment a single object into multiple parts (e.g., don't describe "cup handle" separately from "cup").
        Separate each item with " . ", such as "black stapler . red cup . blue box".
        Ignore the white robot base and any background elements.
        Focus only on moveable objects that are clearly separate from each other.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=1200,
    )
    resp_raw = resp.choices[0].message.content
    return [obj.strip() for obj in resp_raw.split(" . ")]
