import dataclasses
import pathlib

import tyro

from r2st.openai import list_objects_in_image

"""

# Example usage:
uv run python scripts/query_openai.py --image data/red_T_block_1.png
"""


@dataclasses.dataclass
class Args:
    image: pathlib.Path
    """Path to input image (JPG/PNG)."""

    model: str = "gpt-4o-mini"
    """OpenAI model to use (currently handled by library)."""


def main(args: Args) -> None:
    assert args.image.exists(), f"Image file '{args.image}' not found"
    assert args.image.is_file(), f"Image path '{args.image}' is not a file"
    results = list_objects_in_image(str(args.image))
    print("✅ Object list generated successfully!")
    print(f"Objects: {' . '.join(results)}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
