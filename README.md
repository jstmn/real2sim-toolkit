# Real2sim_toolkit

This repository is an API for creating and maintaining a digital twin of a physical environment.

## Installation

```bash
uv sync

# Download model weights
wget --no-check-certificate https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O src/r2st/models/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O src/r2st/models/groundingdino_swint_ogc.pth


# Set your Meshy and OpenAI API keys (recommended to add to your ~/.bashrc)
export MESHY_API_KEY=your_meshy_api_key
export OPENAI_API_KEY=your_openai_api_key
```

## Examples

Example 1: Generate a mesh for the object in `data/red_T_block_1.png` and visualize it with viser:

```bash
uv run python scripts/main.py --image data/red_T_block_1.png --visualize
```


## Third party models used

* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [Grounding DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/GroundingDINO)