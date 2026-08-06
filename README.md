# Real2sim_toolkit

This repository is an API for creating and maintaining a digital twin of a physical environment.

## Installation

```bash
uv sync

# 
wget --no-check-certificate https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O src/r2st/models/sam_vit_h_4b8939.pth

# Set your Meshy and OpenAI API keys (recommended to add to your ~/.bashrc)
export MESHY_API_KEY=your_meshy_api_key
export OPENAI_API_KEY=your_openai_api_key
```

## Third party models used

* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [Grounding DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/GroundingDINO)
