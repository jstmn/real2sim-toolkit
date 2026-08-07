# Real2sim_toolkit

This repository is an API for creating and maintaining a digital twin of a physical environment.

## Installation

```bash
# Download model weights
wget --no-check-certificate https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O src/r2st/models/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O src/r2st/models/groundingdino_swint_ogc.pth

# Set your Meshy and OpenAI API keys (recommended to add to your ~/.bashrc)
export MESHY_API_KEY=your_meshy_api_key
export OPENAI_API_KEY=your_openai_api_key

# Download FoundationPose scorer/refiner weights and put them under src/r2st/FoundationPose/weights/
# (refiner: 2023-10-28-18-33-37, scorer: 2024-01-11-20-02-45)
# https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i

# Build and start the FoundationPose docker container (docker installation steps at https://docs.docker.com/engine/install/ubuntu/). Note that you need `nvidia-container-toolkit` installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
# The image is built locally (not pulled) because it's built on CUDA 12.8 + PyTorch 2.8 to support
# current-gen GPUs (e.g. RTX 50-series/Blackwell); pinned to your driver/GPU is not needed since
# CUDA 12.8 covers everything back through sm_70.
cd src/r2st/FoundationPose/docker
docker build --network host -t foundationpose -f dockerfile ..
bash run_container.sh
bash build_all.sh # <-- IN THE DOCKER CONTAINER (builds the mycpp extension; kaolin/pytorch3d/nvdiffrast are already baked into the image)


# Initialize uv
uv sync
```

## Examples

Example 1: Generate a mesh for the object in `data/red_T_block_1.png` and visualize it with viser:

```bash
uv run python scripts/generate_mesh.py --image data/red_T_block_1.png --visualize
```

Example 2: Track the object in :


## Third party models used

* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [Grounding DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/GroundingDINO)
* [FoundationPose](https://github.com/OpenGVLab/FoundationPose)



## Common issues

Error: `docker: Error response from daemon: AMD CDI spec not found`. Fix:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -iA3 Runtimes   # confirm "nvidia" shows up
```

Error building `kaolin`/`pytorch3d` inside the container: `nvcc fatal: Unsupported gpu architecture 'compute_XX'` or a PyTorch warning that your GPU's CUDA capability (e.g. `sm_120` for RTX 50-series/Blackwell) isn't supported by the current PyTorch install. This means the image's CUDA/PyTorch versions predate your GPU's architecture — rebuild the image from `src/r2st/FoundationPose/docker/dockerfile` (already pinned to CUDA 12.8 + PyTorch 2.8, which covers everything through Blackwell); don't `docker pull` an older prebuilt image.