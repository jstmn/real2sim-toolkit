# Real2sim_toolkit

This repository is an API for creating and maintaining a digital twin of a physical environment.

## Installation

```bash
# Download model weights
wget --no-check-certificate https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O src/r2st/models/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O src/r2st/models/groundingdino_swint_ogc.pth
gcloud storage cp --recursive gs://r2st-public/2024-01-11-20-02-45 src/r2st/FoundationPose/weights/
gcloud storage cp --recursive gs://r2st-public/2023-10-28-18-33-37 src/r2st/FoundationPose/weights/

# Set your Meshy and OpenAI API keys (recommended to add to your ~/.bashrc)
export MESHY_API_KEY=your_meshy_api_key
export OPENAI_API_KEY=your_openai_api_key

# Build and start the FoundationPose docker container (docker installation steps at https://docs.docker.com/engine/install/ubuntu/). Note that you need `nvidia-container-toolkit` installed as well (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
# The image is built locally (not pulled) because it's built on CUDA 12.8 + PyTorch 2.8 to support
# current-gen GPUs (e.g. RTX 50-series/Blackwell); pinning to your driver/GPU isn't needed b/c
# CUDA 12.8 covers everything back through sm_70.
cd src/r2st/FoundationPose/docker
docker build --network host -t foundationpose -f dockerfile ..
bash run_container.sh
# In the docker container:
cd /real2sim-toolkit/src/r2st/FoundationPose && bash build_all.sh

# Clone ManiSkill and Jrl2
git clone git@github.com:jstmn/ManiSkill.git thirdparty/ManiSkill
git clone git@github.com:jstmn/Jrl2.git thirdparty/Jrl2

# Initialize uv
uv sync


uv run python src/r2st/GroundingDINO/setup.py develop

```

## Examples

**Example 1: Generate a mesh for the object in `data/red_T_block_1.png` and visualize it with viser:**

```bash
uv run python examples/generate_mesh.py --images data/red_T_block_1.png --visualize
uv run python examples/generate_mesh.py --images data/raise_cube_0__camera_base__t=0.rgb.png --visualize
```


**Example 2: Estimate camera extrinsics and save results to a yaml file.** This script runs the CMA-ES optimization procedure to estimate the extrinsics of a specified camera given RGBD images, joint angles, and the urdf of the robot (urdf from [Jrl2](https://github.com/jstmn/Jrl2)).
```bash
uv run python examples/estimate_camera_extrinsics.py \
  --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
  --robot-id xarm7 \
  --camera cam_1 \
  --camera-model-id d435 \
  --output-path data/demonstrations/0802/extrinsics.yaml \
  --visualize \
  --visualize-robot-masks
```


**Example 3: Generate a mesh for the "mustard bottle" seen in the first frame of a
demonstration, then track that object through the demonstration:**
Pass `--visualize` to start a viser server with the mesh and a timestep slider over predicted poses.
Note: FoundationPose runs inside the Docker container (see Installation above), but the rest of the toolkit runs on the host in the `uv` venv. 
`r2st.pose_grpc` bridges the two: a server (`r2st.pose_grpc.server`) runs inside the container and exposes `FoundationPoseTracker`'s `register`/`track` over gRPC; a client (`r2st.pose_grpc.client.FoundationPoseClient`) is used from host-side code (e.g. `examples/track_object.py`) to call it.

```bash
# First download the saved demonstrations to data/0802
mkdir -p data
gcloud storage cp --recursive gs://r2st-public/demonstrations/ data/


# Start the server (`cd src/r2st/FoundationPose/docker; bash run_container.sh`), then in the container:
cd /real2sim-toolkit/src && python -m r2st.pose_grpc.server

uv run python examples/track_object.py \
    --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
    --camera cam_1 \
    --camera-model-id d435 \
    --object-description "mustard bottle" \
    --visualize

uv run python examples/track_object.py \
    --h5-path data/demonstrations/raise_cube_0_merged.h5 \
    --camera camera_north \
    --camera-model-id d435 \
    --object-description "blue cube" \
    --visualize
```

## Camera models

`--camera-model-id` selects measured intrinsics (and depth-to-color extrinsics) from `r2st.constants`.

| `--camera-model-id` | Device |
| --- | --- |
| `d435` | Intel RealSense D435 |

## Pose tracking (gRPC)

```bash
```

Then, on the host:

```python
from r2st.pose_grpc.client import FoundationPoseClient

client = FoundationPoseClient("localhost:50051")  # --network=host, so localhost reaches the container
pose_cam = client.register(mesh_path, color_rgb, depth_m, mask, K)  # first frame
pose_cam = client.track(color_rgb, depth_m, K)  # every subsequent frame
```

If you change `src/r2st/pose_grpc/pose_tracking.proto`, regenerate the Python bindings from
`src/r2st/pose_grpc/`:

```bash
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pose_tracking.proto
```

then re-apply the `from r2st.pose_grpc import ...` fix noted at the top of
`pose_tracking_pb2_grpc.py` (protoc emits a bare top-level import that breaks once the file is
imported as a package).

## Third party models used

* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [Grounding DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/GroundingDINO)
* [FoundationPose](https://github.com/OpenGVLab/FoundationPose)
* [Cutie](https://github.com/hkchengrex/Cutie) — optional 2D tracker used by `FoundationPoseTracker(use_2d_tracker=True)` to
  re-anchor FoundationPose's translation each frame (ported from
  [FoundationPose++](https://github.com/lidingsheng/FoundationPose-plus-plus))



## Common issues

Error: `docker: Error response from daemon: AMD CDI spec not found`. Fix:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -iA3 Runtimes   # confirm "nvidia" shows up
```

Error building `kaolin`/`pytorch3d` inside the container: `nvcc fatal: Unsupported gpu architecture 'compute_XX'` or a PyTorch warning that your GPU's CUDA capability (e.g. `sm_120` for RTX 50-series/Blackwell) isn't supported by the current PyTorch install. This means the image's CUDA/PyTorch versions predate your GPU's architecture — rebuild the image from `src/r2st/FoundationPose/docker/dockerfile` (already pinned to CUDA 12.8 + PyTorch 2.8, which covers everything through Blackwell); don't `docker pull` an older prebuilt image.