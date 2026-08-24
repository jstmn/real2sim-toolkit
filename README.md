# Real2sim_toolkit

This repository is an API for creating and maintaining a digital twin of a physical environment.

## Installation

```bash
# Download FoundationPose weights
gcloud storage cp --recursive gs://r2st-public/2024-01-11-20-02-45 src/r2st/FoundationPose/weights/
gcloud storage cp --recursive gs://r2st-public/2023-10-28-18-33-37 src/r2st/FoundationPose/weights/

# Set your Meshy and OpenAI API keys (recommend adding to ~/.bashrc)
export MESHY_API_KEY=your_meshy_api_key
export OPENAI_API_KEY=your_openai_api_key

# Clone this repo with submodules (`git clone --recurse-submodules ...`), or init them after clone:
git submodule update --init src/r2st/sam3

# Clone ManiSkill and Jrl2 (required before `uv sync`)
git clone git@github.com:jstmn/ManiSkill.git thirdparty/ManiSkill
git clone git@github.com:jstmn/Jrl2.git thirdparty/Jrl2

# SAM 3 checkpoints are gated. Request access at https://huggingface.co/facebook/sam3
# then authenticate after `uv sync` (which installs the local `src/r2st/sam3` package):

# Initialize uv
uv sync
uv run hf auth login

# REQUIRED: Get sam3 access at https://huggingface.co/facebook/sam3

# Build and start the FoundationPose docker container (docker installation steps at https://docs.docker.com/engine/install/ubuntu/). Note that you need `nvidia-container-toolkit` installed as well (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
# The image is built locally (not pulled) because it's built on CUDA 12.8 + PyTorch 2.8 to support
# current-gen GPUs (e.g. RTX 50-series/Blackwell). That covers GPU *architectures* back through
# sm_70; the host NVIDIA *driver* still needs to be new enough for CUDA 12.8 (Linux driver >= 570).
# Check with `nvidia-smi` (CUDA Version in the header). Driver 550 is CUDA 12.4 and will fail
# inside the container with CUDA error 804.
cd src/r2st/FoundationPose/docker
docker build --network host -t foundationpose -f dockerfile ..
bash run_container.sh
# In the docker container:
cd /real2sim-toolkit/src/r2st/FoundationPose && bash build_all.sh


### OPTIONAL
# Download saved demonstrations (used by Examples 2, 4, and 5). You need to be logged into Google Cloud Platform to do this. Run `gcloud auth login` to do so.
mkdir -p data
gcloud storage cp --recursive gs://r2st-public/demonstrations/ data/
```

## Examples

**Example 1: Segment an object with SAM 3.** This is the smoke test for text-prompted masking. It writes a dimmed overlay, a masked crop, and a bool `.npy` mask. The first run downloads the gated SAM 3 checkpoint from Hugging Face (see Installation).

```bash
uv run python examples/generate_mask.py --image data/red_T_block_1.png --object-description "red T block"
uv run python examples/generate_mask.py --image data/raise_cube_0__camera_base__t=0.rgb.png --object-description "blue cube"
```


**Example 2: Track a mask through a demonstration.** SAM 3 segments the prompt on **frame 0** (union of the top `--sam-kmax` masks whose score is above `--sam-score-threshold`). That union is propagated through later RGB frames with SAM 3 `mask_input` plus the previous mask's bbox. Writes per-frame `.npy` masks, dimmed overlays, and an mp4 under `<h5_dir>/<h5_stem>/`.

```bash
uv run python examples/generate_masks_across_trajectory.py \
    --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
    --camera cam_1 \
    --object-description "robot arm"
```


**Example 3: Generate a mesh for the object in `data/red_T_block_1.png` and visualize it with viser:**

```bash
uv run python examples/generate_mesh.py --images data/red_T_block_1.png --visualize
uv run python examples/generate_mesh.py --images data/raise_cube_0__camera_base__t=0.rgb.png --visualize
```


**Example 4: Estimate camera extrinsics and save results to a yaml file.** This script runs the CMA-ES optimization procedure to estimate the extrinsics of a specified camera given RGBD images, joint angles, and the urdf of the robot (urdf from [Jrl2](https://github.com/jstmn/Jrl2)).
```bash
uv run python examples/estimate_camera_extrinsics.py \
  --h5-path data/demonstrations/0802/0802_mustard/demonstration_0/merged_sensor_data.h5 \
  --robot-id xarm7__gripper \
  --camera cam_1 \
  --camera-model-id d435 \
  --depth-intrinsics-source rgb \
  --n-timesteps 2 \
  --output-path data/demonstrations/0802/extrinsics.yaml \
  --seed-from-gui \
  --visualize \
  --visualize-robot-masks
```

Exactly one seed mode is required:

- `--seed-automatically` evaluates the configured spherical grid before CMA-ES.
- `--seed-from-gui` starts Viser and waits for a manual seed. Click the 3D view, then use
  world-frame (robot-base) hotkeys: R/F X, T/G Y, Y/H Z, U/J roll about X, I/K pitch about Y,
  O/L yaw about Z (top row +, bottom row -). Press Enter or click **Select seed and start CMA-ES**.
  `--gui-translation-step-m` and `--gui-rotation-step-deg` control the increments.

`--robot-id` is a Jrl2 robot name. A bare name is the arm only. `{robot}__{eef}` is the same arm with that
end effector (`__` delimits robot vs EEF; `_` stays inside each token):

| `--robot-id` | End effector |
| --- | --- |
| `xarm7` | none (wrist flange only) |
| `xarm7__gripper` | UFACTORY parallel-jaw gripper |
| `xarm7__bio_gripper` | UFACTORY BIO gripper |
| `xarm7__vacuum_gripper` | UFACTORY vacuum gripper |

Demo `obs/qpos` is arm joints only. Extra EEF joints (e.g. `drive_joint` on `xarm7__gripper`)
are pinned at 0 (closed gripper).

Robot masks: SAM 3 runs on **frame 0** (union of the top `--sam-kmax` masks). That union is
propagated through the rest of the trajectory with SAM 3 `mask_input` plus the previous mask's bbox,
so contact with an object does not expand the robot mask.

**Example 5: Generate a mesh for the "mustard bottle" seen in the first frame of a
demonstration, then track that object through the demonstration:**
Pass `--visualize` to start a viser server with the mesh and a timestep slider over predicted poses.
Note: FoundationPose runs inside the Docker container (see Installation above), but the rest of the toolkit runs on the host in the `uv` venv. 
`r2st.pose_grpc` bridges the two: a server (`r2st.pose_grpc.server`) runs inside the container and exposes `FoundationPoseTracker`'s `register`/`track` over gRPC; a client (`r2st.pose_grpc.client.FoundationPoseClient`) is used from host-side code (e.g. `examples/track_object.py`) to call it.

```bash
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
`--depth-intrinsics-source rgb|depth` is required and says which of those K's the h5 depth channel uses (`rgb` = already color-aligned).

| `--camera-model-id` | Device |
| --- | --- |
| `d435` | Intel RealSense D435 |

## Pose tracking (gRPC)

Start the server inside the FoundationPose container (`bash run_container.sh` from `src/r2st/FoundationPose/docker`):

```bash
cd /real2sim-toolkit/src && python -m r2st.pose_grpc.server
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

* [SAM 3](https://github.com/jstmn/sam3) (fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3))
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

Error inside the container: `CUDA initialization: ... Error 804: forward compatibility was attempted on non supported HW`, or on the host: `The NVIDIA driver on your system is too old`. CUDA 12.8 userspace needs a driver that reports CUDA >= 12.8 (typically 570+). `nvidia-smi` showing CUDA 12.4 / driver 550 is too old; upgrading the driver is required — rebuilding the image will not help.

`gcloud storage cp` fails with `Reauthentication failed`: run `gcloud auth login`.

`docker build` fails with `docker-credential-pass: executable file not found in $PATH`: `credsStore` in `~/.docker/config.json` is `pass`, but the helper is not on `PATH`. Put `$HOME/.local/bin` on `PATH` (a literal `~/.local/bin` in PATH is not expanded) and retry.
