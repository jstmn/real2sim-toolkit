#!/usr/bin/env bash
set -euo pipefail

# Always resolve paths from this script's location so it works regardless of cwd.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# docker/ -> FoundationPose/ -> r2st/ -> src/ -> repo root
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CONTAINER_REPO=/real2sim-toolkit

docker rm -f foundationpose

docker run \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it \
  --network=host \
  --name foundationpose \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$REPO_ROOT:$CONTAINER_REPO" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  --ipc=host \
  foundationpose:latest \
  bash -c "cd $CONTAINER_REPO && bash"
