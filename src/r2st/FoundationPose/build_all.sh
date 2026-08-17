#!/usr/bin/env bash
# Build native extensions inside the Docker container's `my` conda env.
# kaolin is installed from a prebuilt wheel in the Dockerfile (see docker/dockerfile),
# so there is no /kaolin source build here.
set -euo pipefail

DIR=$(pwd)

cd "$DIR/mycpp" && rm -rf build && mkdir -p build && cd build && \
    cmake .. -DPYTHON_EXECUTABLE="$(which python)" && make -j"$(nproc)"

# Optional: BundleSDF CUDA ops (required for model-free NeRF path only)
# cd "$DIR/bundlesdf/mycuda" && rm -rf build *egg* && pip install --no-build-isolation -e .

cd "${DIR}"
