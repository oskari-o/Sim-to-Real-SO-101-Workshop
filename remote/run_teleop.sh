#!/bin/bash
# Start the teleop-docker container on a headless remote GPU host.
# Same as the course's command, minus the X11 bits, plus all NVIDIA driver
# capabilities (needed for the video encoder used by WebRTC livestreaming).
set -e
REPO="${REPO:-$HOME/Sim-to-Real-SO-101-Workshop}"
cd "$REPO"
docker run --name teleop -it --privileged --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" -e NVIDIA_DRIVER_CAPABILITIES=all \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
   -v ./docker/env:/root/env \
   -v "$(pwd)/source:/workspace/Sim-to-Real-SO-101-Workshop/source" \
   -v "$(pwd)/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs" \
   -v "$(pwd)/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets" \
   -v "$(pwd)/docker/real/scripts:/workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts" \
   -v "$(pwd)/remote:/workspace/Sim-to-Real-SO-101-Workshop/remote" \
   teleop-docker:latest
