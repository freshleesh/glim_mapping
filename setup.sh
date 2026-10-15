#!/usr/bin/env bash
# One-time setup on a new Ubuntu machine: Docker (+ NVIDIA runtime if a GPU is present),
# GLIM images, and the PLY-exporter image.  Run:  ./setup.sh   (asks for sudo)
set -e
REPO="$(dirname "$(readlink -f "$0")")"

if ! command -v docker >/dev/null; then
  echo "[docker] installing docker.io"
  sudo apt update && sudo apt install -y docker.io
  sudo systemctl enable --now docker
fi
if ! id -nG "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  echo "[docker] added $USER to docker group (new shells pick it up; scripts use 'sg docker' meanwhile)"
fi
D() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "docker $*"; fi; }

GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  GPU=1
  if ! command -v nvidia-ctk >/dev/null; then
    echo "[gpu] installing nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt update && sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
else
  echo "[gpu] no working nvidia-smi -> CPU only (set use_gpu: false in your map yaml)"
fi

echo "[python] host deps: numpy scipy matplotlib pyyaml"
python3 -c "import numpy, scipy, matplotlib, yaml" 2>/dev/null \
  || sudo apt install -y python3-numpy python3-scipy python3-matplotlib python3-yaml

echo "[images] pulling GLIM"
D pull koide3/glim_ros2:jazzy
[ $GPU = 1 ] && D pull koide3/glim_ros2:jazzy_cuda13.1
echo "[images] building glim_tools:jazzy (PLY exporter)"
D build -t glim_tools:jazzy "$REPO/tools/export_ply"
[ $GPU = 1 ] && D run --rm --gpus all koide3/glim_ros2:jazzy_cuda13.1 nvidia-smi -L
echo "DONE."
