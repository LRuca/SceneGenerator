#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tione/notebook/home/keithanwang/vlm-vla/gafu
source "$ROOT/sceneblocker/baseline/env.sh"
SAM_PREFIX="$ROOT/conda/envs/sceneblocker-sam3"
SAM_SOURCE="$SB_BASELINE_ROOT/dependencies/sam3-eef6839b"
if [[ ! -x "$SAM_PREFIX/bin/python" ]]; then
 "$ROOT/conda/miniforge/bin/conda" create -y -p "$SAM_PREFIX" --override-channels -c conda-forge python=3.12 pip 'setuptools<81'
fi
"$SAM_PREFIX/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$SAM_PREFIX/bin/python" -m pip install 'numpy>=1.26,<2' scipy pillow opencv-python imageio imageio-ffmpeg matplotlib einops decord pycocotools psutil pandas ninja triton 'setuptools<81'
"$SAM_PREFIX/bin/python" -m pip install -e "$SAM_SOURCE"
"$SAM_PREFIX/bin/python" -m pip freeze > "$SB_BASELINE_ROOT/reports/sam3-pip-freeze.txt"
