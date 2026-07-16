#!/usr/bin/env bash
# Set up the CUDA training environment on a Huawei Cloud NVIDIA GPU instance
# (ECS Pi/PnT or ModelArts Notebook with A100/A800/V100). Run once per instance.
set -euo pipefail

ENV_NAME=${ENV_NAME:-deepjump}
PY=${PY:-3.11}
# Pick the CUDA wheel matching the instance driver (nvidia-smi). cu121 covers most A100/A800.
TORCH_CUDA=${TORCH_CUDA:-cu121}

echo ">> creating conda env $ENV_NAME (python $PY)"
conda create -y -n "$ENV_NAME" python="$PY"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">> installing PyTorch ($TORCH_CUDA) + deps"
pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch
pip install numpy h5py huggingface_hub tqdm pyyaml matplotlib tensorboard

echo ">> installing deepjump (editable)"
pip install -e .

echo ">> sanity"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count())
PY
echo ">> done. Activate with:  conda activate $ENV_NAME"
