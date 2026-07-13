#!/usr/bin/env bash
# Launch DDP training on one node with all visible GPUs.
#
#   CONFIG=configs/paper_h128_d1.yaml ./cloud/run_ddp.sh
#   CONFIG=configs/paper_h128_d1.yaml RESUME=runs/paper_h128_d1/last.ckpt ./cloud/run_ddp.sh
#
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT and run on each node.
set -euo pipefail

CONFIG=${CONFIG:?set CONFIG=configs/paper_h128_d1.yaml}
NGPU=${NGPU:-$(python -c 'import torch;print(torch.cuda.device_count())')}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RESUME=${RESUME:-}

echo ">> DDP: $NGPU GPUs x $NNODES nodes  config=$CONFIG  resume=${RESUME:-none}"
ARGS=(--config "$CONFIG")
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")

torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$NGPU" \
  scripts/train_ddp.py "${ARGS[@]}"
