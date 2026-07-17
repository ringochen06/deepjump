#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/data/deepjump}
python=/data/venvs/deepjump/bin/python
torchrun=/data/venvs/deepjump/bin/torchrun
config=configs/v100_tensorcloud_unroll3_adapt1000.yaml
source_ckpt=runs/v100_paperstyle_unroll3_1000/ckpt_1000.pt
ckpt_dir="$repo/runs/v100_tensorcloud_unroll3_adapt1000"
run_dir="$repo/runs/tensorcloud_unroll3_adapt_20260717"

cd "$repo"
[[ -s "$source_ckpt" ]] || {
  printf 'missing source checkpoint: %s\n' "$source_ckpt" >&2
  exit 2
}
[[ ! -e "$ckpt_dir" ]] || {
  printf 'refusing to overwrite output directory: %s\n' "$ckpt_dir" >&2
  exit 2
}
mkdir -p "$run_dir/gates"

timeout --signal=TERM --kill-after=2m 40m "$torchrun" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$config" --warm-start "$source_ckpt" \
  >"$run_dir/train.log" 2>&1

for step in 250 500 1000; do
  ckpt="$ckpt_dir/ckpt_${step}.pt"
  out="$run_dir/gates/ckpt${step}"
  [[ -s "$ckpt" ]] || {
    printf 'missing checkpoint: %s\n' "$ckpt" >&2
    exit 2
  }
  mkdir -p "$out"

  CUDA_VISIBLE_DEVICES=0 "$python" scripts/robustness_eval.py \
    --ckpt "$ckpt" --samples-per-trajectory 1 --ode-steps 1 \
    --sample-seed 20260716 --output "$out/fixed_ode1.json" \
    >"$out/fixed_ode1.log" 2>&1 &
  fixed_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$python" scripts/rollout_robustness_eval.py \
    --ckpt "$ckpt" --domains 5 --starts 5 --steps 20 --methods ode_1 \
    --seed 20260716 --output "$out/rollout20_ode1.json" \
    >"$out/rollout20_ode1.log" 2>&1 &
  rollout_pid=$!
  CUDA_VISIBLE_DEVICES=2 "$python" scripts/transition_robustness_eval.py \
    --ckpt "$ckpt" --domains 10 --starts 50 --draws 4 --methods ode_1 \
    --real-frames 500 --max-features 512 --lag 10 --seed 20260717 \
    --output "$out/transition_fair_ode1.json" \
    >"$out/transition_fair_ode1.log" 2>&1 &
  transition_pid=$!

  status=0
  : >"$out/status.tsv"
  for item in "fixed:$fixed_pid" "rollout:$rollout_pid" "transition:$transition_pid"; do
    label=${item%%:*}
    pid=${item##*:}
    if wait "$pid"; then code=0; else code=$?; status=1; fi
    printf '%s\t%s\t%s\n' "$label" "$pid" "$code" >>"$out/status.tsv"
  done
  (( status == 0 )) || exit 1
done
