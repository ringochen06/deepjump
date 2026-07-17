#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/data/deepjump}
python=/data/venvs/deepjump/bin/python
ckpt="$repo/runs/v100_paperstyle_unroll3_1000/ckpt_1000.pt"
run_dir="$repo/runs/paperstyle_noise_scan_20260717"
sigmas=(0 0.025 0.05 0.1)

cd "$repo"
mkdir -p "$run_dir"
: >"$run_dir/status.tsv"

pids=()
labels=()
gpu=0
for sigma in "${sigmas[@]}"; do
  label=${sigma//./p}
  out="$run_dir/sigma_${label}"
  mkdir -p "$out"

  CUDA_VISIBLE_DEVICES=$gpu "$python" scripts/rollout_robustness_eval.py \
    --ckpt "$ckpt" --domains 5 --starts 5 --steps 20 --methods ode_1 \
    --noise-sigma "$sigma" --seed 20260716 \
    --output "$out/rollout20_ode1.json" >"$out/rollout20_ode1.log" 2>&1 &
  pids+=("$!"); labels+=("sigma_${label}_rollout20"); gpu=$((gpu + 1))

  CUDA_VISIBLE_DEVICES=$gpu "$python" scripts/transition_robustness_eval.py \
    --ckpt "$ckpt" --domains 10 --starts 50 --draws 16 --methods ode_1 \
    --noise-sigma "$sigma" --real-frames 500 --max-features 512 --lag 10 \
    --seed 20260717 --output "$out/transition_fair_ode1.json" \
    >"$out/transition_fair_ode1.log" 2>&1 &
  pids+=("$!"); labels+=("sigma_${label}_transition"); gpu=$((gpu + 1))
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then code=0; else code=$?; status=1; fi
  printf '%s\t%s\t%s\n' "${labels[$i]}" "${pids[$i]}" "$code" \
    >>"$run_dir/status.tsv"
done
exit "$status"
