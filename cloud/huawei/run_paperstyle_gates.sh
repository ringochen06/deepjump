#!/usr/bin/env bash
set -uo pipefail

repo=${1:-/data/deepjump}
train_pattern='^/data/venvs/deepjump/bin/python /data/venvs/deepjump/bin/torchrun .*v100_paperstyle_d1_2000'
run_dir="$repo/runs/paperstyle_d1_2000_20260717"
ckpt_dir="$repo/runs/v100_paperstyle_d1_2000"
python=/data/venvs/deepjump/bin/python

while pgrep -f "$train_pattern" >/dev/null; do
  sleep 30
done

for step in 500 1000 2000; do
  if [[ ! -s "$ckpt_dir/ckpt_${step}.pt" ]]; then
    printf 'missing checkpoint: %s\n' "$ckpt_dir/ckpt_${step}.pt" >&2
    exit 2
  fi
  mkdir -p "$run_dir/gates/ckpt${step}"
  "$python" "$repo/scripts/probe_vector_qk_gates.py" \
    --ckpt "$ckpt_dir/ckpt_${step}.pt" \
    --output "$run_dir/gates/ckpt${step}/vector_qk_gates.json" \
    >"$run_dir/gates/ckpt${step}/vector_qk_gates.log" 2>&1
done

pids=()
labels=()
task_index=0
for step in 500 1000 2000; do
  out="$run_dir/gates/ckpt${step}"

  gpu=$((task_index % 8)); task_index=$((task_index + 1))
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/robustness_eval.py" \
    --ckpt "$ckpt_dir/ckpt_${step}.pt" --samples-per-trajectory 1 \
    --ode-steps 1 --sample-seed 20260716 \
    --output "$out/fixed_ode1.json" >"$out/fixed_ode1.log" 2>&1 &
  pids+=("$!"); labels+=("ckpt${step}_fixed")

  gpu=$((task_index % 8)); task_index=$((task_index + 1))
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/rollout_robustness_eval.py" \
    --ckpt "$ckpt_dir/ckpt_${step}.pt" --domains 5 --starts 5 \
    --steps 20 --methods ode_1 --seed 20260716 \
    --output "$out/rollout20_ode1.json" >"$out/rollout20_ode1.log" 2>&1 &
  pids+=("$!"); labels+=("ckpt${step}_rollout")

  gpu=$((task_index % 8)); task_index=$((task_index + 1))
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/transition_robustness_eval.py" \
    --ckpt "$ckpt_dir/ckpt_${step}.pt" --domains 10 --starts 50 --draws 4 \
    --methods ode_1 --real-frames 500 --max-features 512 --lag 10 --seed 20260717 \
    --output "$out/transition_fair_ode1.json" >"$out/transition_fair_ode1.log" 2>&1 &
  pids+=("$!"); labels+=("ckpt${step}_transition")
done

status=0
: >"$run_dir/gates/status.tsv"
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    code=0
  else
    code=$?
    status=1
  fi
  printf '%s\t%s\t%s\n' "${labels[$i]}" "${pids[$i]}" "$code" \
    >>"$run_dir/gates/status.tsv"
done
exit "$status"
