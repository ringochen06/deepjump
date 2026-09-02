#!/usr/bin/env bash
set -euo pipefail

# Local convenience wrapper for the verified formal-500k checkpoint.
DEEPJUMP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FORMAL_SOURCE=${DEEPJUMP_FORMAL_SOURCE:-/Users/ringochen/hkucds/deepjump-post500k-unroll}
FORMAL_COMMIT=d469bfebc55a087725527721d1798b7f592fb5bb
CHECKPOINT=${DEEPJUMP_FORMAL_CKPT:-$DEEPJUMP_ROOT/artifacts/formal500k/20260726T164217Z/ckpt_500000.pt}
CHECKPOINT_SHA=d0e7ae08f1a9e4f3ae11fa73c45f4e6005e9eac66754070b5b92fcaab91348e6
INPUT_H5=${DEEPJUMP_FORMAL_INPUT:-/Users/ringochen/hkucds/data/mdcath/data/mdcath_dataset_1a92A00.h5}
STEPS=${1:-20}
MODE=${2:-mean}
START=${3:-native}
DEVICE=${DEEPJUMP_ANIMATION_DEVICE:-cpu}

if [[ ! $STEPS =~ ^[0-9]+$ ]] || (( STEPS < 1 )); then
  printf 'steps must be a positive integer\n' >&2
  exit 2
fi
if [[ $MODE != mean && $MODE != ode ]]; then
  printf 'mode must be mean or ode\n' >&2
  exit 2
fi
if [[ $START != native && $START != unfolded && $START != extended ]]; then
  printf 'start must be native, unfolded, or extended\n' >&2
  exit 2
fi
for required in "$CHECKPOINT" "$INPUT_H5" "$FORMAL_SOURCE/src/deepjump/model/deepjump.py"; do
  [[ -f $required ]] || { printf 'missing required file: %s\n' "$required" >&2; exit 2; }
done

actual_commit=$(git -C "$FORMAL_SOURCE" rev-parse HEAD)
[[ $actual_commit == "$FORMAL_COMMIT" ]] || {
  printf 'formal source commit mismatch: %s\n' "$actual_commit" >&2
  exit 2
}
git -C "$FORMAL_SOURCE" diff --quiet "$FORMAL_COMMIT" -- src/deepjump || {
  printf 'tracked formal source differs from commit %s\n' "$FORMAL_COMMIT" >&2
  exit 2
}
actual_sha=$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')
[[ $actual_sha == "$CHECKPOINT_SHA" ]] || {
  printf 'checkpoint SHA256 mismatch: %s\n' "$actual_sha" >&2
  exit 2
}

ode_steps=1
if [[ $MODE == ode ]]; then
  ode_steps=20
fi

temperature=320
replica=0
frame=0
initial_pdb=
default_output=$DEEPJUMP_ROOT/runs/visualization/formal500k_1a92A00/rollout.pdb
if [[ $START == unfolded ]]; then
  temperature=450
  replica=2
  frame=211
  default_output=$DEEPJUMP_ROOT/runs/visualization/formal500k_1a92A00_hot_unfolded/from_unfolded.pdb
elif [[ $START == extended ]]; then
  initial_pdb=$DEEPJUMP_ROOT/runs/visualization/formal500k_1a92A00_extended/initial_extended.pdb
  [[ -f $initial_pdb ]] || {
    printf 'missing synthetic extended PDB: %s\n' "$initial_pdb" >&2
    exit 2
  }
  default_output=$DEEPJUMP_ROOT/runs/visualization/formal500k_1a92A00_extended/from_extended.pdb
fi
OUTPUT_PDB=${DEEPJUMP_ANIMATION_OUT:-$default_output}

rollout_command=(
  python "$DEEPJUMP_ROOT/scripts/export_rollout_pdb.py"
  --ckpt "$CHECKPOINT"
  --input-h5 "$INPUT_H5"
  --temperature "$temperature"
  --replica "$replica"
  --frame "$frame"
  --steps "$STEPS"
  --delta-ns 1
  --mode "$MODE"
  --ode-steps "$ode_steps"
  --integrator euler
  --tau-max 1.0
  --drift-anchor state
  --seed 0
  --device "$DEVICE"
  --fps 5
  --out "$OUTPUT_PDB"
)
if [[ -n $initial_pdb ]]; then
  rollout_command+=(--initial-pdb "$initial_pdb")
fi
PYTHONPATH="$FORMAL_SOURCE/src" "${rollout_command[@]}"

printf '\nOpen the animation with:\n'
printf '/Applications/PyMOL.app/Contents/bin/pymol %q\n' "${OUTPUT_PDB%.pdb}.pml"
