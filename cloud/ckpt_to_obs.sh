#!/usr/bin/env bash
# Periodically archive checkpoints to OBS so a reclaimed/preempted GPU instance loses no
# progress. Run in the background alongside training; --resume from the synced ckpt after a
# restart. GPU-hours are the expensive resource -- OBS storage is cheap insurance.
#
#   RUN_DIR=runs/v100_paper_d1 BUCKET=obs://my-mdcath bash cloud/ckpt_to_obs.sh &
#
# After a restart, pull the latest back:
#   obsutil sync obs://my-mdcath/ckpts/v100_paper_d1 runs/v100_paper_d1
#
# The trainer writes checkpoints atomically (save_ckpt: torch.save to .tmp then os.replace),
# and the numbered ckpt_<step>.pt files are immutable once written -- so a sync never reads a
# half-written file. As defense-in-depth we still wait for last.ckpt's size to settle before
# syncing.
set -euo pipefail

RUN_DIR=${RUN_DIR:?set RUN_DIR=runs/<name>}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
EVERY=${EVERY:-600}                     # seconds between uploads
NAME=$(basename "$RUN_DIR")
DST="$BUCKET/ckpts/$NAME"

command -v obsutil >/dev/null || { echo "!! obsutil not found"; exit 1; }
echo ">> archiving $RUN_DIR -> $DST every ${EVERY}s (Ctrl-C to stop)"

fsize() { stat -c%s "$1" 2>/dev/null || echo -1; }

while true; do
  if [ -d "$RUN_DIR" ]; then
    # Wait for last.ckpt to stop growing (guard even against a non-atomic writer).
    last="$RUN_DIR/last.ckpt"
    if [ -f "$last" ]; then
      s1=$(fsize "$last"); sleep 3; s2=$(fsize "$last")
      [ "$s1" != "$s2" ] && { echo "   [$(date +%H:%M:%S)] last.ckpt still changing, will catch next round"; }
    fi
    # Incremental sync of the whole run dir (immutable ckpt_<step>.pt + atomic last.ckpt + history.json).
    obsutil sync "$RUN_DIR" "$DST" && echo "   [$(date +%H:%M:%S)] synced $NAME -> OBS"
  fi
  sleep "$EVERY"
done
