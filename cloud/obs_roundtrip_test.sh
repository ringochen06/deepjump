#!/usr/bin/env bash
# Small OBS round-trip sanity check: upload two tiny files, download them back, verify byte
# equality. Run this BEFORE the 306 GB stage_to_obs.sh so a mis-configured bucket/AK-SK/region
# fails in seconds, not after a long transfer.
#
#   BUCKET=obs://my-mdcath bash cloud/obs_roundtrip_test.sh
#   CLEANUP=1 BUCKET=obs://my-mdcath bash cloud/obs_roundtrip_test.sh   # also remove the test objects
set -euo pipefail

BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
OBS_PREFIX=${OBS_PREFIX:-mdcath}
TESTKEY="$BUCKET/$OBS_PREFIX/_roundtrip_test"
CLEANUP=${CLEANUP:-0}

command -v obsutil >/dev/null || { echo "!! obsutil not found -- install & 'obsutil config' first"; exit 1; }

work=$(mktemp -d)
up="$work/up"; down="$work/down"
mkdir -p "$up/sub" "$down"
printf 'deepjump obs roundtrip a\n' > "$up/a.txt"
printf 'deepjump obs roundtrip b\n' > "$up/sub/b.txt"

echo ">> [1/3] upload  $up -> $TESTKEY"
obsutil sync "$up" "$TESTKEY"

echo ">> [2/3] download $TESTKEY -> $down"
obsutil sync "$TESTKEY" "$down"

echo ">> [3/3] verify"
n_up=$(find "$up" -type f | wc -l | tr -d ' ')
n_dn=$(find "$down" -type f | wc -l | tr -d ' ')
echo "   files: up=$n_up down=$n_dn"
if diff -r "$up" "$down"; then
  echo "   OK: round-trip byte-identical"
else
  echo "!! FAIL: downloaded content differs from uploaded"; rm -rf "$work"; exit 1
fi

if [ "$CLEANUP" = "1" ]; then
  echo ">> cleanup: removing $TESTKEY"
  obsutil rm "$TESTKEY" -r -f || echo "   (cleanup skipped/failed -- remove $TESTKEY manually if needed)"
fi
rm -rf "$work"
echo ">> round-trip test passed."
