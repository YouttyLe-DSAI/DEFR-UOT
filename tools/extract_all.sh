#!/bin/bash
# Produce the 20 dumps the cross-corpus OT analysis needs.
#
# Both corpora must already be set up in this session: run the symlink build and
# the annotation retarget for MAFW *and* DFEW before starting, because each
# checkpoint is run over both test sets.
#
#   ./tools/extract_all.sh /kaggle/input/models/.../checkpoint dumps
set -euo pipefail

CKPT_ROOT="${1:?usage: extract_all.sh <checkpoint-root> [out-dir] [method]}"
OUT_DIR="${2:-dumps}"
METHOD="${3:-av}"

for fold in 1 2 3 4 5; do
  for ckpt in DFEW MAFW; do
    for test in DFEW MAFW; do
      name="$(echo "$ckpt" | tr 'A-Z' 'a-z')_$([ "$test" = DFEW ] && echo dfew || echo mafw11)_fold${fold}_${METHOD}.npz"
      if [ -f "$OUT_DIR/$name" ]; then
        echo "skip  $name (already there)"
        continue
      fi
      echo "=== ckpt=$ckpt  test=$test  fold=$fold ==="
      python tools/extract_features.py \
        --ckpt-dataset "$ckpt" --test-dataset "$test" --fold "$fold" \
        --checkpoint "$CKPT_ROOT/${ckpt}_224/fold${fold}_224.pth" \
        --out-dir "$OUT_DIR" --method "$METHOD"
    done
  done
done

echo
echo "done -- $(ls -1 "$OUT_DIR"/*.npz 2>/dev/null | wc -l) dumps in $OUT_DIR"
du -sh "$OUT_DIR"
