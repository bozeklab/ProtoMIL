#!/bin/bash

set -e

START_FOLD_ID=${1:-0}
END_FOLD_ID=${2:-9}

echo 10-fold crosvalidation, running folds from "$START_FOLD_ID" to "$END_FOLD_ID".

for FOLD_ID in $(seq "$START_FOLD_ID" "$END_FOLD_ID"); do
  for SEED_ID in 0 1 2 3 4; do
    echo CROSSVAL: fold_id: "$FOLD_ID", seed_id: $SEED_ID
    python main.py "$@" --fold_id "$FOLD_ID" --random_seed_id $SEED_ID
  done
done
