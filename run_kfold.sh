#!/bin/bash

set -e

# fold_id=0 busy for test fold
for FOLD_ID in 1 2 3 4 5 6 7 8 9; do
  for SEED_ID in 0 1 2 3 4; do
    python main.py "$@" --fold_id $FOLD_ID --random_seed_id $SEED_ID
  done
done