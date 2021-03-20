#!/bin/bash

set -e

for FOLD_ID in 4 8 9; do
  for SEED_ID in 0 1 2 3 4; do
    python main.py "$@" --fold_id $FOLD_ID --random_seed_id $SEED_ID
  done
done

