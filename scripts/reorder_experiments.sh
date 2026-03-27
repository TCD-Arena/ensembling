#!/bin/bash

# Define source and destination paths
SOURCE_DIR="/home/datasets4/stein/robust_exp/release_ensemble_preds/eval/WCG/"
DEST_DIR="/home/datasets4/stein/robust_exp/release_ensemble_preds/eval_corrected/WCG/"

# 1. Create the destination directory if it doesn't exist
echo "Creating destination directory..."
mkdir -p "$DEST_DIR"

# 2. Run rsync
# -a : Archive mode (recurses into directories, preserves permissions/dates)
# -v : Verbose (shows you what is being copied)
# --exclude : Skips any file or folder matching the pattern
echo "Starting copy process..."

rsync -av \
    --exclude 'svarrfci' \
    --exclude 'fpcmci' \
    "$SOURCE_DIR" "$DEST_DIR"

echo "Done! Files copied to $DEST_DIR"