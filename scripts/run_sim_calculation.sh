#!/bin/bash

# Need to have instree_analysis repo cloned
SCRIPT_DIR="/home/jack/Code/Research/instree_analysis"
FOLDER1="${SCRIPT_DIR}/experiment_image/v1/v1_dog/D_backpack_dog_sub/v0"
FOLDER2="${SCRIPT_DIR}/experiment_image/v1/0312/dog/v1"

cd "$SCRIPT_DIR" || exit
echo "Folder 1: $FOLDER1"
echo "Folder 2: $FOLDER2"
python tools/score_calculation.py \
    --folder1 $FOLDER1 \
    --folder2 $FOLDER2 \
    --model_type clip \
    # --model_id_clip openai/clip-vit-large-patch14 \
    # --show_sim_detail \