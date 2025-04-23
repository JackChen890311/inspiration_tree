#!/bin/bash

# Need to have instree_analysis repo cloned
DATASET_NAME="v2"
EXP_NAME="20250423_0410_on_v2"
EXP_SEED="111"

BASE_DIR="/home/jack/Code/Research/instree_analysis/experiment_image"
SCRIPT_DIR="/home/jack/Code/Research/instree_analysis"
EXP_IMG_DIR="${BASE_DIR}/${DATASET_NAME}/${EXP_NAME}"
ORIGIN_IMG_DIR="${BASE_DIR}/${DATASET_NAME}/${DATASET_NAME}_masked"
OUTPUT_DIR="${BASE_DIR}/scores/${DATASET_NAME}"

# Generate images
cd "$SCRIPT_DIR" || exit
python utils/image_generator.py \
    --dataset_name "$DATASET_NAME" \
    --exp_name "$EXP_NAME" \
    --exp_seed "$EXP_SEED" \
    # --multiseed \

# Evaluate with different models
for MODEL in clip dino
do
    python tools/evaluate_experiment.py \
        --exp_img_dir "$EXP_IMG_DIR" \
        --origin_img_dir "$ORIGIN_IMG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_type "$MODEL"
done
