#!bin/bash

GPU_ID=0
SEED=111
PROMPT="object object"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"


run_script () {
    local NODE=$1
    python main_singleseed.py \
        --parent_data_dir "$PARENT" \
        --prompt "${PROMPT}" \
        --node "$NODE" \
        --test_name "$NODE" \
        --GPU_ID "${GPU_ID}" \
        --seed "$SEED" \
        $2
}

for PARENT in "canada_bear" "cat_sculpture" "D_backpack_dog" "D_fancy_boot" "wooden_puppet"
do
    run_script "v0"
    run_script "c0" "--contrastive"
done