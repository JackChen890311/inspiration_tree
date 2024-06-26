#!bin/bash

GPU_ID=0
NODE="a0"
PROMPT="object object"
PARENT="green_dall"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python main_singleseed.py \
    --parent_data_dir $PARENT \
    --prompt "${PROMPT}" \
    --node $NODE \
    --test_name $NODE \
    --GPU_ID "${GPU_ID}" \