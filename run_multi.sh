#!bin/bash

GPU_ID=0
NODE="v0"
PARENT="canada_bear"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python main_multiseed.py \
    --parent_data_dir $PARENT \
    --node $NODE \
    --test_name $NODE \
    --GPU_ID "${GPU_ID}" \
    --multiprocess 2
