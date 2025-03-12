#!bin/bash

GPU_ID=0
SEED=111
NODE="v0" # Format: {chr}{num}, e.g. v0, v1, v2, v3
PROMPT="object object"
IN="input_concepts/"
OUT="outputs/"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

train () {
    local NODE=$1
    python main_singleseed.py \
        --parent_data_dir "$PARENT" \
        --prompt "${PROMPT}" \
        --node "$NODE" \
        --test_name "$NODE" \
        --GPU_ID "${GPU_ID}" \
        --seed "$SEED" 
}

test () {
    local TOKENPATH=$1
    local NODE=$2
    python consistency_score.py \
        --path_to_new_tokens "$TOKENPATH" \
        --node "$NODE" \
        --seed "$SEED"
}

train_multi () {
    local NODE=$1
    python main_multiseed.py \
        --parent_data_dir $PARENT \
        --node $NODE \
        --test_name $NODE \
        --GPU_ID "${GPU_ID}" \
        --multiprocess 1
}

for PARENT in $(ls "$IN"); 
do
    train $NODE
    test "${OUT}/${PARENT}" $NODE
    # train_multi $NODE
done