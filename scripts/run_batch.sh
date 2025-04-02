#!bin/bash

GPU_ID=0
SEED=111
NODE="v0" # Format: {chr}{num}, e.g. v0, v1, v2, v3
PROMPT="object object"
IN="input_concepts/"
OUT="outputs/"
EXP_FILE_NAME="${OUT}/exp.txt"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

train () {
    local NODE=$1
    echo "Training node $NODE"
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
    local SEED=$3
    echo "Testing node $NODE"
    python consistency_score.py \
        --path_to_new_tokens "$TOKENPATH" \
        --node "$NODE" \
        --seed "$SEED"
}

train_multi () {
    local NODE=$1
    echo "Training node with multiseeds $NODE"
    python main_multiseed.py \
        --parent_data_dir $PARENT \
        --node $NODE \
        --test_name $NODE \
        --GPU_ID "${GPU_ID}" \
        --multiprocess 1
}

collect_score(){
    python collect_scores.py \
        --output_path "${OUT}" \
        --node_name "${NODE}" \
        --exp_file_name "${EXP_FILE_NAME}"
}

start_exp(){
    # Remove all old outputs
    echo "Removing all old outputs"
    rm -rf "${OUT}"
    mkdir -p "${OUT}"

    # Ensure the output directory exists
    mkdir -p "$(dirname "$EXP_FILE_NAME")"
    read -p "Enter details of the experiment: " DETAILS
    echo "$DETAILS" > "$EXP_FILE_NAME"
    echo "Experiment details saved to $EXP_FILE_NAME"
}


start_exp

for PARENT in $(ls "$IN"); 
do
    echo "Parent: $PARENT"

    # for single seed experiments
    # train $NODE
    # test "${OUT}/${PARENT}" $NODE $SEED

    # for multi seed experiments
    train_multi $NODE
    for SEED in 0 111 1000 1234;
    do
        test "${OUT}/${PARENT}" $NODE $SEED
    done
done

collect_score