#!bin/bash

GPU_ID=0
SEED=0
NODE="v0" # Format: {chr}{num}, e.g. v0, v1, v2, v3
PROMPT="object object"
IN="input_concepts/"
OUT="outputs/"
EXP_FILE_NAME="${OUT}/exp.txt"
MODE="single" # single seed or multi seed
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
        --seed "$SEED" \
        --apply_otsu
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

start_exp(){
    read -p "Do you want to remove all old outputs? (y/n): " REMOVE
    if [[ "$REMOVE" == "y" ]]; then
        echo "Removing all old outputs"
        rm -rf "${OUT}"
        mkdir -p "${OUT}"

        mkdir -p "$(dirname "$EXP_FILE_NAME")"
        read -p "Enter details of the experiment: " DETAILS
        echo "$DETAILS" > "$EXP_FILE_NAME"
        echo "Experiment details saved to $EXP_FILE_NAME"
        echo "Starting new experiments..."
    else
        echo "Keeping old outputs and continuing"
        cat $EXP_FILE_NAME
        echo "."
        echo "Continuing existing experiments..."
    fi
}


start_exp

for PARENT in $(ls "$IN"); 
do
    echo "Parent: $PARENT"

    # if exist "$OUT/$PARENT" then continue
    if [ -d "${OUT}/${PARENT}" ]; then
        echo "Output directory for $PARENT already exists. Skipping..."
        continue
    fi

    # if mode single train single else multi train multi fi
    if [ "$MODE" == "single" ]; then
        echo "Single seed mode"
        train $NODE
    elif [ "$MODE" == "multi" ]; then
        echo "Multi seed mode"
        train_multi $NODE
    else
        echo "Invalid mode. Please set MODE to 'single' or 'multi'."
        exit 1
    fi
done
