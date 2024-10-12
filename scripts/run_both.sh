#!bin/bash

GPU_ID=0
SEED=111
PROMPT="object object"
PARENT="cat_sculpture"
path="outputs/${PARENT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# with contrastive
NODE="c0"
python main_singleseed.py \
    --parent_data_dir $PARENT \
    --prompt "${PROMPT}" \
    --node $NODE \
    --test_name $NODE \
    --GPU_ID "${GPU_ID}" \
    --seed $SEED \
    --contrastive \

python consistency_score.py \
    --path_to_new_tokens $path \
    --node $NODE \
    --seed $SEED \

# without contrastive
NODE="v0"
python main_singleseed.py \
    --parent_data_dir $PARENT \
    --prompt "${PROMPT}" \
    --node $NODE \
    --test_name $NODE \
    --GPU_ID "${GPU_ID}" \
    --seed $SEED \

python consistency_score.py \
    --path_to_new_tokens $path \
    --node $NODE \
    --seed $SEED \