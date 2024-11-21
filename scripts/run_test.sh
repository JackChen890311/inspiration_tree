#!bin/bash

path="outputs/D_backpack_dog"
node="f0"
seed="111"

python consistency_score.py \
    --path_to_new_tokens $path \
    --node $node \
    --seed $seed \