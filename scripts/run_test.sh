#!bin/bash

path="outputs/white_cat_square"
node="v0"
seed="111"

# Old method
# for step in $(seq 50 50 1000)
# do
#     python seed_selection.py \
#         --path_to_new_tokens $path \
#         --node $node \
#         --step $step \
#         --seeds $seed \
# done

python consistency_score.py \
    --path_to_new_tokens $path \
    --node $node \
    --seed $seed \