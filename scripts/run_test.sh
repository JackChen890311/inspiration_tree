#!bin/bash

OUT="outputs/"
PARENT="canada_bear"
node="v0"
seed="111"

python consistency_score.py \
    --path_to_new_tokens "${OUT}/${PARENT}" \
    --node $node \
    --seed $seed \