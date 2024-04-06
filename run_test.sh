#!bin/bash

path="outputs/green_dall"
node="v0"
seeds="111"

for step in $(seq 50 100 1000)
do
    python seed_selection.py \
        --path_to_new_tokens $path \
        --node $node \
        --step $step \
        --seeds $seeds
done
