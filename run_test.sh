#!bin/bash

path="outputs/canada_bear"
node="v0"
seeds="111"

for step in $(seq 200 50 1000)
do
    python seed_selection.py \
        --path_to_new_tokens $path \
        --node $node \
        --step $step \
        --seeds $seeds
done
