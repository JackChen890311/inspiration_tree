#!bin/bash

path1="outputs/canada_bear"
node1="v0"
seed1="111"
name1="origin"

path2="outputs/canada_bear"
node2="c0"
seed2="111"
name2="cl"


python utils/plot_two_score.py \
    --path1 $path1 \
    --node1 $node1 \
    --seed1 $seed1 \
    --name1 $name1 \
    --path2 $path2 \
    --node2 $node2 \
    --seed2 $seed2 \
    --name2 $name2 \