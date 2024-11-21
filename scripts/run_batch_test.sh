#!/bin/bash

OUT="outputs/"
SEED="111"

run_script () {
    local TOKENPATH=$1
    local NODE=$2
    python consistency_score.py \
        --path_to_new_tokens "$TOKENPATH" \
        --node "$NODE" \
        --seed "$SEED"
}

for PARENT in "canada_bear" "cat_sculpture" "D_backpack_dog" "D_fancy_boot" "wooden_puppet"
do
    run_script "${OUT}/${PARENT}" "v0"
    run_script "${OUT}/${PARENT}" "c0"

    python utils/plot_two_score.py \
        --path1 "${OUT}/${PARENT}" \
        --node1 "v0" \
        --seed1 $SEED \
        --name1 "Origin(v0)" \
        --path2 "${OUT}/${PARENT}" \
        --node2 "c0" \
        --seed2 $SEED \
        --name2 "Contrastive(c0)"
done
