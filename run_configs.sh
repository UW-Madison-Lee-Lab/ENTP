#!/bin/bash

trap "exit" INT

for FILE in $(ls configs/*.json); do
    # python generate_len_gen_addition_data.py "$FILE"
    python generate_addition_data.py "$FILE"
    python train_text_mlp.py "$FILE"

    # python train_data_gen.py "$FILE"

    # python train_len_gen_counting.py "$FILE"
done