#!/bin/bash

trap "exit" INT

for FILE in $(ls configs/*.json); do
    # python generate_len_gen_addition_data.py "$FILE"
    # python train_text.py "$FILE"
    # python evaluate.py "$FILE"

    python train_data_gen.py "$FILE"

    # python train_len_gen_counting.py "$FILE"
done