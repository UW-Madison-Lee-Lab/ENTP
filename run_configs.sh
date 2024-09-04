#!/bin/bash

trap "exit" INT

for FILE in $(ls configs/*.json); do
    # python generate_addition_data.py "$FILE"
    # python train.py "$FILE"
    # python evaluate.py "$FILE"

    # python train_data_gen.py "$FILE"

    python train_memory_bound.py "$FILE"
done