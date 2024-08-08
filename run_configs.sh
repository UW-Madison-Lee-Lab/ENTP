#!/bin/bash

trap "exit" INT

for FILE in configs/*; do
    if [ -f "$FILE" ]; then
        if [[ "$FILE" == *.json ]]; then
            python generate_addition_data.py "$FILE"
            echo "$FILE"
            wc -l data/addition/train_plain_addition.txt
            # python train.py "$FILE"
            # python evaluate.py "$FILE"
        fi
    fi
done