#!/bin/bash

for FILE in configs/*; do
    if [ -f "$FILE" ]; then
        if [[ "$FILE" == *.json ]]; then
            python generate_configs.py "$FILE"
            python train.py "$FILE"
            python evaluate.py "$FILE"
        fi
    fi
done