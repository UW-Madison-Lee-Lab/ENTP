#!/bin/bash

trap "exit" INT

for FILE in models/*; do
    if [ -f "$FILE" ]; then
        if [[ "$FILE" == models/reversed_addition_len_gen*.pt ]]; then
            FILE="${FILE##models/}"
            cp "models/$FILE" "models/seperate/$FILE"
            echo "copied $FILE"
        fi
    fi
done