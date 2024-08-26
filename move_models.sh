#!/bin/bash

trap "exit" INT

for FILE in models/*; do
    if [ -f "$FILE" ]; then
        if [[ "$FILE" == models/counting_extra_small*.pt ]]; then
            FILE="${FILE##models/}"
            cp "models/$FILE" "models/upload/$FILE"
            echo "copied $FILE"
        fi
    fi
done