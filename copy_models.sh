#!/bin/bash

trap "exit" INT

for FILE in models/*; do
    if [ -f "$FILE" ]; then
        if [[ "$FILE" == models/*_small_deep_openwebtext.pt ]]; then
            FILE="${FILE##models/}"
            cp "models/$FILE" "models/seperate/$FILE"
            echo "copied $FILE"
        fi
    fi
done