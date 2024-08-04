#!/bin/bash

cd ..

for i in {0..9} 
do 
    config_file="configs/config$i.json"
    python generate_addition_data.py "$config_file"
    python train.py "$config_file"
done