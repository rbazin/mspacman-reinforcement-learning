#!/bin/bash

for ((i = 1; i <= $1; i++ ))
do
    python3.9 play.py --bin_file "./roms/mspacman.bin" --weights "./weights.csv"
done