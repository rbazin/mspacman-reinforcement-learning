#!/bin/bash

for i in [0..10]
do
   python learn.py --bin_file ./roms/mspacman.bin --nbr_episodes 50
done