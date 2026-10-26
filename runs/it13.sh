#!/bin/bash
# Iteration 13: three more seeds of the (unchanged) from-scratch ERM on the development folds, so both families have 5 seeds.
cd /home/claude/sidl
python scripts/run_array_loso.py --out runs/it1 --methods erm --holdouts Actualmed Italy China Spain --seeds 2 3 4 --epochs 15
