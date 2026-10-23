#!/bin/bash
# Iteration 1 baselines on the development folds (the augmentation-free baselines used in the final tables).
# NOTE: the original iteration-1 run also contained augmentation variants implemented with whole-batch augmentation;
# that implementation was superseded in iteration 2 (per-sample probability/strength) and is not reproduced here.
cd /home/claude/sidl
python scripts/run_array_loso.py --out runs/it1 --methods erm groupdro dann --holdouts Actualmed Italy China Spain --seeds 0 1 --epochs 15
