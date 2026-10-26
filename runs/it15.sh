#!/bin/bash
# Iteration 15: leave-site-out inner models on pretrained features (ERM head), for cross-site conformal calibration.
cd /home/claude/sidl
python scripts/run_feat_lso_conformal.py --out runs/it15_lso --holdouts Actualmed Italy China Spain UK Australia Other Unknown --seeds 0 1 2 --method erm
