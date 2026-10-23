#!/bin/bash
cd /home/claude/sidl
for lam in 0.02 0.1; do
  tag=$(echo $lam | tr -d .)
  python scripts/run_synth.py --out runs/it4_l$tag --spreads 0.0 0.4 --methods dann cdann_p --seeds 0 1 --holdouts V0 V2 V4 --epochs 12 --set aug_p=0.0 adv_weight=$lam > runs/it4_l$tag.log 2>&1 &
done
wait
