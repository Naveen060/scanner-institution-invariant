#!/bin/bash
cd /home/claude/sidl
DEV="Actualmed Italy China Spain"
for lam in 0.1 0.5; do
  tag=$(echo $lam | tr -d .)
  python scripts/run_array_loso.py --out runs/it5_l$tag --methods pai pai_u --seeds 0 1 --holdouts $DEV --epochs 15 --set adv_weight=$lam padv_weight=1.0 aug_p=0.0 > runs/it5_l$tag.log 2>&1 &
done
wait
