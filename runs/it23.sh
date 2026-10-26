#!/bin/bash
# Iteration 23: post hoc sensitivity check. The reported site-profile augmentation and PAI runs used the default
# uncompensated sharpness statistic; repeat both with the noise-compensated statistic (compensate=1) on all eight held-out
# sites, same frozen settings otherwise. Not used for any decision.
cd /home/claude/sidl
ALL="Actualmed Italy China Spain UK Australia Other Unknown"
for sd in 0 1; do
  (
    python scripts/run_array_loso.py --out runs/it23_phys --methods phys_aug --seeds $sd --holdouts $ALL --epochs 15 --set aug_p=0.3 aug_strength=0.5 compensate=1 > runs/it23_phys_s$sd.log 2>&1
    python scripts/run_array_loso.py --out runs/it23_pai --methods pai --seeds $sd --holdouts $ALL --epochs 15 --set adv_weight=0.1 padv_weight=1.0 aug_p=0.0 compensate=1 > runs/it23_pai_s$sd.log 2>&1
  ) &
done
wait
echo done > runs/it23.done
