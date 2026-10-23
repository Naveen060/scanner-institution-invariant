#!/bin/bash
cd /home/claude/sidl
DEV="Actualmed Italy China Spain"
M="sidl_full sidl_phys sidl_cc"
python scripts/run_array_loso.py --out runs/it8_mild --methods $M --seeds 0 --holdouts $DEV --epochs 15 --set aug_p=0.3 aug_strength=0.5 adv_weight=0.1 > runs/it8_a.log 2>&1 &
python scripts/run_array_loso.py --out runs/it8_mild --methods $M --seeds 1 --holdouts $DEV --epochs 15 --set aug_p=0.3 aug_strength=0.5 adv_weight=0.1 > runs/it8_b.log 2>&1 &
wait
