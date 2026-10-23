#!/bin/bash
cd /home/claude/sidl
DEV="Actualmed Italy China Spain"
run() { # name method p strength
  python scripts/run_array_loso.py --out runs/it2_$1 --methods $2 --seeds 0 1 --holdouts $DEV --epochs 15 --set aug_p=$3 aug_strength=$4 >> runs/it2_$1.log 2>&1
}
( run gen_p03_s05 harmonize 0.3 0.5; run gen_p05_s10 harmonize 0.5 1.0 ) &
( run phys_p03_s05 phys_aug 0.3 0.5; run phys_p05_s10 phys_aug 0.5 1.0 ) &
wait
