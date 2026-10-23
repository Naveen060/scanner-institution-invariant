#!/bin/bash
cd /home/claude/sidl
M="erm erm_p dann cdann cdann_p"
for sd in 0 1; do
  python scripts/run_synth.py --out runs/it3 --spreads 0.0 0.25 0.4 --methods $M --seeds $sd --holdouts V0 V2 V4 --epochs 12 --set aug_p=0.0 > runs/it3_s$sd.log 2>&1 &
done
wait
