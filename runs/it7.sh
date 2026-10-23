#!/bin/bash
cd /home/claude/sidl
python scripts/run_synth_lso.py --out runs/it7 --severities 0.5 1.0 1.5 --seeds 0 > runs/it7_a.log 2>&1 &
python scripts/run_synth_lso.py --out runs/it7 --severities 0.5 1.0 1.5 --seeds 1 > runs/it7_b.log 2>&1 &
wait
