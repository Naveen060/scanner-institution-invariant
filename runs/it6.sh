#!/bin/bash
cd /home/claude/sidl
python scripts/run_lso_conformal.py --out runs/it6 --holdouts Actualmed Italy --seeds 0 1 --epochs 15 > runs/it6_a.log 2>&1 &
python scripts/run_lso_conformal.py --out runs/it6 --holdouts China Spain --seeds 0 1 --epochs 15 > runs/it6_b.log 2>&1 &
wait
