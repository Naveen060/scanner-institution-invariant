#!/bin/bash
# Iteration 22: repeat the semisynthetic experiments (iterations 3, 4, 7) with every source radiograph used at most once
# (--strict, 140 images per virtual site; the originals used 160 per site and reused 14-23% of the images across sites).
cd /home/claude/sidl
M="erm erm_p dann cdann cdann_p"
for sd in 0 1; do
  (
    python scripts/run_synth.py --out runs/it3_nr --spreads 0.0 0.25 0.4 --methods $M --seeds $sd --holdouts V0 V2 V4 --epochs 12 --per-site 140 --strict --set aug_p=0.0 > runs/it22_it3_s$sd.log 2>&1
    for lam in 0.02 0.1; do
      tag=$(echo $lam | tr -d .)
      python scripts/run_synth.py --out runs/it4_nr_l$tag --spreads 0.0 0.4 --methods dann cdann_p --seeds $sd --holdouts V0 V2 V4 --epochs 12 --per-site 140 --strict --set aug_p=0.0 adv_weight=$lam > runs/it22_it4_${tag}_s$sd.log 2>&1
    done
    python scripts/run_synth_lso.py --out runs/it7_nr --severities 0.5 1.0 1.5 --seeds $sd --per-site 140 --strict > runs/it22_it7_s$sd.log 2>&1
  ) &
done
wait
echo done > runs/it22.done
