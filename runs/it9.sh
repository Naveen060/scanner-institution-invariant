#!/bin/bash
cd /home/claude/sidl
CONF="UK Australia Other Unknown"
R="python scripts/run_array_loso.py --holdouts $CONF --seeds 0 1 --epochs 15"
(
 $R --out runs/conf_erm      --methods erm
 $R --out runs/conf_groupdro --methods groupdro
 $R --out runs/conf_dann     --methods dann
 $R --out runs/conf_gen      --methods harmonize --set aug_p=0.3 aug_strength=0.5
 $R --out runs/conf_phys     --methods phys_aug  --set aug_p=0.3 aug_strength=0.5
 $R --out runs/conf_pai      --methods pai       --set adv_weight=0.1 padv_weight=1.0 aug_p=0.0
 $R --out runs/conf_sidl_cc  --methods sidl_cc   --set aug_p=0.3 aug_strength=0.5 adv_weight=0.1
 python scripts/run_lso_conformal.py --out runs/conf_lso --holdouts UK --seeds 0 1 --epochs 15
) > runs/it9_q1.log 2>&1 &
(
 $R --out runs/conf_sidl_full --methods sidl_full --set aug_p=0.3 aug_strength=0.5 adv_weight=0.1
 $R --out runs/conf_sidl_phys --methods sidl_phys --set aug_p=0.3 aug_strength=0.5 adv_weight=0.1
 python scripts/run_lso_conformal.py --out runs/conf_lso --holdouts Australia Other Unknown --seeds 0 1 --epochs 15
) > runs/it9_q2.log 2>&1 &
wait
