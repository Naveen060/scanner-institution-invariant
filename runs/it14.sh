#!/bin/bash
# Iteration 14: frozen-config confirmation run for the pretrained family, plus 3 extra seeds of the frozen CNN ERM.
cd /home/claude/sidl
CONF="UK Australia Other Unknown"
M="erm erm_p groupdro dann cdann cdann_p coral cc_full sitecenter"
(python scripts/run_feat_loso.py --out runs/it14_feat --methods $M --seeds 0 1 2 3 4 --holdouts $CONF --set adv_weight=0.5) > runs/it14_feat.log 2>&1 &
(python scripts/run_array_loso.py --out runs/conf_erm --methods erm --holdouts $CONF --seeds 2 3 4 --epochs 15) > runs/it14_erm.log 2>&1 &
wait
