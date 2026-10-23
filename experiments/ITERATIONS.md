# Iteration log

Each iteration: hypothesis -> change -> result -> decision. Results are copied from `runs/` as produced; nothing is edited by hand.
Compute: 2 CPU cores, no GPU, models trained from scratch (no pretrained weights reachable), images 64x64.
Data: open multi-source chest X-ray benchmark (924 images, 10 sites; COVID-19 vs other), built by `scripts/build_covid_cxr.py`.

## Protocol commitments (written before looking at results beyond iteration 1 interim)
- Development folds (used for ALL design/tuning decisions): Actualmed, Italy, China, Spain.
- Confirmation folds (NOT inspected until the final frozen-config run): UK, Australia, Other, Unknown.
- Model selection inside every run uses source-site validation data only; hyperparameters are chosen on development folds only.
- Negative results are kept in this log and reported in the paper.

## Iteration 0 - label-free diagnostics (scripts/diagnostics.py, scripts/shortcut_removal.py)
- Hypothesis: acquisition statistics alone leak the disease label across sites; site-profile augmentation removes this.
- Result: profile-only (5 numbers) logistic regression predicts COVID on held-out sites with mean AUROC 0.662 (GBM 0.588), chance 0.5.
  Prevalence spread across sites (SD) = 0.274. Site predictable from profile: acc 0.367 vs 0.224 majority-class chance.
- Site-profile augmentation did NOT reduce the leak (0.662 -> 0.660); generic aug: 0.665.
- Interpretation: the label-correlated acquisition signal is largely WITHIN-site (e.g. portable/AP vs PA protocol, patient severity),
  so randomizing the site-level signature does not remove it. This is a negative result for my original hypothesis.
- Semi-synthetic check: site-profile estimates recover known degradation order perfectly (Spearman +-1.0 on all 5 params), BUT
  the sharpness proxy increases with degradation because added noise inflates gradients -> sharpness must be noise-compensated.

## Iteration 1 - baselines and SIDL variants (dev folds, 2 seeds, 15 epochs)  [interim, in progress]
- Interim (the four development folds only; confirmation folds were NOT inspected): SIDL variants do NOT beat ERM; harmonize/sidl_cc worse by ~0.07 AUROC.
- Worst-site conformal degenerate (all sets size 2): per-site validation sets too small -> infinite quantile. Fixed (min_site_n=25).

## Iteration 1 - final (dev folds Actualmed/Italy/China/Spain, 2 seeds, 15 epochs, full-batch full-strength aug)
- AUROC: GroupDRO 0.761, ERM 0.758, DANN 0.752, sidl_phys 0.708, sidl 0.702, sidl_full 0.697, sidl_cc 0.690, harmonize 0.675.
- Every variant containing augmentation was worse than ERM (up to -0.083); methods without augmentation were tied with ERM.

## Iteration 2 - augmentation strength / per-sample probability (dev folds, 2 seeds)  [hypothesis: aggressive full-batch aug caused the drop]
| config | mean AUROC | diff vs ERM (0.758) |
|---|---|---|
| generic p=1.0 s=1.0 (it1) | 0.675 | -0.083 |
| generic p=0.3 s=0.5 | 0.748 | -0.011 |
| generic p=0.5 s=1.0 | 0.745 | -0.013 |
| site-profile p=0.3 alpha=0.5 | 0.757 | -0.001 |
| site-profile p=0.5 alpha=1.0 | 0.731 | -0.027 |
- Hypothesis CONFIRMED: the iteration-1 drop was caused by over-strong, full-batch augmentation; with mild per-sample augmentation both are ~ERM.
- Neither augmentation IMPROVES ERM on this benchmark. Decision: keep mild setting (p=0.3, strength 0.5) as default; test mechanisms on the controlled semi-synthetic benchmark.
- Also verified (ground truth): noise-compensated sharpness now decreases with true blur (Spearman -1.0; uncompensated +1.0, i.e. wrong sign). Added as `compensate=True`.

## Iteration 1b - conformal modes on iteration-1 predictions (dev folds, alpha=0.1, new min_site_n=25 for worst-site)
| method | mode | mean cov | min cov | mean set size |
|---|---|---|---|---|
| ERM | pooled | 0.829 | 0.743 | 1.325 |
| ERM | profile_weighted (new, label-free) | 0.834 | 0.732 | 1.327 |
| ERM | site_nn (new) | 0.870 | 0.751 | 1.446 |
| ERM | worst_site | 0.892 | 0.832 | 1.523 |
| GroupDRO | pooled / worst_site | 0.846 / 0.903 | 0.821 / 0.854 | 1.406 / 1.571 |
- Pooled conformal under-covers on unseen sites (ERM 0.829 vs target 0.90): site shift breaks exchangeability, as expected.
- Worst-site calibration brings ERM close to target (0.892) for ~15% larger sets.
- NEGATIVE: profile-weighted conformal (covariate = 5-d acquisition profile, density ratio from unlabeled test-site images) gives no gain over pooled.
  Decision: drop profile-weighted conformal as a contribution; keep worst-site as the recommended mode and report profile-weighted honestly as a negative.

## Iteration 3 - semi-synthetic label-shift sweep (known acquisition transforms; 5 virtual sites; holdouts V0,V2,V4; 2 seeds; 12 epochs; adv_weight=0.5, no aug)
AUROC (mean over holdouts, seeds):
| prior spread | ERM | ERM+prior-reweight | DANN | cDANN | cDANN+prior |
|---|---|---|---|---|---|
| 0.00 | 0.732 | 0.735 | 0.672 | 0.682 | 0.712 |
| 0.25 | 0.721 | 0.725 | 0.681 | 0.685 | 0.681 |
| 0.40 | 0.703 | 0.703 | 0.611 | 0.626 | 0.656 |
- Paired diff vs ERM at spread 0.4: DANN -0.092 [-0.168,-0.016], cDANN -0.077 [-0.137,-0.019], cDANN+prior -0.047 [-0.093,-0.013]; ERM+prior ~0.
- Ordering at high label shift (cDANN+prior > cDANN > DANN) matches theory (unconditional alignment is most harmful under label shift), BUT all adversaries are below ERM,
  and they are already below ERM at spread 0 (-0.05 to -0.06): the adversary at weight 0.5 hurts representation learning on this small data.
- Decision: sweep adversary weight (iteration 4); select by source-validation AUROC (deployable), report oracle selection as sensitivity.

## Iteration 4 - adversary strength sweep (semi-synthetic; spreads 0 and 0.4; 3 holdouts x 2 seeds = 6 units per cell; no aug)
Test AUROC (ERM: 0.732 at spread 0, 0.703 at spread 0.4):
| method | spread | lam=0.02 | lam=0.1 | lam=0.5 |
|---|---|---|---|---|
| DANN | 0.0 | 0.722 | 0.704 | 0.672 |
| DANN | 0.4 | 0.672 | 0.659 | 0.611 |
| cDANN+prior | 0.0 | 0.733 | 0.720 | 0.712 |
| cDANN+prior | 0.4 | 0.683 | 0.615 | 0.656 |
- Adversaries approach ERM as the weight shrinks and NEVER beat it (best: cDANN+prior lam=0.02 at spread 0: +0.001; at spread 0.4: -0.020).
- cDANN+prior vs DANN at spread 0.4: +0.045 (lam .5), -0.044 (lam .1), +0.011 (lam .02): sign flips -> the conditional-vs-unconditional advantage is NOT robust with 6 units per cell (underpowered).
- Decision: do not claim that label-shift-aware adversaries help. Move to a different mechanism (profile-adversarial invariance, iteration 5) and to test-time label-shift handling.
- Side diagnostic (shortcut_reliance.py, iteration-1 models, dev folds): within-class Spearman between model P(COVID) and a profile-only score is high for all models
  (ERM 0.456, GroupDRO 0.470, DANN 0.400, sidl_phys 0.400): classifiers' held-out scores co-vary strongly with acquisition statistics.

## Iteration 5b - test-time label-shift handling on iteration-1 ERM/GroupDRO predictions (dev folds; scripts/analyze_labelshift.py)
- EM prior estimation error |pi_hat - pi_true|: 0.186 (ERM) vs 0.258 when ignoring shift; balanced accuracy gets WORSE after adjustment (0.649 -> 0.605 EM; 0.617 with the oracle prior).
- Label-shift conformal (weighted, per-class mass): coverage 0.813 (EM prior) / 0.845 (oracle prior) vs pooled 0.829 vs worst-site 0.892.
- NEGATIVE: label-shift correction does not repair under-coverage on unseen sites; covariate/calibration shift dominates. 
- Insight: conformal scores computed on validation images of SEEN sites are optimistic for an UNSEEN site (the model has fit those sites). -> try calibration scores from
  models that never saw the calibration site (leave-site-out cross-conformal, iteration 6).

## Iteration 5 - profile-adversarial invariance (PAI) on real dev folds (2 seeds; ERM mean AUROC 0.758)
| method | adv_weight | mean AUROC | diff vs ERM |
|---|---|---|---|
| PAI (class-conditional heads) | 0.1 | 0.753 | -0.005 [-0.021, 0.010] |
| PAI (unconditional head) | 0.1 | 0.758 | -0.000 [-0.023, 0.028] |
| PAI (class-conditional) | 0.5 | 0.707 | -0.051 [-0.075, -0.027] |
| PAI (unconditional) | 0.5 | 0.722 | -0.036 [-0.082, -0.006] |
- Shortcut-reliance (within-class Spearman of P(COVID) with a profile-only score): ERM 0.456 -> PAI(cond) 0.406, PAI(uncond) 0.428.
- NEGATIVE for accuracy: removing measurable acquisition information from z does not improve held-out AUROC (equal at lam=0.1, worse at 0.5); it lowers acquisition
  sensitivity slightly. Decision: report PAI as a negative / diagnostic result; accuracy-side invariance methods do not beat ERM on this data (3 independent attempts: augmentation, site adversaries, PAI).

## Iteration 6 - leave-site-out (LSO) cross-conformal, ERM, dev folds (4 folds x 2 seeds; 72 inner models; scripts/run_lso_conformal.py, analyze_lso_conformal.py)
At alpha=0.1 (target coverage 0.90): mean / min coverage over folds, mean set size
| calibration | cov mean | cov min | size | #folds under-covered by >5pp |
|---|---|---|---|---|
| pooled (seen-site val) | 0.829 | 0.743 | 1.325 | 2/4 |
| site-balanced (seen-site val) | 0.835 | 0.746 | 1.323 | 2/4 |
| worst-site (seen-site val) | 0.892 | 0.832 | 1.523 | 1/4 |
| LSO-split | 0.888 | 0.732 | 1.544 | 1/4 |
| LSO-worst | 0.937 | 0.832 | 1.734 | 1/4 |
| LSO-CV+ (sites as folds) | 0.950 | 0.911 | 1.720 | 0/4 |
| LSO-CV+ balanced | 0.940 | 0.884 | 1.660 | 0/4 |
Across alpha in {0.04..0.30} (fold-averaged curves): coverage-calibration error (mean |cov-(1-alpha)|): LSO-split 0.012, worst-site 0.012, LSO-CV+bal 0.038, pooled 0.045, LSO-CV+ 0.057;
fraction of (fold,alpha) cells under-covered by >5pp: LSO-CV+ 0.000, LSO-worst 0.056, LSO-CV+bal 0.111, worst-site 0.167, LSO-split 0.25, pooled 0.306;
set size interpolated at 90% coverage: LSO-CV+ 1.496, LSO-CV+bal 1.498, worst-site 1.546, pooled 1.556.
- Reading: LSO-CV+ eliminates under-coverage but is conservative (over-covers); at matched coverage set sizes are about equal (-4% vs pooled). Seen-site worst-site is already well calibrated on average.
  The deployable value of LSO is therefore safety (never badly under-covers) rather than efficiency. Needs confirmation on frozen folds + controlled semi-synthetic data.

## Iteration 6b - optimism of seen-site validation vs leave-site-out estimate (ERM, dev folds, 8 fold-seed units)
| held-out site | seen-site val AUROC | leave-site-out estimate | actual held-out AUROC |
|---|---|---|---|
| Actualmed | 0.833 | 0.743 | 0.745 |
| China | 0.877 | 0.747 | 0.808 |
| Italy | 0.868 | 0.732 | 0.761 |
| Spain | 0.844 | 0.749 | 0.720 |
- Mean signed error (estimate - actual): seen-site validation +0.097; leave-site-out -0.016. Mean absolute error 0.097 vs 0.031.
- This optimism gap is the mechanism behind under-coverage of conformal sets calibrated on seen-site validation data.
- Site-bagged ensemble of the 9 inner models (free by-product): AUROC 0.765 vs 0.758 (single full model), ECE 0.122 vs 0.149 (small, within noise).

## Iteration 7 - LSO conformal on the CONTROLLED semi-synthetic benchmark (5 virtual sites, severity 0.5/1.0/1.5, prior spread 0.25; holdouts V0,V2,V4; 2 seeds)
alpha=0.1 (target 0.90): mean coverage (min) / size
| severity | pooled | site-balanced | worst-site | LSO-split | LSO-CV+ | LSO-worst |
|---|---|---|---|---|---|---|
| 0.5 | 0.896 (0.875)/1.37 | 0.892 (0.878)/1.36 | 0.909 (0.866)/1.44 | 0.933 (0.931)/1.55 | 0.928 (0.894)/1.54 | 0.964 (0.956)/1.70 |
| 1.0 | 0.922 (0.878)/1.58 | 0.910 (0.878)/1.54 | 0.912 (0.828)/1.55 | 0.918 (0.906)/1.54 | 0.931 (0.891)/1.55 | 0.965 (0.950)/1.72 |
| 1.5 | 0.921 (0.916)/1.65 | 0.908 (0.878)/1.61 | 0.920 (0.906)/1.66 | 0.834 (0.688)/1.43 | 0.931 (0.859)/1.66 | 0.960 (0.891)/1.79 |
- NEGATIVE for LSO on simulated shift: seen-site pooled/site-balanced calibration is already well calibrated (coverage 0.89-0.92); LSO variants are not better and LSO-split under-covers at severity 1.5.
- Contrast with real data (iteration 6, pooled coverage 0.829): under-coverage on real multi-source data is NOT reproduced by simulated acquisition shift;
  real inter-site shift contains site idiosyncrasies beyond acquisition statistics. Report as a benchmark-realism finding.

## Iteration 8 - mild-config SIDL variants on dev folds (aug_p=0.3, strength 0.5, adv_weight 0.1; 2 seeds) and full dev-fold comparison
| method | mean AUROC | diff vs ERM [95% CI over 4 folds] | folds better |
|---|---|---|---|
| ERM | 0.758 | - | - |
| GroupDRO | 0.761 | +0.003 [-0.016, 0.023] | 2/4 |
| DANN (adv 0.5) | 0.752 | -0.007 [-0.032, 0.019] | 2/4 |
| generic aug (mild) | 0.748 | -0.011 [-0.040, 0.020] | 1/4 |
| site-profile aug (mild) | 0.757 | -0.001 [-0.024, 0.013] | 3/4 |
| PAI (adv 0.1) | 0.753 | -0.005 [-0.021, 0.010] | 2/4 |
| SIDL-phys (mild) | 0.756 | -0.002 [-0.004, 0.001] | 1/4 |
| SIDL-cc (mild) | 0.722 | -0.036 [-0.081, 0.010] | 1/4 |
| SIDL-full (mild) | 0.738 | -0.021 [-0.059, 0.017] | 2/4 |
- Conclusion on dev folds: no invariance mechanism (augmentation, site/profile adversaries, CORAL+consistency, GroupDRO) improves AUROC over ERM; all differences are within fold noise.
- FROZEN for the confirmation run (no further tuning): erm, groupdro, dann(adv 0.5), harmonize(p .3, s .5), phys_aug(p .3, s .5), pai(adv .1), sidl_cc(mild), sidl_phys(mild), sidl_full(mild),
  and ERM leave-site-out (LSO) inner models for conformal. Confirmation folds: UK, Australia, Other, Unknown (2 seeds). Run once.

## Iteration 9 - frozen-config confirmation on untouched folds (UK, Australia, Other, Unknown; 2 seeds; 15 epochs; launched once via runs/it9.sh)
Configs exactly as frozen at the end of iteration 8; no re-runs, no tuning. Held-out-site AUROC, mean of 2 seeds (full table: paper/tables/perfold.tex):
| method | UK | Australia | Other | Unknown | mean (conf.) | mean dev (it8) |
|---|---|---|---|---|---|---|
| ERM | 0.859 | 0.565 | 0.629 | 0.775 | 0.707 | 0.758 |
| GroupDRO | 0.837 | 0.679 | 0.657 | 0.766 | 0.735 | 0.761 |
| DANN | 0.746 | 0.651 | 0.645 | 0.764 | 0.702 | 0.752 |
| generic aug | 0.828 | 0.625 | 0.619 | 0.792 | 0.716 | 0.748 |
| site-profile aug | 0.703 | 0.642 | 0.641 | 0.791 | 0.694 | 0.757 |
| PAI | 0.853 | 0.551 | 0.635 | 0.793 | 0.708 | 0.753 |
| SIDL-cc | 0.747 | 0.656 | 0.635 | 0.760 | 0.699 | 0.722 |
| SIDL-phys | 0.831 | 0.674 | 0.650 | 0.770 | 0.732 | 0.756 |
| SIDL-full | 0.782 | 0.415 | 0.626 | 0.837 | 0.665 | 0.738 |
- Neither ordering from the dev folds carried over cleanly (site-profile aug dropped from -0.001 to -0.013 vs ERM on the confirmation folds; SIDL-phys rose from -0.002 to +0.025).
- Test-set sizes make single folds very noisy: Australia has 6 positives, China 5 negatives, Hannover/Figure1 (training-only) 3 and 5 negatives.
- RESULT (all 8 folds, hierarchical paired bootstrap = resample folds, then resample test images stratified by class; 1500 reps): no method differs significantly from ERM.
  GroupDRO +0.015 [-0.015,+0.055]; SIDL-phys +0.011 [-0.016,+0.050]; generic aug -0.001; PAI -0.002; DANN -0.006; site-profile aug -0.007; SIDL-cc -0.022; SIDL-full -0.031 [-0.085,+0.016].
  Worst-fold AUROC: ERM 0.565 (Australia) vs 0.64-0.66 for GroupDRO/DANN/aug/SIDL-phys, BUT that worst fold has 6 positives and PAI/SIDL-full do not improve it, so this is not evidence of robustness.
- Balanced accuracy: GroupDRO 0.689 vs ERM 0.634 (exploratory; threshold-dependent, group reweighting shifts the operating point; not a pre-specified endpoint).

## Iteration 9b - conformal calibration on confirmation folds (ERM full models + 36 LSO inner models x 2 seeds; scripts/analyze_lso_conformal.py)
alpha=0.1 (target 0.90), mean / min coverage over the 4 confirmation folds, size: pooled 0.843/0.758/1.45; site-balanced 0.846/0.750/1.43; worst-site 0.916/0.751/1.67;
LSO-split 0.923/0.867/1.64; LSO-CV+ 0.933/0.867/1.77; LSO-worst 0.983/0.961/1.88.
- REPLICATES the dev-fold finding: pooled/site-balanced under-cover unseen sites (2/4 folds more than 5pp short); LSO-CV+ has 0/4 folds under-covered by >5pp, worst-site 1/4 (Other: 0.751).
- Size at matched 90% coverage (interpolated over alpha) is ~1.57-1.60 for EVERY mode (pooled 1.62 on confirmation folds): the calibrations lie on one coverage-size curve; they differ in operating point, not efficiency.
  (Dev folds had suggested LSO-CV+ -4% at matched coverage; on confirmation folds it is +4%. Treat as no difference.)

## Iteration 9c - site-bagged ensemble (mean of the 9 LSO inner models), all 8 folds (scripts/ensemble_check.py; pre-noted on dev folds in iteration 6b)
- AUROC 0.748 vs 0.733 (single full ERM): +0.016 [-0.008,+0.049]; ECE 0.154 vs 0.157; better on 5/8 folds. Not significant.

## Iteration 10 - consolidation (scripts/final_report.py regenerates every table, figure and number macro in the paper from runs/)
- Optimism of seen-site validation (16 fold-seed units, ERM): mean signed error vs actual held-out AUROC +0.118 (seen-site val.) vs +0.007 (leave-site-out); MAE 0.130 vs 0.070.
  Neither estimator ranks held-out sites (Spearman across 8 folds: seen-site -0.31, LSO -0.05); fold-level AUROC varies 0.56-0.86 while the LSO estimate sits at 0.72-0.75 for all folds.
  Spearman between per-fold optimism gap and pooled-conformal coverage across the 8 folds: -0.45 (n=8; same sign as the mechanism, not significant).
- Final claims supported by the evidence are listed in the paper abstract; unsupported claims (any accuracy gain from invariance/augmentation; label-shift-aware advantage; profile-weighted or label-shift conformal; simulation reproducing real under-coverage) are reported as negative results.
