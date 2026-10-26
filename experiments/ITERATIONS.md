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

# Round 2 (journal-readiness): does a second encoder family change the conclusions?
Motivation: a reviewer's first question about iterations 1-10 is whether "no invariance method beats ERM" and "pooled conformal under-covers unseen sites" are properties of a small from-scratch CNN. Round 2 tests this with a frozen pretrained chest X-ray encoder.
Same rules as round 1: development folds (Actualmed, Italy, China, Spain) guide every decision; confirmation folds (UK, Australia, Other, Unknown) are run once with frozen configs. Negative results are reported as negative.
Encoder: torchxrayvision DenseNet121 "all" weights (trained on NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, RSNA; no COVID-19 labels), frozen, 224x224 input, 1024-d penultimate embedding (scripts/extract_xrv_features.py).
Not checked: whether benchmark images also appear in those pretraining sets.

## Iteration 11 - pretrained frozen encoder + small head (ERM), development folds, 5 seeds (scripts/run_feat_loso.py, sidl/featheads.py)
Held-out AUROC (mean of 5 seeds): Actualmed 0.619, Italy 0.684, China 0.906, Spain 0.749; mean 0.739 (seed-to-seed sd of the 4-fold mean 0.038). Linear head: 0.720.
From-scratch CNN ERM on the same folds (2 seeds): 0.745 / 0.760 / 0.808 / 0.720; mean 0.758.
- NEGATIVE: the pretrained encoder is not better overall. It is clearly better on China (+0.10) and Spain (+0.03) and worse on Actualmed (-0.13) and Italy (-0.08).
- Head hyperparameter grid (epochs 15/60, projection 64, dropout 0.5, weight decay 0.1, lr 3e-4, lr 3e-4 + 60 epochs): mean dev AUROC 0.737-0.743 for all seven settings. Tuning the head does not change the conclusion; default head kept.

## Iteration 12 - invariance heads on the pretrained features, development folds, 5 seeds
Same methods as round 1, implemented on the embedding (ERM, prior reweighting, GroupDRO, DANN, class-conditional adversary, +prior, CORAL, class-conditional recipe without augmentation, transductive per-site feature centring). Three settings: adversary weight 0.5 (default), 0.1, and CORAL weight 1.0 + adversary 0.1.
Mean dev AUROC (ERM 0.739): GroupDRO 0.760 (+0.021), class-conditional recipe 0.741, CORAL 0.740, class-conditional adversary 0.739, prior reweighting 0.738, DANN 0.736, site centring 0.715 (-0.024, uses target features), class-conditional adversary + prior 0.706 (-0.033); weaker-adversary and stronger-CORAL settings 0.709-0.737 (all below ERM).
- NEGATIVE, replicates round 1: no invariance method beats ERM. GroupDRO is again the largest positive difference (round 1: +0.015 on all folds), paired bootstrap +0.021 [-0.007, +0.051] vs pretrained ERM.
- Sensitivity to the adversary weight (0.1 vs 0.5) and to CORAL weight changes means by up to 0.03, comparable to the seed sd; there is no setting that helps.
- Transductive site centring (needs unlabeled target images) does best on Actualmed (0.664 vs 0.619) but loses 0.09 on China; mean is worse than ERM.

## Iteration 13 - does combining the two encoder families help? (development folds, 5 seeds each; scripts/feat_compare.py)
Added 3 seeds of the unchanged from-scratch ERM (runs/it13.sh) so both families have 5 seeds on the development folds. Ensemble = mean predicted probability.
Held-out AUROC, dev folds (Actualmed / Italy / China / Spain; mean):
| model | Actualmed | Italy | China | Spain | mean |
|---|---|---|---|---|---|
| CNN ERM, single (5-seed avg) | 0.767 | 0.756 | 0.794 | 0.702 | 0.755 |
| pretrained ERM, single (5-seed avg) | 0.619 | 0.684 | 0.906 | 0.749 | 0.739 |
| CNN 5-model ensemble | 0.786 | 0.766 | 0.800 | 0.735 | 0.772 |
| pretrained 5-model ensemble | 0.646 | 0.686 | 0.914 | 0.754 | 0.750 |
| CNN + pretrained (5+5 ensemble) | 0.725 | 0.751 | 0.902 | 0.778 | 0.789 |
- Paired hierarchical bootstrap (folds, then images by class): cross-family ensemble vs CNN single +0.025 [-0.042,+0.099] (matched seeds); vs CNN 5-model ensemble +0.017 [-0.050,+0.096]; vs pretrained 5-model ensemble +0.039 [-0.005,+0.080].
- Averaging seeds within the CNN family already helps: +0.017 [+0.002,+0.032].
- DIRECTION POSITIVE, NOT SIGNIFICANT on 4 development folds. The ensemble is best or near-best on 3 of 4 folds but loses to the CNN on Actualmed (0.725 vs 0.767 single).
- Caution from this iteration: two-seed averages are unreliable (pretrained ERM seeds 0-1 averaged 0.766, seeds 0-4 averaged 0.739). All comparisons from here use 5 seeds.
- A GroupDRO pretrained head added to the ensemble gives 0.796 (+0.007 [-0.010,+0.024]); not carried forward (GroupDRO is a dev-fold pick among nine comparisons).

## Iteration 14 - FROZEN before any confirmation fold was run for the pretrained family
- Pre-specified primary comparisons (confirmation folds UK, Australia, Other, Unknown, plus all 8 folds): (A) CNN + pretrained ERM ensemble (5+5, mean probability) vs the CNN 5-model ensemble; (B) same ensemble vs single CNN ERM (5-seed average).
- Pre-specified secondary: the nine feature-head methods at the default settings of iteration 12 (adversary weight 0.5, CORAL weight 0.1; no per-fold selection) vs pretrained ERM, to check whether the "no invariance gain" result replicates on the confirmation folds.
- Pretrained ERM uses the iteration-11 default head (the grid showed no improvement). From-scratch ERM gets seeds 2-4 on the confirmation folds with the config frozen since iteration 8 (runs/it14.sh); seeds 0-1 were already run in iteration 9.
- Run once. Whatever comes out is reported.

## Iteration 14 - RESULT of the frozen confirmation run (UK, Australia, Other, Unknown; 5 seeds per model; run once; scripts/round2_report.py -> runs/round2.json)
Held-out AUROC, mean over sites (dev 4 / conf 4 / all 8) and worst site:
| model | dev | conf | all | worst |
|---|---|---|---|---|
| CNN ERM (5-seed avg) | 0.755 | 0.699 | 0.727 | 0.595 (Australia) |
| pretrained ERM (5-seed avg) | 0.739 | 0.758 | 0.749 | 0.619 (Actualmed) |
| CNN 5-model ensemble | 0.772 | 0.716 | 0.744 | 0.617 |
| pretrained 5-model ensemble | 0.750 | 0.771 | 0.760 | 0.646 |
| CNN + pretrained (5+5 ensemble) | 0.789 | 0.770 | 0.780 | 0.671 (Other) |
- Pre-specified (A): ensemble - CNN 5-model ensemble: dev +0.017 [-0.050,+0.096]; conf +0.055 [-0.062,+0.181]; all +0.036 [-0.031,+0.110].
- Pre-specified (B): ensemble - single CNN ERM: dev +0.034 [-0.034,+0.111]; conf +0.071 [-0.043,+0.194]; all +0.053 [-0.013,+0.129].
- RESULT: POSITIVE IN DIRECTION ON BOTH FOLD SETS (same sign on dev and confirmation), best mean and best worst-site AUROC, but NOT significant (all hierarchical 95% intervals include zero). Reported as suggestive.
- The two families fail on different sites: the CNN is weak on Australia (0.595) and China-like small sites, the pretrained head on Actualmed (0.619) and UK (0.646); the ensemble is never the worst on those and has the best worst-site value. Per-site, the ensemble beats the single CNN on 5/8 sites.
- On the confirmation folds the pretrained model alone (0.771 as a 5-model ensemble) is as good as the ensemble (-0.001 [-0.069,+0.066] for ensemble minus pretrained ensemble); on the development folds the sign was the other way. The data do not separate "ensemble" from "pretrained is better".
- Seed averaging alone helps the CNN: 5-model ensemble minus single, +0.017 [+0.006,+0.030] over 8 sites.
- Secondary (nine feature heads at the frozen defaults vs pretrained ERM, all 8 sites): GroupDRO +0.018 [-0.000,+0.039] (dev +0.021, conf +0.016); prior reweighting -0.006; DANN -0.002; class-conditional adversary -0.001; CORAL -0.001; class-conditional recipe -0.006 [-0.026,+0.011]; adversary+prior -0.025 [-0.062,+0.005]; transductive site centring -0.040 [-0.076,-0.006] (significantly worse).
  REPLICATES iteration 9: no invariance method helps; GroupDRO is again the best single method, with the same sign in two encoder families and two fold sets, effect ~0.02, interval touching zero, nine comparisons uncorrected. Treated as suggestive only.

## Iteration 15 - conformal calibration on the pretrained embeddings (ERM head; 8 sites x 3 seeds; 8 LSO inner models per site-seed on the 9 source sites; scripts/run_feat_lso_conformal.py)
alpha = 0.1 (target 0.90), all 8 sites, mean coverage / min / sites >5pp short / mean size:
pooled 0.824 / 0.737 / 6 of 8 / 1.32; site-balanced 0.812 / 0.690 / 6 / 1.30; worst-site 0.879 / 0.782 / 2 / 1.51; LSO-split 0.895 / 0.820 / 2 / 1.57; LSO-CV+ 0.937 / 0.866 / 0 / 1.69; LSO-worst 0.961 / 0.899 / 0 / 1.83.
Set size at matched 90% coverage (alpha grid): 1.53-1.60 for every calibration (pooled 1.575, LSO-CV+ 1.526).
- REPLICATES (positive for the conclusions, negative for any efficiency claim): with a different encoder family, seen-site calibration still under-covers unseen sites (6/8 sites), LSO-CV+ removes the large shortfalls (0/8) at ~28% larger sets, and all calibrations lie on one coverage-size curve.
- The CNN pipeline reproduces the paper numbers exactly when run through the same code on seeds 0-1 (pooled 0.836, LSO-CV+ 0.942, sizes 1.389 and 1.747), which checks the analysis path.

## Iteration 16 - conformal calibration for the CNN + pretrained ENSEMBLE (seeds 0-1; scripts/make_ensemble_runs.py, analyze_lso_conformal.py)
Ensemble probabilities average the CNN and the pretrained head (same seed), for the full model, the LSO inner models and the held-out predictions. Calibration images and held-out images are identical across the families (asserted).
all 8 sites, alpha = 0.1: pooled 0.836 / 0.773 / 5 of 8 / 1.29; site-balanced 0.821 / 0.733 / 6 / 1.27; worst-site 0.889 / 0.773 / 2 / 1.48; LSO-split 0.904 / 0.781 / 2 / 1.51; LSO-CV+ 0.936 / 0.870 / 0 / 1.64; LSO-worst 0.973 / 0.914 / 0 / 1.79.
- A better model does not remove the coverage gap: the pooled shortfall (0.836 vs 0.90) is the same as for the single CNN (0.836).
- Matched-coverage set size (alpha grid): ensemble 1.48-1.52 vs CNN 1.57-1.60 vs pretrained 1.51-1.58. Per-site paired difference at matched 90% coverage, ensemble minus CNN: pooled -0.042 [-0.123,+0.042], LSO-CV+ -0.059 [-0.117,+0.005] (6 of 8 sites smaller). Direction consistent with the higher AUROC; intervals include zero.

## Iteration 17 - descriptive checks on the pretrained embeddings (scripts/feat_diagnostics.py -> runs/feat_diagnostics.json)
- Site is decodable from the frozen embedding: patient-grouped 5-fold logistic regression predicts the source among 10 with accuracy 0.473 (majority class 0.224). The encoder does not remove site information.
- Seen-site validation optimism of the pretrained ERM head: validation AUROC 0.863 vs held-out 0.749 = +0.115 (MAE 0.145, 40 site-seed units); CNN +0.118. Replicates.

## Iteration 18 - consolidation for the IEEE manuscript (paper/ieee/main.tex; scripts/round2_tables.py)
- Tables, figure and number macros for round 2 are generated from runs/ by scripts/round2_report.py (comparisons) and scripts/round2_tables.py (tables, fig_round2, paper/numbers_r2.tex). Hand-typed numbers in the manuscript are limited to ones already in this log (checked against it).
- The paper's conformal comparison across families uses seeds 0-1 for all three families so they are matched (CNN 0.836/0.942, pretrained 0.834/0.937, ensemble 0.836/0.936 for pooled/LSO-CV+; sites >5pp short 4/5/5 for pooled). Iteration 15 above used three seeds for the pretrained head (pooled 0.824, 6 of 8 short); the two-seed numbers are the ones in the paper. Matched-coverage set sizes in the paper table (CNN 1.57-1.60, pretrained 1.51-1.57, ensemble 1.48-1.52) use a monotone-coverage interpolation; iteration 15/16 text above gives the unsmoothed ranges (differences of at most 0.01).
- Per-site paired set size at matched 90% coverage, ensemble minus CNN (scripts/round2_tables.py, seeds 0-1): LSO-CV+ -0.059 [-0.120, +0.002], smaller at 6 of 8 sites; pooled -0.042 [-0.123, +0.042], smaller at 5 of 8.
- Housekeeping found while consolidating: the three extra CNN ERM seeds had been written into the folders the single-column paper reads (runs/it1, runs/conf_erm), which would have changed its ERM numbers; they were moved to runs/cnn_extra and the original report was re-run: all tables and number macros unchanged.
- Figures regenerated with embedded TrueType fonts (no Type 3) and a column-sized optimism figure; numbers unchanged.
- Not done (environment limits): a second public multi-site dataset (hosts blocked), fine-tuning the encoder, checking overlap between the benchmark and the encoder's pretraining images.

## Iteration 19 - reviewer-style robustness checks (scripts/reviewer_checks.py -> runs/reviewer_checks.json, runs/reviewer_checks.txt; descriptive, nothing earlier was changed)
1. Is the under-coverage more than noise? Hierarchical bootstrap (sites, then images; 3000 reps; seeds 0-1 averaged), mean coverage over 8 sites at alpha = 0.1, target 0.90:
   pooled: CNN 0.836 [0.784, 0.890], pretrained 0.834 [0.786, 0.883], ensemble 0.836 [0.798, 0.876]; fraction of replicates >= 0.90: 0.010 / 0.007 / 0.003.
   LSO-CV+: CNN 0.942 [0.908, 0.971], pretrained 0.937 [0.902, 0.968], ensemble 0.936 [0.897, 0.972].
   Worst-site and LSO-split intervals contain 0.90 (e.g. CNN worst-site 0.904 [0.845, 0.954]).
   -> pooled under-coverage is statistically distinguishable from the target in all three model families; LSO-CV+ is above it.
2. Coverage by class (mean over sites with >= 5 images of the class), pooled: non-COVID images 0.741 (CNN) / 0.745 (pretrained) / 0.747 (ensemble), COVID images 0.902 / 0.915 / 0.926.
   LSO-CV+: non-COVID 0.906 / 0.879 / 0.876, COVID 0.983 / 0.985 / 0.990. The marginal shortfall is concentrated in the non-COVID class.
3. Sensitivity to the two smallest test sets (China: 5 negatives; Australia: 6 positives), 6 remaining sites:
   CNN methods vs ERM: all within [-0.027, 0.000] (GroupDRO -0.000 [-0.026, +0.024] vs +0.015 on 8 sites): the CNN GroupDRO gain came from the two tiny sites.
   Pretrained GroupDRO vs ERM: +0.025 [+0.005, +0.049] (8 sites: +0.018 [-0.000, +0.039]). A post hoc subset and one of eleven comparisons; Holm-adjusted p over the eleven pre-listed comparisons is 0.52 (raw 0.052).
   Ensemble minus single CNN: +0.012 [-0.041, +0.065] (8 sites: +0.053): most of the ensemble's advantage came from the two tiny sites (largest on Australia, 0.595 -> 0.836). Worst site on 6 sites: CNN 0.628, pretrained 0.619, ensemble 0.671.
   -> the ensemble result is weaker than the 8-site summary suggests; the manuscript says so.
4. Multiplicity (11 two-sided bootstrap p-values: 8 pretrained heads vs ERM, ensemble vs CNN, ensemble vs five-CNN ensemble, pretrained vs CNN): smallest raw p 0.028 (site centring, the worse direction), then GroupDRO 0.052; after Holm correction the smallest is 0.31. Nothing survives correction.

## Iteration 20 - manuscript rewrite for readability (no new experiments; text and presentation only)
Changes to paper/ieee/main.tex: the Introduction now states four research questions; SIDL is defined where it first appears; "iteration N" wording is gone from the text (the log keeps it); number-heavy paragraphs point to the tables; a robustness subsection reports iteration 19; a practical-guidance subsection and a threats-to-validity subsection (internal, external, construct) replace the single Limitations paragraph; the two claims that iteration 19 weakened (CNN GroupDRO gain, size of the ensemble's lead) are stated as weakened in the abstract, results and discussion.
Prose checks (scripts/text_checks.py on the compiled PDF, LanguageTool 6.8 offline plus a vowel-group readability estimate; Grammarly, QuillBot, iThenticate and Turnitin were not available):
  before: mean sentence length 22.8 words, 27 sentences over 35 words, Flesch-Kincaid grade 11.6 (heuristic), 8 LanguageTool flags after dropping spelling noise;
  after:  mean 18.2 words, 11 sentences over 35 words, grade 10.0, 5 flags, all of them false positives or PDF-extraction artefacts (a line-break hyphen in COVID-19, a heading read as running text, a stray quote at a column break).
Spelling changed to US ("centering", "semisynthetic"). No number changed except those added from iteration 19. Build: 10 pages, no undefined references, no Type 3 fonts.

## Iteration 21 - strict IEEE conformance pass (paper/ieee/main.tex, refs.bib, scripts/figstyle.py; text and presentation only; no result changed)
Checked against the IEEE Transactions author instructions. Changes:
- Front matter: author line without "Dr." or designations (IEEE author lines carry neither; designations belong in the biographies); unnumbered first-page footnote with the manuscript-date placeholders, corresponding-author placeholder and the shared affiliation; abstract cut from 254 to 247 words with every abbreviation spelled out; four alphabetized index terms.
- Structure: Roman-numeral section headings, lettered subsection headings in title case, equations numbered and cited as "(1)" to "(4)", the per-site table moved to an appendix that precedes the Acknowledgment, the unnumbered Data/Ethics/Conflict sections folded into Section IV and the submission form (ethics and code-availability sentences are now in the text; the repository is reference [40]); abbreviations defined at first use in the body.
- Tables: Roman numerals, short title-case titles above, notes below, 8 pt text; five wide tables set across both columns.
- Figures: regenerated with scripts/figstyle.py (FIG_PROFILE=ieee): Liberation Sans (Arial metrics) at 8 pt, no title inside the graphic, "(a)/(b)" labels below each part, circle/triangle markers and hatching so colour is not the only cue, vector PDF with TrueType fonts plus 600 dpi PNG, printed sizes of 3.5 in or 7.16 in; IEEE file names in paper/ieee/figures_for_submission (rao1 to rao5).
- References: IEEE style (initials first, abbreviated venues, "et al." above six authors, article numbers, no repeated-name dashes, capitalized "Accessed"); the IEEEtranBSTCTL control entry must be cited before the first \cite, otherwise BibTeX ignores it (it was ignored in the first build of this pass; fixed, which turned seven-author-plus entries into "First Author et al."). No DOIs, months or conference page ranges were added because they could not be verified offline.
- Verification: pdflatex + bibtex, 10 pages, Letter, no overfull boxes or undefined references, all fonts embedded, 0 Type 3 fonts. Scripts re-run deterministically; tables and numbers*.tex unchanged. The single-column 8-page paper still builds at 8 pages.
- Word file: rebuilt from the same source (paper_ieee.docx); validated against its pandoc source, rendered through LibreOffice (11 pages). Equations are Unicode text in the Word file because Word equation objects could not be rendered here. Not opened in Microsoft Word.

## Iteration 22 - final audit: semisynthetic image reuse found, runs repeated without reuse (iterations 3, 4 and 7 superseded)
Trigger: a section-by-section audit of paper/ieee/main.tex against the code, to check that every statement about the benchmark could be traced to the code that produced it.
Finding: in `sidl/synth.py`, when the pool of one class ran out while filling the virtual sites, `make_virtual_sites` drew again from the whole class pool, so the same radiograph could appear at more than one virtual site (with a different transform and noise draw). The original runs used 160 images per site; measured over the settings used, 14-23% of the images were repeats. The effect on the earlier results is unknown without a rerun, so the runs were repeated.
Change: `make_virtual_sites(..., strict=True)` raises instead of recycling; `run_synth.py` and `run_synth_lso.py` take `--strict`, and `run_synth_lso.py` takes `--per-site`. Every semisynthetic experiment was repeated with `--strict --per-site 140` (`runs/it22.sh`; outputs `runs/it3_nr`, `runs/it4_nr_l002`, `runs/it4_nr_l01`, `runs/it7_nr`; `scripts/analyze_lso_conformal.py` reproduces the original CSV exactly from the original runs, so the pipeline itself is unchanged). The old directories stay in the repository as the superseded record. `scripts/final_report.py` reads the `_nr` runs by default (`SYNTH_RUNS=""` selects the superseded ones).
Results, old (with reuse) -> new (no reuse), same seeds, units and methods:
- Label-shift sweep, held-out AUROC, ERM at spreads 0 / 0.25 / 0.4: 0.732 / 0.721 / 0.703 -> 0.720 / 0.745 / 0.740.
- Spread 0.4, paired difference to ERM: DANN -0.092 [-0.175, -0.016] -> -0.137 [-0.229, -0.066]; cDANN -0.077 [-0.142, -0.019] -> -0.056 [-0.096, -0.022]; cDANN + prior -0.047 [-0.092, -0.013] -> -0.010 [-0.045, +0.022]; ERM + prior -0.000 [-0.040, +0.038] -> +0.030 [+0.004, +0.065].
- Adversary weight sweep: the old statement "lowering the weight moves the adversaries toward ERM and never above it" no longer holds. At spread 0.4 and weights 0.1 / 0.02, DANN is 0.743 / 0.736 and cDANN + prior is 0.743 / 0.756 against ERM 0.740; ERM + prior alone is 0.770. So the gain is from the reweighting, not the adversary. At weight 0.5 and spread 0 the adversaries still lose 0.017-0.032.
- Conformal by severity (0.5 / 1.0 / 1.5): pooled coverage 0.896 / 0.922 / 0.921 -> 0.907 / 0.912 / 0.898; LSO-split at severity 1.5 0.834 (worst site 0.688) -> 0.925 (0.861); LSO-CV+ 0.928 / 0.931 / 0.931 -> 0.923 / 0.929 / 0.932; worst-site at severity 1.5 0.920 -> 0.881.
- Profile recovery (Spearman of each of the five profile statistics against site degradation, 6 settings): mean and contrast fall, noise and sharpness and bias rise, every |rho| >= 0.90 (the old runs gave the same signs). Sharpness rises with degradation because the added noise outweighs the added blur.
Conclusions affected: (1) "reweighting alone is neutral" became "+0.030 [+0.004, +0.065], six units, four uncorrected comparisons, and not seen on the real sites"; (2) "adversaries never beat ERM at lower weights" was dropped; (3) "LSO-split under-covers at the highest severity" was dropped; (4) the central claim stands and is stronger: simulated acquisition shift does not reproduce the real under-coverage (pooled 0.898-0.912 on the simulation against 0.836 on real sites). Both manuscripts and all semisynthetic tables, numbers and macros were regenerated.
Other audit fixes in this pass (text only): "nine comparisons" corrected to eight (nine feature heads include ERM itself) in the README and the 8-page paper; a near-duplicate check was added (`scripts/duplicate_check.py`, `runs/duplicate_check.json`: 27 pairs of 32x32 versions with correlation above 0.95, only 2 across sources); related work and protocol text revised against the primary sources (see the checklist); a research-question-to-evidence table was added to the Discussion.

## Iteration 23 - post hoc check of the uncorrected sharpness statistic (no decision used it)
Why: the audit found that the site-profile statistics use `_sharpness(..., compensate=False)`, the code default: the gradient energy includes the white-noise term, so added noise raises "sharpness" (on the semisynthetic benchmark the statistic rises with degradation although blur lowers it). The reported site-profile augmentation, PAI and leak results all use this uncorrected version. The code also has a noise-corrected version that was never used.
What: repeat site-profile augmentation (aug_p 0.3, strength 0.5) and PAI (adversary weight 0.1, profile weight 1.0) on all eight held-out sites, seeds 0 and 1, with `compensate=True` and everything else frozen (`runs/it23.sh`; outputs `runs/it23_phys`, `runs/it23_pai`); `scripts/compensate_check.py` compares them with the reported runs using the main-table hierarchical paired bootstrap and repeats the label-leak diagnostic with the corrected statistic (`runs/compensate_check.json`).
A first attempt passed `compensate=1`; `run_array_loso.py` parsed boolean overrides as `v == "True"`, so the flag was read as False and the runs reproduced the reported phys_aug runs exactly (max |difference| in AUROC 0.0 over 16 runs; kept in `runs/it23_noop_phys` as a determinism check; no earlier run passed a boolean override, so no earlier result is affected). The parsers in `run_array_loso.py`, `run_synth.py` and `run_lso_conformal.py` now accept 1/true/yes.
Results (mean over eight sites; paired difference to ERM [hierarchical bootstrap 95% interval]): site-profile augmentation uncorrected 0.726, -0.007 [-0.060, +0.038]; corrected 0.744, +0.011 [-0.023, +0.057]. PAI uncorrected 0.730, -0.002 [-0.020, +0.014]; corrected 0.727, -0.006 [-0.029, +0.019]. Most of the change for site-profile augmentation is at UK (0.703 -> 0.848 over two seeds), where one reported seed had AUROC 0.543 (the other 0.863; the corrected runs gave 0.826 and 0.870): seed variance at a single small site, not evidence that the correction helps. Label-leak AUROC (logistic regression on the five statistics, leave-one-site-out): 0.662 uncorrected, 0.660 corrected.
Conclusion: no change to any conclusion. Reported in the manuscript as a fifth, post hoc robustness check; the limitation (uncorrected sharpness) is stated where the statistics are introduced.
Both manuscripts, the IEEE Word file and the source package were rebuilt after iterations 22 and 23. The IEEE manuscript is now 12 pages (it was 10): added related work, protocol and data details, the semisynthetic subsection, two evidence tables, the research-question table and the fifth robustness check.
