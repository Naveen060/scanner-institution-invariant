# Scanner- and institution-invariant diagnosis: a leave-site-out study

Code, frozen experiment logs and manuscripts. Two manuscripts are kept: the 8-page single-column version *Scanner-Invariant and Institution-Invariant Deep Learning for Reliable Medical Image Diagnosis Across Heterogeneous Clinical Environments* (`paper/main.pdf`, CNN results only) and the IEEE two-column version (`paper/ieee/main.pdf`, same title, without the subtitle), which adds a second, pretrained-encoder family (iterations 11-17) and the robustness checks of iterations 19-20.

## What this project found (all numbers are produced by `scripts/final_report.py` from `runs/`)

Setting: open multi-source chest-radiograph benchmark (924 images, 10 sources, COVID-19 vs other), small CNN trained from scratch at 64x64 on CPU, eight held-out sites (4 development, 4 confirmation with frozen configs), two seeds.

1. **Negative accuracy result.** None of nine methods (ERM, GroupDRO, DANN, generic augmentation, site-profile augmentation, profile-adversarial invariance, and three SIDL combinations) is significantly better than ERM at held-out-site AUROC (hierarchical paired bootstrap; all 95% intervals include 0; largest gains GroupDRO +0.015, SIDL-phys +0.011).
2. **Label-free leak diagnostic.** Five acquisition statistics predict the diagnosis on held-out sites with AUROC 0.66.
3. **Conformal prediction for unseen sites.** Seen-site validation overstates held-out AUROC by 0.118; pooled split-conformal sets cover 0.836 on average at a 0.90 target (4 of 8 sites more than 5 points short). A leave-site-out CV+ rule (sites as folds) leaves no site more than 5 points short at ~26% larger sets; all calibrations lie on one coverage-size curve.
4. **Simulation realism.** Controlled simulated acquisition shift does not reproduce the real under-coverage.

Both manuscripts use the original title. It promises more than the evidence supports (one modality, one task, no benefit from the invariance recipe), so the abstract states the negative result up front; an earlier draft of the IEEE version used a question-style title (*Does Site Invariance Help at Unseen Hospitals? ...*) and was reverted at the author's request.

### Round 2: a second encoder family (iterations 11-17, `experiments/ITERATIONS.md`)

Frozen TorchXRayVision DenseNet-121 embeddings (224x224) with small trainable heads, five seeds, same development/confirmation protocol (settings frozen and two primary comparisons written down before the first confirmation run).

5. **The negative result replicates.** No invariance head beats ERM on the embedding; GroupDRO is again the largest (+0.018 [-0.000, +0.039] over eight sites, uncorrected for nine comparisons); transductive site centring is significantly worse (-0.040).
6. **The conformal and optimism findings replicate.** Pooled coverage 0.834 (pretrained head) and 0.836 (ensemble) vs target 0.90, 5 of 8 sites more than 5 points short; LSO-CV+ 0.937 / 0.936 with none; seen-site validation optimism +0.115 (CNN +0.118).
7. **Suggestive, not significant:** averaging five CNNs with five pretrained heads gave the best mean and worst-site AUROC (0.780 / 0.671 vs 0.727 / 0.595 for a single CNN; +0.053 [-0.013, +0.129]); same sign on development and confirmation folds, but every interval includes zero and on the confirmation folds the pretrained ensemble alone was as good.

### Round 3: reviewer-style checks and a plainer manuscript (iterations 19-20)

8. **The coverage shortfall is statistically real and lives in one class.** Hierarchical bootstrap 95% intervals for mean pooled coverage at a 0.90 target: CNN [0.784, 0.890], pretrained [0.786, 0.883], ensemble [0.798, 0.876]; LSO-CV+ intervals sit at or above 0.90. Under pooled calibration non-COVID images are covered 0.74-0.75 and COVID images 0.90-0.93.
9. **Two earlier hints were weaker than they looked.** Without the two tiny test sets (China: 5 negatives, Australia: 6 positives) the CNN GroupDRO gain is 0.000 [-0.026, +0.024] and the ensemble's lead over a single CNN falls from +0.053 to +0.012 [-0.041, +0.065]. Pretrained-head GroupDRO is +0.025 [+0.005, +0.049] on those six sites (a post hoc subset).
10. **Nothing survives multiple-comparison correction.** Eleven second-study comparisons: smallest raw p 0.028 (site centering, worse than ERM), GroupDRO 0.052; smallest Holm-adjusted p 0.31.

The IEEE manuscript was rewritten in plainer prose (research questions up front, SIDL defined, number-heavy paragraphs moved into tables, practical-guidance and threats-to-validity subsections); `scripts/text_checks.py` runs LanguageTool and a readability report on the compiled PDF. Grammarly, QuillBot and iThenticate could not be used here; the similarity-risk advice is in `paper/ieee/SUBMISSION_CHECKLIST.md`.

## Layout

| Path | Purpose |
|---|---|
| `sidl/` | library: `model` (SmallCNN, class-conditional adversary, profile adversary), `losses` (CORAL, consistency, GroupDRO), `augment` (generic), `siteprofile` (site-profile augmentation), `conformal`, `pwconformal`, `labelshift`, `synth` (semi-synthetic benchmark), `trainer`, `data`, `arraydata`, `metrics` |
| `scripts/build_covid_cxr.py` | build the benchmark from three public GitHub collections |
| `scripts/run_array_loso.py` | resumable leave-one-site-out runner (JSONL log) |
| `scripts/run_lso_conformal.py`, `analyze_lso_conformal.py` | leave-site-out inner models and conformal analysis |
| `scripts/run_synth.py`, `run_synth_lso.py`, `analyze_synth.py` | semi-synthetic label-shift and acquisition-shift experiments |
| `scripts/diagnostics.py`, `shortcut_removal.py`, `shortcut_reliance.py` | label-free diagnostics |
| `scripts/ensemble_check.py` | site-bagged ensemble check |
| `scripts/final_report.py` | regenerates every table, figure and number macro in the single-column paper |
| `scripts/extract_xrv_features.py`, `sidl/featheads.py`, `scripts/run_feat_loso.py`, `scripts/run_feat_lso_conformal.py` | round 2: frozen pretrained embeddings, feature-space training heads, leave-one-site-out and leave-site-out conformal runners |
| `scripts/feat_compare.py`, `scripts/round2_report.py`, `scripts/make_ensemble_runs.py`, `scripts/feat_diagnostics.py`, `scripts/round2_tables.py` | round 2: paired comparisons, ensemble builder (feeds `analyze_lso_conformal.py`), embedding diagnostics, and the tables, figure and `paper/numbers_r2.tex` for the IEEE paper |
| `scripts/reviewer_checks.py` | robustness checks: bootstrap intervals and per-class conformal coverage, six-site sensitivity, Holm-adjusted p-values (writes `runs/reviewer_checks.{txt,json}`) |
| `scripts/text_checks.py` | prose checks on the compiled PDF (LanguageTool, readability, repetition); not part of the scientific pipeline |
| `paper/ieee/` | IEEE manuscript (`main.tex`, `refs.bib`) and `SUBMISSION_CHECKLIST.md` |
| `runs/it*.sh` | the exact launch commands for each iteration |
| `experiments/ITERATIONS.md` | log of every iteration, including failures |
| `LITERATURE_NOTES.md` | related-work notes (verified entries are marked) |
| `run_experiment.py`, `prepare_manifest.py`, `make_tables.py`, `configs/` | a generic manifest-based pipeline for other datasets; only smoke-tested, not used for the reported results |

## Reproduce

```bash
pip install -r requirements.txt
python scripts/unit_tests.py
# rebuild the benchmark (clone the three public repos under one root first):
#   ieee8023/covid-chestxray-dataset, agchung/Actualmed-COVID-chestxray-dataset, agchung/Figure1-COVID-chestxray-dataset
python scripts/build_covid_cxr.py --root /path/to/repos --out data/covid_cxr
bash runs/it1.sh   # ... through runs/it9.sh (CPU only; several hours on 2 cores)
python scripts/final_report.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

# round 2 (pretrained-encoder family): 224px cache, frozen embeddings, then the launch commands in runs/it13.sh - runs/it15.sh
python scripts/build_covid_cxr.py --root /path/to/repos --size 224 --out data/covid_cxr224
pip install torchxrayvision
python scripts/extract_xrv_features.py
python scripts/run_feat_loso.py --out runs/it11 --methods erm erm_linear --seeds 0 1 2 3 4 --holdouts Actualmed Italy China Spain
python scripts/round2_report.py && python scripts/reviewer_checks.py && python scripts/round2_tables.py
cd paper/ieee && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`runs/cnn_extra/` holds three extra CNN ERM seeds (2-4) run after the freeze; the single-column paper's tables use seeds 0-1 only, so its numbers are unchanged.

## Protocol guarantees

- The held-out site is never used for training, validation, model selection or conformal calibration.
- Development folds (Actualmed, Italy, China, Spain) were used for all design decisions; confirmation folds (UK, Australia, Other, Unknown) were run once with frozen configs. Early partial iteration-1 outputs on two confirmation sites are excluded from every analysis.
- Statistics: hierarchical paired bootstrap (resample sites, then test images stratified by class).

## Honest notes

- The iteration-1 *augmentation* numbers (e.g. generic augmentation 0.675 AUROC) came from an earlier whole-batch augmentation implementation that was replaced in iteration 2; they are logged but not reproducible with the final code. The final tables use iteration 1 only for the augmentation-free baselines (ERM, GroupDRO, DANN).
- Single modality; small, shortcut-prone, retrospective data; from-scratch 64x64 CNN (two seeds) and, in round 2, a frozen pretrained encoder (five seeds, not fine-tuned); no private data used.
- Round 2: the CNN confirmation results were known before the pretrained family was designed; overlap between benchmark images and the encoder's pretraining collections was not checked; the intervals in the tables are not multiplicity-corrected (Holm-adjusted p-values over the eleven second-study comparisons are in iteration 19; none is significant).
- Leave-site-out conformal rules have no finite-sample guarantee under site shift; coverage is measured empirically.
- Novelty statements in the paper are hedged: related work was searched, not systematically reviewed. Worst-site conformal calibration is an existing idea.
