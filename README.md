# Scanner- and institution-invariant diagnosis: a leave-site-out study

Code, frozen experiment logs and manuscript for *Scanner-Invariant and Institution-Invariant Deep Learning for Reliable Medical Image Diagnosis Across Heterogeneous Clinical Environments* (paper: `paper/main.pdf`).

## What this project found (all numbers are produced by `scripts/final_report.py` from `runs/`)

Setting: open multi-source chest-radiograph benchmark (924 images, 10 sources, COVID-19 vs other), small CNN trained from scratch at 64x64 on CPU, eight held-out sites (4 development, 4 confirmation with frozen configs), two seeds.

1. **Negative accuracy result.** None of nine methods (ERM, GroupDRO, DANN, generic augmentation, site-profile augmentation, profile-adversarial invariance, and three SIDL combinations) is significantly better than ERM at held-out-site AUROC (hierarchical paired bootstrap; all 95% intervals include 0; largest gains GroupDRO +0.015, SIDL-phys +0.011).
2. **Label-free leak diagnostic.** Five acquisition statistics predict the diagnosis on held-out sites with AUROC 0.66.
3. **Conformal prediction for unseen sites.** Seen-site validation overstates held-out AUROC by 0.118; pooled split-conformal sets cover 0.836 on average at a 0.90 target (4 of 8 sites more than 5 points short). A leave-site-out CV+ rule (sites as folds) leaves no site more than 5 points short at ~26% larger sets; all calibrations lie on one coverage-size curve.
4. **Simulation realism.** Controlled simulated acquisition shift does not reproduce the real under-coverage.

The title promises more than the evidence supports: the study covers one modality and one task and finds no benefit from the invariance recipe. See `paper/main.tex` Section "Discussion and Limitations".

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
| `scripts/final_report.py` | regenerates every table, figure and number macro in the paper |
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
```

## Protocol guarantees

- The held-out site is never used for training, validation, model selection or conformal calibration.
- Development folds (Actualmed, Italy, China, Spain) were used for all design decisions; confirmation folds (UK, Australia, Other, Unknown) were run once with frozen configs. Early partial iteration-1 outputs on two confirmation sites are excluded from every analysis.
- Statistics: hierarchical paired bootstrap (resample sites, then test images stratified by class).

## Honest notes

- The iteration-1 *augmentation* numbers (e.g. generic augmentation 0.675 AUROC) came from an earlier whole-batch augmentation implementation that was replaced in iteration 2; they are logged but not reproducible with the final code. The final tables use iteration 1 only for the augmentation-free baselines (ERM, GroupDRO, DANN).
- Single modality; small, shortcut-prone, retrospective data; from-scratch 64x64 CNN; two seeds; no pretrained weights; no private data used.
- Leave-site-out conformal rules have no finite-sample guarantee under site shift; coverage is measured empirically.
- Novelty statements in the paper are hedged: related work was searched, not systematically reviewed. Worst-site conformal calibration is an existing idea.
