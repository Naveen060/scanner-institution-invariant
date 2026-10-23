# Related-work notes

Search dates: 2026-10-23. Entries marked VERIFIED had their abstract page fetched and title/authors checked; the rest are cited from memory of well-known work and were confirmed to exist by a title search. The paper's novelty statements are hedged because this was a targeted search, not a systematic review.

## Directly relevant, VERIFIED
- Maguolo & Nanni, "A critic evaluation of methods for COVID-19 automatic detection from X-ray images" (arXiv 2004.12823; Information Fusion 2021): masking the lung region barely changes accuracy; source dataset identifiable. -> our profile-leak diagnostic extends this idea; not new.
- Tartaglione et al., "Unveiling COVID-19 from chest X-ray with deep learning: a hurdles race with small data" (arXiv 2004.05405; IJERPH 2020).
- Zou & Liu, "Coverage-guaranteed prediction sets for out-of-distribution data" (AAAI 2024; arXiv 2403.19950): worst-case quantile over an f-divergence ball around source domains. -> our worst-site mode is a simple variant; not new.
- Tsibulsky, Nevo, Shalit, "Set valued predictions for robust domain generalization" (ICML 2025; arXiv 2507.03146): SET-COVER; "Robust Conformal" per-domain calibration as baseline.
- Liu, Levis, Normand, Han, "Multi-source conformal inference under distribution shift" (ICML 2024; arXiv 2405.09331): target-site prediction intervals from source sites with density ratios; no leave-one-source-out.
- Mehrtens, Bucher, Brinker, "Pitfalls of conformal predictions for medical image classification" (arXiv 2506.18162).
- Liu et al., "Adversarial unsupervised domain adaptation with conditional and label shift: Infer, align and iterate" (ICCV 2021; arXiv 2107.13469): UDA with target data; we study generalization without target data.
- Barber, Candes, Ramdas, Tibshirani, "Conformal prediction beyond exchangeability" (arXiv 2202.13415; Ann. Stat. 2023).
- Existence confirmed by search: Pooch et al. 2020 (arXiv 1909.01940), Cohen et al. 2020 (arXiv 2006.11988), Roberts et al. 2021 (Nat Mach Intell), Podkopaev & Ramdas 2021 (UAI), Saerens et al. 2002 (Neural Comput.), Barber et al. 2021 jackknife+ (arXiv 1905.02928).

## Not verified (not cited in the paper)
- arXiv 2103.11163 (empirical framework for domain generalization in clinical settings), 2103.12595 (augmentation mimicking multi-scanner MRI variability), 2308.04395, MIDL 2023 P054 (acquisition-parameter prediction from breast MRI), PMLR v329 boman26a, ICML 2026 missingness-aware conformal prediction. These could change novelty assessment of site-profile augmentation and cross-site conformal; read before any stronger claim.

## Positioning
Each ingredient has prior work. The defensible contributions are the frozen leave-site-out evaluation with its negative accuracy result, the label-free leak/reliance diagnostics, the quantified validation optimism, and the empirical comparison of conformal calibrations on real held-out sites (including the site-fold CV+ rule and the finding that all calibrations share one coverage-size curve).
