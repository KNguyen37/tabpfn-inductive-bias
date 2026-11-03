# Decisions

Newest changes go at the bottom of each section. Changing a decision needs both teammates' agreement.

## Response to proposal feedback (Nov 3, 2026)

### 1. Step score S: replaced
S is an L2/L1 ratio of the local gradient, so it measures direction and ignores steepness. On known surfaces, a smooth
ramp and a sharp edge at the same angle get the same score (1.000 at 0°, 0.707 at 45°). Evenly spread directions give
π/4 ≈ 0.785, close to the pilot's 0.78 for smooth models. The pilot showed that trees are axis-aligned, not that they are step-like.

- S is kept and renamed **axis alignment**.
- **Jump share** is the new sharpness metric: the fraction of total probability change carried by steep cells. It is stable under rotation and grid resolution on known surfaces.
- **Transition width** is a secondary diagnostic, because plateaus inflate it.
- Jaggedness J now uses the L2 form.

Evidence: `results/metric_sanity/`, `tests/test_metrics.py`.

**Open until Nov 10:** the jump-share threshold τ.
- Classic-model scores depend on τ. Random forest scores 0.94 at τ=0.01 and 0.36 at τ=0.1, because averaging 300 trees yields many small jumps.
- τ=0.05 is the draft value, chosen on classic models only. Averaged over four datasets, logistic regression, SVM and GP are at or below 0.12, and MLP, random forest, gradient boosting and k-NN are at or above 0.61. The averages hide some spread: SVM reaches 0.42 on circles.
- Report results at several τ values, not just one.
- The threshold is per grid cell, and cell size varies with each dataset's extent. Consider restating it per standardized unit before the freeze.

### 2. Statistical strength: more seeds and a new decision rule
10 seeds replace 3. With 3 seeds, the smallest two-sided p-value from an exact paired sign-flip test is 0.25; with 10 it is about 0.002.
"Inside the seed spread means null" is replaced by paired differences, 95% CIs, effect sizes and a pre-set margin (`PROJECT.md`).
Every stochastic model now receives the seed. The pilot fixed `random_state` at 0.

### 3. Real datasets: both options
About 6 datasets, 3 favoring each side, reported as a multi-dataset case study that makes no general claim about TabPFN.
Each dataset's role is assigned from classic-model Δ alone, before TabPFN sees it, to avoid cherry-picking.
The main figure plots Δ against TabPFN's tree-vs-smooth resemblance, with one point per dataset.
Fallback: the 2-dataset case study. "Rule-like" becomes "tree-favoring", which is what Δ actually measures.

**Screen result and proposed datasets (Nov 3, Nhan; proposed, awaiting Kiet's confirmation; Issue #5).**
Classic models only; TabPFN has not seen any of these datasets. `CANDIDATES` in `src/tabbias/realdata.py` lists 35
OpenML datasets; 33 were screened at n=100 and n=500 over 10 seeds (4,200 fits, 0 failures), and ionosphere and sonar
were too small for either size. 8 more were checked and excluded with a reason. Every candidate, including skipped and
excluded ones, is in `results/metrics/real_screen.csv`; per-fit results are in `results/metrics/real_screen_runs.csv`.
Δ is mean accuracy of RF and GBDT minus that of SVM, MLP and GP, with a 95% t-interval over seeds. Dataset names below
are the keys used in the CSV and code.

| Side | Dataset (OpenML ID) | Rows | Features | Δ acc, n=100 [95% CI] | Δ acc, n=500 [95% CI] |
|---|---|---|---|---|---|
| Tree | covertype (44121) | 566,602 | 10 | +0.048 [+0.032, +0.063] | +0.021 [+0.008, +0.034] |
| Tree | run_or_walk (40922) | 88,588 | 6 | +0.101 [+0.087, +0.115] | +0.036 [+0.030, +0.043] |
| Tree | kdd_ipums_la_97_small (44124) | 5,188 | 20 | +0.052 [+0.032, +0.072] | +0.040 [+0.028, +0.052] |
| Smooth | california (44090) | 20,634 | 8 | −0.019 [−0.034, −0.005] | +0.004 [−0.007, +0.014] |
| Smooth | eeg_eye_state (1471), provisional | 14,980 | 14 | −0.019 [−0.037, −0.001] | −0.031 [−0.069, +0.007] |
| Smooth | kc1 (1067) | 2,109 | 21 | −0.015 [−0.023, −0.006] | −0.009 [−0.017, −0.000] |

Selection rules, set before the screen results were read: real measured data (simulated sets such as Higgs, MiniBooNE
and MagicTelescope are excluded); Δ's CI excludes 0 with the same sign at both n=100 and n=500; best accuracy below 0.98
at n=100; at least 1,300 rows; at most 50 features; distinct domains; then the largest |Δ| per side, preferring more rows.

- **Rule change after seeing the classic screen (disclose in the paper).** Only kc1 meets the both-sizes rule on the
  smooth side, and only barely (n=500 upper bound −0.00004). After reading the classic results, the smooth-side rule was
  relaxed to "Δ's CI excludes 0 at n=100", which admits california and eeg_eye_state. This happened before TabPFN saw
  any real data, so CLAUDE.md rule 5 does not apply, but it is a role rule chosen after seeing classic results.
  Real smooth-favoring data being rare is itself worth reporting.
- **eeg_eye_state is provisional.** Its rows are one continuous recording in time order, so random splits put
  near-identical neighbouring samples in train and test, which favors local models. Its Δ above comes from such random
  splits. Before TabPFN sees it, re-screen it on classic models with contiguous time-block splits and keep it only if
  it still favors smooth models; otherwise use banknote or drop to two smooth datasets.
- **Banknote is saturated** (best accuracy 0.996 at n=100), as the proposal anticipated, so it is not a primary pick.
- **Alternative tree pick:** electricity (44120) has published reference plots but its n=100 CI includes 0
  (+0.013 [−0.003, +0.030]); it could replace kdd_ipums_la_97_small.
- **Not picked:** mozilla4 (largest Δ, but its features include a class `id` and start/end times, which may let trees
  exploit identifiers; unchecked); jannis (contents concealed, 54 features).
- **Accuracy and log loss disagree:** in 25 of 60 screened settings Δ in accuracy favors trees while Δ in log loss
  favors smooth models, so log loss is not a simple second check for roles.

### 4. Classmate study: removed
No timing, accuracy or before/after measures. Replaced by 3–5 structured walkthroughs with people experienced in ML, reported
qualitatively, plus authors' worked examples.

### 5. Visualization: D3 only, with an explicit claim
Streamlit is removed; the interface is D3 only, on GitHub Pages.
The claim: the interface shows what static figures cannot, namely (a) a boundary evolving as context is added, and
(b) direct comparison of two models through `|pA − pB|`. The paper will include a worked example of each.

A static page cannot run TabPFN, so the proposal's "add context points" becomes a slider over precomputed nested contexts (10 ⊂ 30 ⊂ 60 ⊂ 200).

### 6. Model access
**Go for H5, H6 and RQ3.** On Nov 3, `tabpfn` 9.2.0 ran `ModelVersion.V2` and 3.5 on a Colab A100 (40 GB) without errors, predicting a 200×200 grid on spirals at n=60 and n=200 (`results/smoke/tabpfn_smoke.json`). A full job took about 3.2 s for v2 and 4.8 s for 3.5.

- **Compute estimate:** about 4 GPU hours for the planned 3,640 jobs. This is rough: it comes from one synthetic problem, excludes Drive write time and ablation, and real-data jobs will differ in size and feature count. Fallback if 3.5 proves too slow: `tabpfn_v35_fast`.
- **Default ensemble size is not fixed:** `n_estimators="auto"` comes from the checkpoint and can rise with feature count. Record the resolved count per run before the sweep, and settle whether defaults are comparable across versions and in the noise-column test (Issue #8).
- **License:** the 3.5 license permits non-commercial use, including academic research, and says outputs may be used only for non-commercial purposes. It does not explicitly address publishing outputs. It prohibits offering the model as a hosted service, so the interface shows precomputed grids only. It also bars using outputs to train a competing model. This is our reading, not legal advice. Label published 3.5 grids as non-commercial.
- **Disclosure for the freeze:** the smoke report exposed jump share and axis alignment for one TabPFN surface per version and size, on one dataset and seed. Those values must not be used to choose τ or margins, and are not evidence about any hypothesis.

## Other decisions
- **Compute:** TabPFN runs on Kiet's Colab GPU, recording the GPU type. Classic models and analysis run locally.
- **Synthetic design:** 8 datasets, 4 variants, n in {10, 30, 60, 200}, 10 seeds. Checkerboard was added because H2 requires it. Reasons for nesting, shared grids and label noise are in `src/tabbias/synth.py`.
- **TabPFN sweep timing:** the synthetic sweep may run before the Nov 10 freeze, on the condition that nobody reads its metrics or surfaces until the freeze is done. Only run counts, timing and failures may be checked. Grids are saved, so shape metrics can be recomputed after the threshold is final. A change to the synthetic design itself would require rerunning.
- **Repo visibility:** public, so instructors can reach it without being added individually.
