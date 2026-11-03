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
- **Repo visibility:** public, so instructors can reach it without being added individually.
