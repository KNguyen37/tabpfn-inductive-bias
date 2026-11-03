# Project overview

The submitted proposal is `docs/proposal/Team16_proposal.pdf`. Where it conflicts with this file or
`DECISIONS.md`, these files win.

## Idea
Most models have a designed bias. TabPFN's is learned from its synthetic training prior, so it must be
measured from outside: fit on a small dataset, predict on a dense grid, and study the surface.
McCarter's "What Exactly Has TabPFN Learned to Do?" did this qualitatively for v1 and v2.
We add metrics over seeds, classic baselines, TabPFN-3.5, real-data tests, and an interactive explainer.

## Research questions
- **RQ1:** How do TabPFN's boundaries and uncertainty differ, measurably, from classic models on identical data?
- **RQ2:** Does its bias appear to shift with the data? Reported as a multi-dataset case study, not a general law.
- **RQ3:** Did TabPFN-3.5's scale, encoding and prior change its bias relative to v2?

## Hypotheses (draft; frozen by Nov 10, before TabPFN sweep results are read)
| | Hypothesis | Decided with |
|---|---|---|
| H1 | TabPFN's surfaces are sharper than GP and SVM but not axis-aligned like trees | jump share; axis alignment |
| H2 | TabPFN under-fits high-frequency targets (tight spirals, fine checkerboards) | accuracy, log loss |
| H3 | TabPFN is humbler than trees far from data, but less so than a GP at tiny n | far-field confidence |
| H4 | TabPFN resembles smooth models on smooth-favoring data and trees on tree-favoring data | resemblance D vs. Δ across datasets |
| H5 | 3.5 is sharper and more rotation-sensitive than v2 | jump share; axis alignment; accuracy drop under rotation |
| H6 | 3.5 captures periodic stripes that v2 misses | accuracy on stripes |

## Decision rule
For each hypothesis, take the paired per-seed difference over 10 seeds and report the mean, 95% CI and effect size.
- **Supported or refuted:** the CI excludes zero and clears the margin.
- **Null:** the CI lies entirely inside the margin.
- **Inconclusive:** anything else.

Margins are set by the Nov 10 freeze and recorded in `DECISIONS.md`. Seeds generalize over sampling within a dataset,
not across datasets, so cross-dataset claims stay descriptive.

## Known validity risks
- Accurate models must agree with each other, so resemblance D is weak where accuracy saturates (Banknote).
  Primary real-data readings use n=100 and n=500, and D should also be reported on points where models disagree.
- At n=1000, Banknote splits use 1,300 of 1,372 rows, so seeds are far from independent.
- 2D probes may not reflect higher dimensions, so RQ2 uses all real features.
- Grid-based metrics are rechecked at a second resolution.

## Cut order if time runs short
1. Ensemble-size ablation
2. Noise-column test
3. H6
4. Back to the 2-dataset case study
5. Story mode
