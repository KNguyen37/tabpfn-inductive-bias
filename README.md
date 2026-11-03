# What Has TabPFN's Prior Learned?

Measuring and visualizing the inductive bias of a tabular foundation model.
Course project for DS-GA 3001: Visualization for Machine Learning, NYU, by Kiet Nguyen and Nhan Doan.

**Status: in progress.** The paper, TabPFN sweeps and interface are not finished yet.

TabPFN is a transformer pretrained on synthetic tables, so its inductive bias is learned, not designed.
We fit it to small problems, predict across a dense 2D grid, and compare the resulting surfaces with
seven classic models and across TabPFN versions.

## Setup
```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

## Reproduce current results
| Output | Command | Compute |
|---|---|---|
| `results/metric_sanity/known_surfaces.csv` | `.venv/bin/python scripts/metric_sanity.py` | Laptop |
| `results/metrics/classic_pilot_rerun.csv` | `.venv/bin/python scripts/run_sweep.py --family classic --datasets moons,circles,xor,spirals_medium --variants clean --ns 60 --seeds 0,1,2 --csv results/metrics/classic_pilot_rerun.csv` | Laptop |
| `results/metric_sanity/jump_share_threshold_sensitivity.csv` | `.venv/bin/python scripts/threshold_sensitivity.py` (after the row above) | Laptop |
| `results/metrics/classic_synth.csv` | `.venv/bin/python scripts/run_sweep.py --family classic` | Laptop |
| `results/metrics/real_screen.csv`, `real_screen_runs.csv` | `.venv/bin/python scripts/screen_datasets.py` (downloads from OpenML into `data/`) | Laptop |
| TabPFN access and timing | `notebooks/colab_smoke_test.ipynb` | Colab GPU |
| `results/metrics/tabpfn_synth.csv` | `notebooks/colab_sweep.ipynb` | Colab GPU |

Sweeps are resumable: run the same command again to continue. TabPFN v2.5 and later require accepting the
Prior Labs license and setting `TABPFN_TOKEN`. Weights are licensed separately from this code.

## Repository guide
- `docs/PROJECT.md`: research questions and hypotheses
- `docs/DECISIONS.md`: how we responded to proposal feedback, and other decisions
- `docs/DATA_CONTRACT.md`: data formats
- `docs/PLAYBOOK.md`: team and agent workflow
- `pilot/`: the proposal's pilot code, kept as submitted
