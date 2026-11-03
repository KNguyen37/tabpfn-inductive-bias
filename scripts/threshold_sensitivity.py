#!/usr/bin/env python
"""Jump share of each classic model at several thresholds, averaged over stored grids.

Run after the pilot rerun in the README. Writes results/metric_sanity/jump_share_threshold_sensitivity.csv.
"""
from pathlib import Path

import pandas as pd

from tabbias.contract import load_grid
from tabbias.metrics import jump_share

TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
ORDER = ["logreg", "svm", "gp", "mlp", "rf", "gbdt", "knn"]


def main():
    rows = []
    for path in Path("results/grids").rglob("n60/seed*/*.npz"):
        grid, _ = load_grid(path)
        rows += [{"model": path.stem, "tau": tau, "jump_share": jump_share(grid, tau=tau)} for tau in TAUS]
    table = pd.DataFrame(rows).pivot_table(index="model", columns="tau", values="jump_share").round(3).loc[ORDER]
    table.to_csv("results/metric_sanity/jump_share_threshold_sensitivity.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
