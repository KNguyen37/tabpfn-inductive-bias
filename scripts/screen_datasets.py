#!/usr/bin/env python
"""Screen real datasets with classic models only (Issue #5). TabPFN is never involved.

For each candidate, n in {100, 500} and seeds 0-9, fit the seven classic models and record
test accuracy and log loss. Delta = trees (RF, GBDT) minus smooth models (SVM, MLP, GP).

    python scripts/screen_datasets.py                       # every candidate
    python scripts/screen_datasets.py --datasets banknote --seeds 0

Resumable: finished (dataset, n, seed) results are cached in data/screen/ and skipped.
Writes results/metrics/real_screen_runs.csv (one row per model fit) and
results/metrics/real_screen.csv (one row per dataset and n, plus every skipped candidate).
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from tabbias.classic import CLASSIC_MODELS, make_classic
from tabbias.contract import SEEDS
from tabbias.realdata import CANDIDATES, EXCLUDED, SCREEN_N, TEST_SIZE, load, make_split, summarize

DATA = Path("data")
RUNS_CSV = Path("results/metrics/real_screen_runs.csv")
SUMMARY_CSV = Path("results/metrics/real_screen.csv")


def use_certifi():
    """The python.org build on macOS ships without CA certificates; certifi supplies them."""
    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def fit_all(X_tr, y_tr, X_te, y_te, seed):
    rows = []
    for model in CLASSIC_MODELS:
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = make_classic(model, seed).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        rows.append({
            "model": model,
            "accuracy": float(((p >= 0.5).astype(int) == y_te).mean()),
            "log_loss": float(log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "fit_seconds": time.perf_counter() - started,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=lambda s: s.split(","), default=list(CANDIDATES))
    parser.add_argument("--ns", type=lambda s: [int(x) for x in s.split(",")], default=list(SCREEN_N))
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    args = parser.parse_args()
    use_certifi()

    cache, status = DATA / "screen", []
    cache.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        candidate = CANDIDATES[name]
        try:
            X, y = load(name, DATA / "openml")
        except Exception as error:  # a failed download is recorded, not dropped
            status.append({"dataset": name, "n": None, "status": f"load failed: {type(error).__name__}: {error}"})
            print(f"{name}: LOAD FAILED {error}", flush=True)
            continue
        for n in args.ns:
            if len(y) < n + TEST_SIZE:
                status.append({"dataset": name, "n": n, "status": f"skipped: {len(y)} rows < n + {TEST_SIZE}"})
                continue
            started = time.perf_counter()
            for seed in args.seeds:
                path = cache / f"{name}_n{n}_seed{seed}.json"
                if path.exists():
                    continue
                rows = fit_all(*make_split(X, y, n, seed), seed)
                path.write_text(json.dumps([{"dataset": name, "n": n, "seed": seed, **r} for r in rows]))
            status.append({"dataset": name, "n": n, "status": "screened"})
            print(f"{name} n={n}: {time.perf_counter() - started:.1f}s", flush=True)

    columns = ["dataset", "n", "seed", "model", "accuracy", "log_loss", "fit_seconds"]
    runs = pd.DataFrame([r for p in sorted(cache.glob("*.json")) for r in json.loads(p.read_text())], columns=columns)
    runs = runs[runs.dataset.isin(args.datasets) & runs.n.isin(args.ns) & runs.seed.isin(args.seeds)]
    runs = runs.sort_values(["dataset", "n", "seed", "model"])
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    runs.round(6).to_csv(RUNS_CSV, index=False)

    summary = summarize(runs) if len(runs) else pd.DataFrame(columns=["dataset", "n"])
    status = pd.DataFrame(status)
    status = pd.concat([status, pd.DataFrame(
        [{"dataset": k, "n": None, "status": f"excluded: {why}"} for k, (_, why) in EXCLUDED.items()])])
    meta = pd.DataFrame([{"dataset": k, "openml_id": c.openml_id, "rows": c.rows,
                          "numeric_features": c.numeric_features, "domain": c.domain, "source": c.source,
                          "origin": c.origin}
                         for k, c in CANDIDATES.items()]
                        + [{"dataset": k, "openml_id": i} for k, (i, _) in EXCLUDED.items()])
    table = status.merge(summary, on=["dataset", "n"], how="left").merge(meta, on="dataset", how="left")
    table = table.sort_values(["dataset", "n"], na_position="last")
    table.round(6).to_csv(SUMMARY_CSV, index=False)
    print(table[table.status == "screened"][
        ["dataset", "n", "delta_accuracy", "delta_accuracy_ci_low", "delta_accuracy_ci_high",
         "delta_log_loss", "best_model_accuracy", "saturated_at_n100"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
