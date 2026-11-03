#!/usr/bin/env python
"""Answer the model-access question: do v2 and 3.5 load, run, and how long does a job take?

Runs on Colab through notebooks/colab_smoke_test.ipynb. It writes a JSON report to commit.
"""
import argparse
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from tabbias.contract import SEEDS, SYNTH_DATASETS, SYNTH_N, SYNTH_VARIANTS, RunKey
from tabbias.sweep import run_one
from tabbias.synth import make_problem
from tabbias.tabpfn_models import make_tabpfn, require_colab_gpu

MODELS = ("tabpfn_v2", "tabpfn_v35")
NS = (60, 200)
WARM_REPEATS = 2
REAL_DATA_JOBS = 6 * 3 * len(SEEDS) * 3 * len(MODELS)  # datasets, n, seeds, interventions, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/smoke/tabpfn_smoke.json"))
    args = parser.parse_args()

    require_colab_gpu()
    import torch

    problem = make_problem("spirals_medium", "clean", 0)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "tabpfn_version": version("tabpfn"),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "runs": [],
        "errors": {},
    }
    for model in MODELS:
        for n in NS:
            for repeat in range(1 + WARM_REPEATS):  # the first repeat includes the download
                key = RunKey("spirals_medium", "clean", n, 0, model)
                started = time.perf_counter()
                try:
                    values, _ = run_one(key, problem, make_tabpfn, batch_size=8192)
                except Exception as error:  # report it: a failure here is itself the finding
                    report["errors"][f"{model}/n{n}"] = f"{type(error).__name__}: {error}"
                    print("FAILED", model, n, error, flush=True)
                    break
                row = {
                    "model": model, "n": n, "repeat": repeat, "cold": repeat == 0,
                    "total_seconds": round(time.perf_counter() - started, 2),
                    "fit_seconds": round(values["fit_seconds"], 2),
                    "predict_seconds": round(values["predict_seconds"], 2),
                    "accuracy": round(values["accuracy"], 3),
                    "jump_share": round(values["jump_share"], 3),
                    "axis_alignment": round(values["axis_alignment"], 3),
                }
                report["runs"].append(row)
                print(row, flush=True)

    warm = [r["total_seconds"] for r in report["runs"] if not r["cold"]]
    synthetic = len(SYNTH_DATASETS) * len(SYNTH_VARIANTS) * len(SYNTH_N) * len(SEEDS) * len(MODELS)
    report["jobs"] = {"synthetic": synthetic, "real": REAL_DATA_JOBS}
    if warm:
        per_job = statistics.median(warm)
        report["median_warm_seconds_per_job"] = per_job
        report["estimated_gpu_hours"] = round((synthetic + REAL_DATA_JOBS) * per_job / 3600, 1)
    report["verdict"] = "GO" if warm and not report["errors"] else "NO-GO: see errors"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
