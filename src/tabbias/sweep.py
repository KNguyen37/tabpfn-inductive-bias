"""The resumable experiment loop, shared by classic models and TabPFN.

A run counts as done only when its JSON record exists, and that record is written
last. After a disconnect, the same command skips finished work and continues.
"""
from __future__ import annotations

import csv
import json
import platform
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from sklearn.metrics import log_loss

from . import metrics as M
from .contract import GRID_RES, METRIC_COLUMNS, RunKey, save_grid, save_run
from .synth import Problem, make_problem


def predict_batched(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    parts = [model.predict_proba(X[i : i + batch_size])[:, 1] for i in range(0, len(X), batch_size)]
    return np.concatenate(parts) if parts else np.empty(0)


def describe_model(model) -> dict:
    """Settings resolved during fit. TabPFN's default ensemble size depends on the checkpoint and features."""
    info = {}
    if hasattr(model, "n_estimators_"):
        info["n_estimators"] = int(model.n_estimators_)
    if hasattr(model, "model_path"):
        info["model_path"] = str(model.model_path)
    return info


def is_done(key: RunKey, out: Path) -> bool:
    """Both files are required, because Google Drive can lose unflushed writes after a disconnect."""
    return key.run_path(out).exists() and key.grid_path(out).exists()


def run_one(key: RunKey, problem: Problem, factory: Callable, batch_size: int) -> tuple[dict, np.ndarray]:
    """Fit one model and return its metrics (plus `model_info`) and prediction grid."""
    X, y = problem.context(key.n)
    model = factory(key.model, key.seed)

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        grid = predict_batched(model, problem.grid_points(), batch_size).reshape(GRID_RES, GRID_RES)
        predict_seconds = time.perf_counter() - started

        p_test = predict_batched(model, problem.X_test, batch_size)
        p_far = predict_batched(model, problem.far_field_points(key.n), batch_size)

    clipped = np.clip(p_test, 1e-6, 1 - 1e-6)
    values = {
        "accuracy": float(((p_test >= 0.5).astype(int) == problem.y_test).mean()),
        "log_loss": float(log_loss(problem.y_test, clipped, labels=[0, 1])),
        "ece": M.expected_calibration_error(p_test, problem.y_test),
        "far_field_confidence": M.far_field_confidence(p_far),
        "jaggedness": M.jaggedness(grid),
        "axis_alignment": M.axis_alignment(grid),
        "jump_share": M.jump_share(grid),
        "transition_width": M.transition_width(grid, problem.half_width),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "model_info": describe_model(model),
    }
    return values, grid


def enumerate_keys(datasets, variants, ns, seeds, models) -> list[RunKey]:
    return [
        RunKey(d, v, n, s, m)
        for d in datasets
        for v in variants
        for s in seeds
        for n in ns
        for m in models
    ]


def run_sweep(
    keys: Iterable[RunKey],
    factory: Callable,
    out: Path,
    device: str = "cpu",
    batch_size: int = 8192,
    max_seconds: float | None = None,
) -> dict:
    """Run every unfinished key. Failures are logged and skipped, so one cannot stop the sweep."""
    out = Path(out)
    keys = list(keys)
    pending = [k for k in keys if not is_done(k, out)]
    tally = {"total": len(keys), "skipped": len(keys) - len(pending), "done": 0, "failed": 0}
    started = time.perf_counter()
    cache_key, problem = None, None

    for index, key in enumerate(pending, 1):
        if max_seconds is not None and time.perf_counter() - started > max_seconds:
            tally["stopped_early"] = True
            break
        try:
            problem_key = (key.dataset, key.variant, key.seed)
            if problem_key != cache_key:
                problem = make_problem(key.dataset, key.variant, key.seed)
                cache_key = problem_key
            values, grid = run_one(key, problem, factory, batch_size)
            model_info = values.pop("model_info")
            save_grid(key.grid_path(out), grid, problem.half_width)
            save_run(key.run_path(out), key, {**values, "device": device},
                     extra={"python": platform.python_version(), **model_info})
            tally["done"] += 1
        except Exception:
            tally["failed"] += 1
            log = out / "failures.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(f"--- {key}\n{traceback.format_exc()}\n")
        if index % 25 == 0 or index == len(pending):
            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(pending)}] {elapsed / index:.2f}s per run, failed={tally['failed']}", flush=True)
    return tally


def aggregate(out: Path, destination: Path) -> int:
    """Merge run records into one CSV, sorted so that reruns produce small git diffs."""
    records = [json.loads(p.read_text()) for p in sorted((Path(out) / "runs").rglob("*.json"))]
    records.sort(key=lambda r: tuple(r[c] for c in ("dataset", "variant", "n", "seed", "model")))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {k: round(v, 6) if isinstance(v, float) else v for k, v in record.items() if k in METRIC_COLUMNS}
            )
    return len(records)
