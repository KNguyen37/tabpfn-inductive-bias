"""The data contract shared by experiments, analysis and the interface.

docs/DATA_CONTRACT.md explains this file. Changing anything here affects every
teammate, so it needs a PR that both people have seen.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1

SEEDS = tuple(range(10))
SYNTH_DATASETS = (
    "moons",
    "circles",
    "xor",
    "spirals_loose",
    "spirals_medium",
    "spirals_tight",
    "stripes",
    "checker_fine",
)
SYNTH_VARIANTS = ("clean", "rot45", "noise10", "noise20")
SYNTH_N = (10, 30, 60, 200)

GRID_RES = 200  # prediction grid per axis
FAR_RES = 120  # far-field grid per axis
GRID_PAD = 0.6  # padding past the data, in standardized units
FAR_FACTOR = 3.0  # far-field half-width relative to the main grid
INTERFACE_RES = 100  # grid per axis shipped to the browser

METRIC_COLUMNS = (
    "dataset",
    "variant",
    "n",
    "seed",
    "model",
    "accuracy",
    "log_loss",
    "ece",
    "far_field_confidence",
    "jaggedness",
    "axis_alignment",
    "jump_share",
    "transition_width",
    "fit_seconds",
    "predict_seconds",
    "device",
)

_SLUG = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class RunKey:
    """Identifies one model fit: one dataset, variant, context size, seed and model."""

    dataset: str
    variant: str
    n: int
    seed: int
    model: str

    def __post_init__(self):
        for field in (self.dataset, self.variant, self.model):
            if not _SLUG.match(field):
                raise ValueError(f"not a valid slug: {field!r}")
        if self.n <= 0 or self.seed < 0:
            raise ValueError(f"invalid n or seed: n={self.n}, seed={self.seed}")

    @property
    def relpath(self) -> Path:
        return Path(self.dataset, self.variant, f"n{self.n}", f"seed{self.seed}", self.model)

    def grid_path(self, root: Path) -> Path:
        return Path(root) / "grids" / self.relpath.with_suffix(".npz")

    def run_path(self, root: Path) -> Path:
        return Path(root) / "runs" / self.relpath.with_suffix(".json")


def _replace_atomically(path: Path, write) -> None:
    """Write through a temp file so an interrupted Colab session cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_grid(path: Path, p: np.ndarray, half_width: float) -> None:
    """Store P(class 1) on a square grid spanning [-half_width, half_width] on both axes.

    Rows follow y and columns follow x, both ascending.
    """
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"grid must be square, got {p.shape}")
    values = p.astype(np.float32)
    _replace_atomically(
        Path(path),
        lambda h: np.savez_compressed(
            h, p=values, half_width=np.float32(half_width), contract_version=CONTRACT_VERSION
        ),
    )


def load_grid(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path) as data:
        return data["p"], float(data["half_width"])


def save_run(path: Path, key: RunKey, metrics: dict, extra: dict | None = None) -> None:
    """Write the run record. It is written last, so its existence means the run finished."""
    missing = set(METRIC_COLUMNS) - set(metrics) - set(asdict(key))
    if missing:
        raise ValueError(f"run record is missing columns: {sorted(missing)}")
    record = {**asdict(key), **metrics, "contract_version": CONTRACT_VERSION, "extra": extra or {}}
    payload = json.dumps(record, indent=1, sort_keys=True).encode()
    _replace_atomically(Path(path), lambda h: h.write(payload))
