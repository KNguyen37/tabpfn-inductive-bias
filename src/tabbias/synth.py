"""Synthetic 2D problems.

Decisions that differ from the pilot, and why:
- Contexts are nested: the n=10 context is a prefix of n=30, and so on. The interface
  slider therefore shows points being added to one set, not four unrelated samples.
- Every prefix is class-balanced, so a size-10 context always has both classes.
- Clean, rotated and noisy variants share one draw, which allows paired comparisons.
- Label noise flips context labels only. Test labels stay clean.
- All context sizes share one scaler and one square grid, which keeps slider frames
  comparable and keeps grid spacing equal on both axes.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay
from sklearn.datasets import make_circles, make_moons

from .contract import FAR_FACTOR, FAR_RES, GRID_PAD, GRID_RES, SYNTH_DATASETS, SYNTH_N, SYNTH_VARIANTS

POOL_PER_SPLIT = max(SYNTH_N)


def _rng(seed: int, *labels: str) -> np.random.Generator:
    return np.random.default_rng([seed, *(zlib.crc32(label.encode()) for label in labels)])


def _spirals(n: int, rng: np.random.Generator, turns_pi: float, noise: float = 0.25):
    half = n // 2
    t = np.sqrt(rng.uniform(0, 1, half)) * turns_pi * np.pi
    arm = np.c_[t * np.cos(t), t * np.sin(t)]
    X = np.r_[arm, -arm] + rng.normal(0, noise, (2 * half, 2))
    y = np.r_[np.zeros(half), np.ones(half)].astype(int)
    return X, y


def _xor(n: int, rng: np.random.Generator, noise: float = 0.15):
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X + rng.normal(0, noise, X.shape), y


def _stripes(n: int, rng: np.random.Generator, periods: float = 3.0, noise: float = 0.02):
    X = rng.uniform(-1, 1, (n, 2))
    y = (np.sin(np.pi * periods * X[:, 0]) > 0).astype(int)
    return X + rng.normal(0, noise, X.shape), y


def _checker(n: int, rng: np.random.Generator, cells: int = 6, noise: float = 0.02):
    X = rng.uniform(-1, 1, (n, 2))
    index = np.floor((X + 1) / 2 * cells).astype(int).clip(0, cells - 1)
    y = ((index[:, 0] + index[:, 1]) % 2).astype(int)
    return X + rng.normal(0, noise, X.shape), y


def _generate(name: str, n: int, seed: int):
    rng = _rng(seed, "data", name)
    sk_seed = int(rng.integers(2**31 - 1))
    if name == "moons":
        return make_moons(n, noise=0.25, random_state=sk_seed)
    if name == "circles":
        return make_circles(n, noise=0.12, factor=0.45, random_state=sk_seed)
    if name == "xor":
        return _xor(n, rng)
    if name == "spirals_loose":
        return _spirals(n, rng, turns_pi=2.0)
    if name == "spirals_medium":
        return _spirals(n, rng, turns_pi=3.0)
    if name == "spirals_tight":
        return _spirals(n, rng, turns_pi=4.5)
    if name == "stripes":
        return _stripes(n, rng)
    if name == "checker_fine":
        return _checker(n, rng)
    raise ValueError(f"unknown synthetic dataset: {name}")


def _balanced_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation that alternates classes, so that every even prefix is balanced."""
    zero = rng.permutation(np.flatnonzero(y == 0))
    one = rng.permutation(np.flatnonzero(y == 1))
    size = min(len(zero), len(one))
    paired = np.c_[zero[:size], one[:size]].ravel()
    return np.r_[paired, zero[size:], one[size:]].astype(int)


@dataclass(frozen=True)
class Problem:
    X_context: np.ndarray  # nested: use X_context[:n]
    y_context: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    half_width: float

    def context(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return self.X_context[:n], self.y_context[:n]

    def grid_points(self) -> np.ndarray:
        axis = np.linspace(-self.half_width, self.half_width, GRID_RES)
        gx, gy = np.meshgrid(axis, axis)
        return np.c_[gx.ravel(), gy.ravel()]

    def far_field_points(self, n: int) -> np.ndarray:
        """Wide-grid points outside the convex hull of the size-n context."""
        axis = np.linspace(-FAR_FACTOR * self.half_width, FAR_FACTOR * self.half_width, FAR_RES)
        gx, gy = np.meshgrid(axis, axis)
        points = np.c_[gx.ravel(), gy.ravel()]
        return points[Delaunay(self.X_context[:n]).find_simplex(points) < 0]


def make_problem(name: str, variant: str, seed: int) -> Problem:
    if variant not in SYNTH_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    X, y = _generate(name, 2 * POOL_PER_SPLIT, seed)
    X, y = np.asarray(X, float), np.asarray(y, int)

    split_rng = _rng(seed, "split", name)
    order = _balanced_order(y, split_rng)
    context_idx, test_idx = order[: POOL_PER_SPLIT], order[POOL_PER_SPLIT:]

    if variant == "rot45":
        angle = np.deg2rad(45)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        X = X @ rotation.T

    mean, std = X[context_idx].mean(axis=0), X[context_idx].std(axis=0)
    X = (X - mean) / std

    y_context = y[context_idx].copy()
    if variant.startswith("noise"):
        rate = int(variant.removeprefix("noise")) / 100
        flips = _rng(seed, "noise", name, variant).random(len(y_context)) < rate
        y_context[flips] = 1 - y_context[flips]

    half_width = float(np.abs(X).max() + GRID_PAD)
    return Problem(X[context_idx], y_context, X[test_idx], y[test_idx], half_width)


__all__ = ["Problem", "make_problem", "SYNTH_DATASETS"]
