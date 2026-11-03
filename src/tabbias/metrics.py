"""Shape and calibration metrics computed from prediction grids.

The proposal's "step score" S measured gradient direction, not sharpness. It is kept
here under its accurate name, `axis_alignment`, and `jump_share` measures sharpness.
tests/test_metrics.py checks both on surfaces of known shape.

Grids must be square with equal spacing on both axes, or the magnitudes below would
depend on orientation.
"""
from __future__ import annotations

import numpy as np

REF_RES = 200  # thresholds below are stated at this grid resolution
CHANGE_EPS = 1e-3
# Draft value, chosen on classic-model surfaces only. Averaged models such as random
# forests make many small jumps, so the score depends on tau: always report sensitivity.
# See results/metric_sanity/jump_share_threshold_sensitivity.csv and docs/DECISIONS.md.
JUMP_TAU = 0.05


def _differences(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.diff(p, axis=1)[:-1, :]
    dy = np.diff(p, axis=0)[:, :-1]
    return dx, dy


def _per_cell_scale(p: np.ndarray) -> float:
    """Multiplier converting a per-cell change at this resolution to the reference resolution."""
    return (p.shape[0] - 1) / (REF_RES - 1)


def axis_alignment(p: np.ndarray) -> float:
    """How axis-aligned the local gradient is: 1 on-axis, 1/sqrt(2) on the diagonal.

    This is the proposal's S. It ignores steepness: a gentle ramp along an axis scores 1.
    """
    dx, dy = _differences(p)
    l1 = np.abs(dx) + np.abs(dy)
    l2 = np.hypot(dx, dy)
    changing = l1 > CHANGE_EPS / _per_cell_scale(p)
    return float(l2[changing].sum() / l1[changing].sum()) if changing.any() else float("nan")


def jump_share(p: np.ndarray, tau: float = JUMP_TAU) -> float:
    """Fraction of all probability change carried by steep cells, from 0 (ramp) to 1 (step).

    A cell is steep if its change exceeds `tau` per cell at the reference resolution.
    The threshold scales with resolution, so resolved surfaces score about the same on
    finer grids. Using the gradient magnitude keeps the score stable under rotation.
    """
    dx, dy = _differences(p)
    magnitude = np.hypot(dx, dy)
    total = magnitude.sum()
    if total == 0:
        return float("nan")
    threshold = tau / _per_cell_scale(p)
    return float(magnitude[magnitude > threshold].sum() / total)


def transition_width(p: np.ndarray, half_width: float, low: float = 0.1, high: float = 0.9) -> float:
    """Mean width, in data units, of the band where low < p < high.

    Secondary diagnostic only: plateaus inside the band (common in random forests) inflate it.
    Returns 0 when no cell falls inside the band.
    """
    dx, dy = _differences(p)
    spacing = 2 * half_width / (p.shape[0] - 1)
    in_band = (p[:-1, :-1] > low) & (p[:-1, :-1] < high)
    variation = np.hypot(dx, dy)[in_band].sum()
    return float((high - low) * in_band.sum() * spacing / variation) if variation > 0 else 0.0


def jaggedness(p: np.ndarray) -> float:
    """Mean change per cell, times 100, at the reference resolution.

    Replaces the proposal's J, whose L1 form changed with rotation and whose
    pilot code multiplied by 100 while Eq. 1 did not.
    """
    dx, dy = _differences(p)
    return float(100 * np.hypot(dx, dy).mean() * _per_cell_scale(p))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    confidence = np.maximum(p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    error = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1.0 else confidence < high)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def far_field_confidence(p_far: np.ndarray) -> float:
    """Mean maximum-class probability on grid points outside the context's convex hull."""
    return float(np.maximum(p_far, 1 - p_far).mean()) if p_far.size else float("nan")
