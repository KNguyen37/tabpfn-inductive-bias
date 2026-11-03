#!/usr/bin/env python
"""Score surfaces of known shape with each metric (response to feedback item 1).

Writes results/metric_sanity/known_surfaces.csv and a figure of the table's key contrast.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabbias.metrics import axis_alignment, jump_share, transition_width

RES, HALF = 200, 3.0
OUT = Path("results/metric_sanity")


def coordinates(angle_deg, res=RES):
    axis = np.linspace(-HALF, HALF, res)
    x, y = np.meshgrid(axis, axis)
    angle = np.deg2rad(angle_deg)
    return x * np.cos(angle) + y * np.sin(angle), x, y


def ramp(k, angle, res=RES):
    return 1 / (1 + np.exp(-np.clip(k * coordinates(angle, res)[0], -40, 40)))


def circle(k, res=RES):
    _, x, y = coordinates(0, res)
    return 1 / (1 + np.exp(-np.clip(k * (np.hypot(x, y) - 1.5), -40, 40)))


def plateaus(angle, res=RES):
    return np.array([0.0, 0.3, 0.6, 1.0])[np.digitize(coordinates(angle, res)[0], [-1, 0, 1])]


SURFACES = {
    **{(f"smooth ramp, {a} deg", "smooth"): ramp(1, a) for a in (0, 22.5, 45, 67.5)},
    **{(f"sharp edge, {a} deg", "sharp"): ramp(60, a) for a in (0, 22.5, 45, 67.5)},
    ("smooth circle", "smooth"): circle(1),
    ("sharp circle", "sharp"): circle(60),
    **{(f"tree-like plateaus, {a} deg", "sharp"): plateaus(a) for a in (0, 45)},
}


def main():
    rows = [
        {
            "surface": name,
            "true_shape": shape,
            "axis_alignment_formerly_S": axis_alignment(p),
            "jump_share": jump_share(p),
            "transition_width": transition_width(p, HALF),
        }
        for (name, shape), p in SURFACES.items()
    ]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(OUT / "known_surfaces.csv", index=False)
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    colors = df.true_shape.map({"smooth": "#2a78d6", "sharp": "#d8413a"})
    order = np.arange(len(df))[::-1]
    for ax, column, title in zip(
        axes, ("axis_alignment_formerly_S", "jump_share"), ("S: does not follow true shape", "Jump share: follows true shape")
    ):
        ax.scatter(df[column], order, c=colors, s=28)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel(column.replace("_", " "))
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(order, df.surface, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "known_surfaces.png", dpi=160)


if __name__ == "__main__":
    main()
