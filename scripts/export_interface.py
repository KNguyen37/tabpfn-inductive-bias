#!/usr/bin/env python
"""Convert stored grids into small JSON files for the D3 interface.

Each file holds an INTERFACE_RES x INTERFACE_RES grid as base64-encoded uint8 values,
plus the run's metrics. See docs/DATA_CONTRACT.md for the browser-side format.

    python scripts/export_interface.py --dest interface/data/fixture \
        --datasets spirals_medium --variants clean --seeds 0
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from tabbias.contract import INTERFACE_RES, load_grid


def downsample(p: np.ndarray, res: int = INTERFACE_RES) -> np.ndarray:
    """Average over equal blocks; requires the grid size to be a multiple of res."""
    factor, remainder = divmod(p.shape[0], res)
    if remainder:
        raise ValueError(f"grid of {p.shape[0]} is not a multiple of {res}")
    return p.reshape(res, factor, res, factor).mean(axis=(1, 3))


def encode(p: np.ndarray) -> str:
    return base64.b64encode(np.round(np.clip(p, 0, 1) * 255).astype(np.uint8).tobytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("results"))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()

    index, grids_root = [], args.src / "grids"
    for path in sorted(grids_root.rglob("*.npz")):
        dataset, variant, n_part, seed_part = path.relative_to(grids_root).parts[:4]
        n, seed, model = int(n_part[1:]), int(seed_part[4:]), path.stem
        if (args.datasets and dataset not in args.datasets) or (args.variants and variant not in args.variants):
            continue
        if args.seeds and seed not in args.seeds:
            continue
        p, half_width = load_grid(path)
        run = args.src / "runs" / path.relative_to(grids_root).with_suffix(".json")
        record = json.loads(run.read_text()) if run.exists() else {}
        metrics = {k: record.get(k) for k in (
            "accuracy", "log_loss", "ece", "far_field_confidence",
            "jaggedness", "axis_alignment", "jump_share", "transition_width")}
        item = {"dataset": dataset, "variant": variant, "n": n, "seed": seed, "model": model}
        relative = Path(dataset, variant, n_part, seed_part, f"{model}.json")
        payload = {**item, "res": INTERFACE_RES, "half_width": half_width,
                   "p_uint8_b64": encode(downsample(p)), "metrics": metrics}
        destination = args.dest / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload))
        index.append({**item, "path": relative.as_posix()})

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "index.json").write_text(json.dumps(index, indent=0))
    print(f"exported {len(index)} grids to {args.dest}")


if __name__ == "__main__":
    main()
