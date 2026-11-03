#!/usr/bin/env python
"""Run a resumable sweep. Rerun the same command to continue after any interruption.

Local, one seed:     python scripts/run_sweep.py --family classic --seeds 0
Local, all seeds:    python scripts/run_sweep.py --family classic
Colab:               see notebooks/colab_sweep.ipynb
"""
import argparse
from pathlib import Path

from tabbias.classic import CLASSIC_MODELS, make_classic
from tabbias.contract import SEEDS, SYNTH_DATASETS, SYNTH_N, SYNTH_VARIANTS
from tabbias.sweep import aggregate, enumerate_keys, run_sweep


def int_list(text):
    return [int(x) for x in text.split(",")]


def str_list(text):
    return text.split(",")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=["classic", "tabpfn"], required=True)
    parser.add_argument("--models", type=str_list, help="comma-separated; defaults depend on the family")
    parser.add_argument("--datasets", type=str_list, default=list(SYNTH_DATASETS))
    parser.add_argument("--variants", type=str_list, default=list(SYNTH_VARIANTS))
    parser.add_argument("--ns", type=int_list, default=list(SYNTH_N))
    parser.add_argument("--seeds", type=int_list, default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--csv", type=Path, help="aggregate CSV; default results/metrics/<family>_synth.csv")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-minutes", type=float, help="stop cleanly after this long")
    args = parser.parse_args()

    if args.family == "classic":
        models, factory, device = args.models or list(CLASSIC_MODELS), make_classic, "cpu"
    else:
        from tabbias.tabpfn_models import make_tabpfn, require_colab_gpu

        require_colab_gpu()
        import torch

        models = args.models or ["tabpfn_v2", "tabpfn_v35"]
        factory, device = make_tabpfn, torch.cuda.get_device_name(0).replace(" ", "_")

    keys = enumerate_keys(args.datasets, args.variants, args.ns, args.seeds, models)
    print(f"{len(keys)} runs requested, writing to {args.out}", flush=True)
    tally = run_sweep(
        keys, factory, args.out, device=device, batch_size=args.batch_size,
        max_seconds=args.max_minutes * 60 if args.max_minutes else None,
    )
    print(tally)

    destination = args.csv or Path("results/metrics") / f"{args.family}_synth.csv"
    print(f"aggregated {aggregate(args.out, destination)} runs into {destination}")
    if tally["failed"]:
        raise SystemExit(f"{tally['failed']} runs failed; see {args.out / 'failures.log'}")


if __name__ == "__main__":
    main()
