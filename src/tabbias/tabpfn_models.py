"""TabPFN adapters. These only run on Colab, by team agreement (docs/PLAYBOOK.md)."""
from __future__ import annotations

import os

# Slug -> (ModelVersion attribute, n_estimators override or None for the default).
TABPFN_MODELS = {
    "tabpfn_v2": ("V2", None),
    "tabpfn_v2_e1": ("V2", 1),
    "tabpfn_v2_e4": ("V2", 4),
    "tabpfn_v2_e8": ("V2", 8),
    "tabpfn_v35": ("V3_5", None),
    "tabpfn_v35_e1": ("V3_5", 1),
    "tabpfn_v35_e4": ("V3_5", 4),
    "tabpfn_v35_e8": ("V3_5", 8),
    "tabpfn_v35_fast": ("V3_5_FAST", None),
}


def require_colab_gpu() -> None:
    """Refuse to run TabPFN anywhere except an explicitly marked Colab GPU session."""
    if os.environ.get("TABBIAS_COMPUTE") != "colab":
        raise RuntimeError(
            "TabPFN runs only on Kiet's Colab session. Use notebooks/colab_sweep.ipynb. "
            "Local runs should use --family classic."
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > GPU.")


def make_tabpfn(name: str, seed: int):
    require_colab_gpu()
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version, n_estimators = TABPFN_MODELS[name]
    overrides = {"random_state": seed, "device": "cuda"}
    if n_estimators is not None:
        overrides["n_estimators"] = n_estimators
    return TabPFNClassifier.create_default_for_version(getattr(ModelVersion, version), **overrides)
