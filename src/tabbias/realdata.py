"""Real OpenML datasets: candidate list, loading, splitting and the tree-vs-smooth gap.

Issue #5 screens candidates with classic models only, before TabPFN sees any of them,
so that each dataset's role (tree-favoring or smooth-favoring) cannot be chosen from
TabPFN's results.

Every ID, row count and numeric-feature count below was checked against the OpenML
API on 2026-11-03 (www.openml.org/api/v1/json/data/<id>), not taken from memory.
`load` asserts the shape again after download.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TREE_MODELS = ("rf", "gbdt")
SMOOTH_MODELS = ("svm", "mlp", "gp")
SCREEN_N = (100, 500)
TEST_SIZE = 300
SATURATION = 0.98  # best mean accuracy at n=100 at or above this is flagged


@dataclass(frozen=True)
class Candidate:
    openml_id: int
    rows: int
    numeric_features: int
    domain: str
    source: str  # where the candidate came from, not its role: roles come from Delta only
    origin: str = "measured"  # "simulated" when the rows come from a scientific simulation


CANDIDATES = {
    # Proposal
    "banknote": Candidate(1462, 1372, 4, "image forensics", "proposal"),
    "electricity": Candidate(44120, 38474, 7, "energy prices", "proposal; Grinsztajn benchmark"),
    # Grinsztajn et al. numeric classification benchmark
    "covertype": Candidate(44121, 566602, 10, "forestry", "Grinsztajn benchmark"),
    "pol": Candidate(44122, 10082, 26, "telecom", "Grinsztajn benchmark"),
    "house_16H": Candidate(44123, 13488, 16, "census", "Grinsztajn benchmark"),
    "MagicTelescope": Candidate(44125, 13376, 10, "astrophysics", "Grinsztajn benchmark", "simulated"),
    "MiniBooNE": Candidate(44128, 72998, 50, "particle physics", "Grinsztajn benchmark", "simulated"),
    "Higgs": Candidate(44129, 940160, 24, "particle physics", "Grinsztajn benchmark", "simulated"),
    "kdd_ipums_la_97_small": Candidate(44124, 5188, 20, "census", "Grinsztajn benchmark"),
    "eye_movements": Candidate(44130, 7608, 20, "psychology", "Grinsztajn benchmark"),
    "jannis": Candidate(44131, 57580, 54, "anonymized", "Grinsztajn benchmark"),
    "california": Candidate(44090, 20634, 8, "housing", "Grinsztajn benchmark"),
    # Other numeric OpenML / UCI datasets
    "phoneme": Candidate(1489, 5404, 5, "speech", "OpenML numeric"),
    "wilt": Candidate(40983, 4839, 5, "remote sensing", "OpenML numeric"),
    "eeg_eye_state": Candidate(1471, 14980, 14, "brain signals", "OpenML numeric"),
    "spambase": Candidate(44, 4601, 57, "email text", "OpenML numeric"),
    "ozone_level": Candidate(1487, 2534, 72, "weather", "OpenML numeric"),
    "pc3": Candidate(1050, 1563, 37, "software defects", "OpenML numeric"),
    "qsar_biodeg": Candidate(1494, 1055, 41, "chemistry", "OpenML numeric"),
    "pc1": Candidate(1068, 1109, 21, "software defects", "OpenML numeric"),
    "diabetes": Candidate(37, 768, 8, "medical", "OpenML numeric"),
    "blood_transfusion": Candidate(1464, 748, 4, "medical", "OpenML numeric"),
    "wdbc": Candidate(1510, 569, 30, "medical", "OpenML numeric"),
    "ilpd": Candidate(1480, 583, 9, "medical", "OpenML numeric"),
    "climate_crashes": Candidate(1467, 540, 20, "climate model", "OpenML numeric", "simulated"),
    "ionosphere": Candidate(59, 351, 34, "radar", "OpenML numeric"),
    "sonar": Candidate(40, 208, 60, "sonar", "OpenML numeric"),
    "kc2": Candidate(1063, 522, 21, "software defects", "OpenML numeric"),
    # Added Nov 3 to widen the range of domains (checked on OpenML the same day)
    "kc1": Candidate(1067, 2109, 21, "software defects", "OpenML numeric"),
    "mozilla4": Candidate(1046, 15545, 5, "software", "OpenML numeric"),
    "delta_ailerons": Candidate(803, 7129, 5, "aircraft control", "OpenML numeric", "simulated"),
    "elevators": Candidate(846, 16599, 18, "aircraft control", "OpenML numeric", "simulated"),
    "run_or_walk": Candidate(40922, 88588, 6, "phone accelerometer", "OpenML numeric"),
    "nomao": Candidate(1486, 34465, 89, "web place data", "OpenML numeric"),
    "churn": Candidate(40701, 5000, 16, "telecom customers", "OpenML numeric"),
}

# Checked on OpenML but not screened, with the reason. Kept so the record shows every candidate.
EXCLUDED = {
    "PhishingWebsites": (4534, "all 30 features are categorical; the screen needs numeric features"),
    "credit": (44089, "finance; left out to keep the domains varied (team choice, Nov 3)"),
    "bank_marketing": (44126, "finance; left out to keep the domains varied (team choice, Nov 3)"),
    "banana": (1460, "artificially generated 2D data; the real-data screen uses real data only"),
    "hill_valley": (1479, "artificially generated; the real-data screen uses real data only"),
    "madelon": (1485, "artificially generated; the real-data screen uses real data only"),
    "Bioresponse": (4134, "1,776 features; too costly for SVM, MLP and GP at this scale"),
    "jm1": (1053, "25 missing values"),
}


def to_xy(frame: pd.DataFrame, target: str, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Numeric features, and a binary target mapped to {0, 1} in sorted label order.

    `features` names the columns to use; by default, every numeric column except the target.
    """
    labels = frame[target].astype(str)
    classes = sorted(labels.unique())
    if len(classes) != 2:
        raise ValueError(f"expected a binary target, got {len(classes)} classes")
    if features is None:
        features = list(frame.drop(columns=target).select_dtypes(include="number").columns)
    X = frame[features].astype(float)
    if X.isna().any().any():
        raise ValueError("unexpected missing values")
    return X.to_numpy(), (labels == classes[1]).astype(int).to_numpy()


def _get_json(url: str) -> dict:
    import json
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def load(name: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download OpenML's Parquet copy once, then keep the features OpenML lists as numeric.

    The feature list comes from OpenML's metadata, so a nominal feature stored as integers
    (an area code, say) is not mistaken for a number. The shape is checked against OpenML.
    """
    import urllib.request

    candidate = CANDIDATES[name]
    api = "https://www.openml.org/api/v1/json/data"
    description = _get_json(f"{api}/{candidate.openml_id}")["data_set_description"]
    target = description["default_target_attribute"]
    features = [
        f["name"]
        for f in _get_json(f"{api}/features/{candidate.openml_id}")["data_features"]["feature"]
        if f["data_type"] == "numeric" and f["name"] != target
        and f.get("is_ignore") != "true" and f.get("is_row_identifier") != "true"
    ]
    path = Path(cache) / f"{candidate.openml_id}.pq"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(description["parquet_url"], tmp)
        tmp.replace(path)
    X, y = to_xy(pd.read_parquet(path), target, features)
    expected = (candidate.rows, candidate.numeric_features)
    if X.shape != expected:
        raise ValueError(f"{name}: got {X.shape}, OpenML lists {expected}")
    return X, y


def make_split(X: np.ndarray, y: np.ndarray, n_train: int, seed: int):
    """Stratified training set of n_train rows and a disjoint stratified test set of TEST_SIZE rows.

    Standardization uses training statistics only.
    """
    if len(y) < n_train + TEST_SIZE:
        raise ValueError(f"needs {n_train + TEST_SIZE} rows, has {len(y)}")
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=n_train, stratify=y, random_state=seed)
    X_te, _, y_te, _ = train_test_split(X_rest, y_rest, train_size=TEST_SIZE, stratify=y_rest, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te


def per_seed_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """Tree-minus-smooth gaps for each dataset, n and seed. Positive always favors trees.

    delta_accuracy = mean accuracy(rf, gbdt) - mean accuracy(svm, mlp, gp)
    delta_log_loss = mean log loss(svm, mlp, gp) - mean log loss(rf, gbdt)
    """
    keys = ["dataset", "n", "seed"]
    tree = runs[runs.model.isin(TREE_MODELS)].groupby(keys)[["accuracy", "log_loss"]].mean()
    smooth = runs[runs.model.isin(SMOOTH_MODELS)].groupby(keys)[["accuracy", "log_loss"]].mean()
    return pd.DataFrame({
        "delta_accuracy": tree.accuracy - smooth.accuracy,
        "delta_log_loss": smooth.log_loss - tree.log_loss,
    }).reset_index()


def mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and a t-interval over seeds."""
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + level / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset and n: Delta in accuracy and log loss with 95% CIs, and a saturation flag."""
    gaps = per_seed_gaps(runs)
    by_model = runs.groupby(["dataset", "n", "model"]).accuracy.mean()
    best = by_model.groupby(["dataset", "n"]).max()
    rows = []
    for (dataset, n), group in gaps.groupby(["dataset", "n"]):
        acc, acc_lo, acc_hi = mean_ci(group.delta_accuracy)
        ll, ll_lo, ll_hi = mean_ci(group.delta_log_loss)
        rows.append({
            "dataset": dataset, "n": n, "seeds": len(group),
            "delta_accuracy": acc, "delta_accuracy_ci_low": acc_lo, "delta_accuracy_ci_high": acc_hi,
            "delta_log_loss": ll, "delta_log_loss_ci_low": ll_lo, "delta_log_loss_ci_high": ll_hi,
            "best_model_accuracy": float(best.loc[(dataset, n)]),
            # Every model's mean accuracy, so other groupings (e.g. linear-friendly) can be read later
            **{f"accuracy_{m}": float(a) for m, a in by_model.loc[(dataset, n)].items()},
        })
    table = pd.DataFrame(rows)
    saturated = table[(table.n == min(SCREEN_N)) & (table.best_model_accuracy >= SATURATION)].dataset
    table["saturated_at_n100"] = table.dataset.isin(saturated)
    return table
