"""Download, cache and split the two real datasets exactly as described in the proposal.

Requires scikit-learn, pandas and pyarrow (all preinstalled on Google Colab).
Run on Colab or any machine with internet access:
    python get_real_data.py            # downloads, caches, writes splits
Outputs (in ./data/):
    banknote.parquet, electricity.parquet          cached raw tables
    checksums.txt                                  SHA-256 of each cached table
    splits/{name}_n{n}_seed{s}.npz                  X_train, y_train, X_test, y_test (standardized)
    fingerprint.csv                                tree-minus-smooth gap per dataset / n / seed
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = {"banknote": 1462, "electricity": 44120}   # OpenML data IDs (each pins one version)
EXPECTED = {"banknote": (1372, 4), "electricity": (38474, 7)}  # rows, features, as listed on OpenML
TRAIN_SIZES = (100, 500, 1000)
TEST_SIZE = 300
SEEDS = (0, 1, 2)
OUT = Path("data")


def load(name, data_id, loader=fetch_openml):
    """Fetch from OpenML once, then reuse the cached Parquet file."""
    path = OUT / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        bunch = loader(data_id=data_id, as_frame=True)
        df = bunch.frame.copy()
        df = df.rename(columns={bunch.target_names[0]: "target"})
        OUT.mkdir(exist_ok=True)
        df.to_parquet(path, index=False)
    X = df.drop(columns="target").astype(float)
    # map the two class labels to {0, 1} in a fixed (sorted) order
    classes = sorted(df["target"].astype(str).unique())
    assert len(classes) == 2, f"{name}: expected a binary target, got {classes}"
    y = (df["target"].astype(str) == classes[1]).astype(int).to_numpy()
    assert X.shape == EXPECTED[name], f"{name}: got {X.shape}, expected {EXPECTED[name]}"
    assert not X.isna().any().any(), f"{name}: unexpected missing values"
    return X, y, path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(X, y, n_train, seed):
    """Stratified train set of n_train rows and a disjoint stratified test set of TEST_SIZE rows;
    standardization uses training statistics only."""
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X.to_numpy(), y, train_size=n_train, stratify=y, random_state=seed)
    X_te, _, y_te, _ = train_test_split(
        X_rest, y_rest, train_size=TEST_SIZE, stratify=y_rest, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te


def fingerprint(X_tr, y_tr, X_te, y_te, seed):
    """Delta = mean accuracy of tree models minus mean accuracy of smooth models."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    trees = [RandomForestClassifier(300, random_state=seed),
             HistGradientBoostingClassifier(random_state=seed)]
    smooth = [SVC(probability=True, random_state=seed),
              MLPClassifier((64, 64), max_iter=2000, random_state=seed),
              GaussianProcessClassifier(ConstantKernel() * RBF(), random_state=seed)]
    acc = lambda ms: np.mean([m.fit(X_tr, y_tr).score(X_te, y_te) for m in ms])
    return acc(trees) - acc(smooth)


def main(loader=fetch_openml, run_fingerprint=True):
    OUT.mkdir(exist_ok=True)
    (OUT / "splits").mkdir(exist_ok=True)
    rows, sums = [], []
    for name, data_id in DATASETS.items():
        X, y, path = load(name, data_id, loader)
        sums.append(f"{name}\tOpenML {data_id}\t{sha256(path)}")
        for n in TRAIN_SIZES:
            for seed in SEEDS:
                X_tr, y_tr, X_te, y_te = make_split(X, y, n, seed)
                np.savez(OUT / "splits" / f"{name}_n{n}_seed{seed}.npz",
                         X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                         feature_names=np.array(X.columns))
                if run_fingerprint:
                    rows.append(dict(dataset=name, n=n, seed=seed,
                                     delta=fingerprint(X_tr, y_tr, X_te, y_te, seed)))
    (OUT / "checksums.txt").write_text("\n".join(sums) + "\n")
    if rows:
        fp = pd.DataFrame(rows)
        fp.to_csv(OUT / "fingerprint.csv", index=False)
        print(fp.groupby(["dataset", "n"]).delta.agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
