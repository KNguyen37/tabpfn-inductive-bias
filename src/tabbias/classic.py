"""The seven classic baselines, with near-default settings.

Unlike the pilot, every stochastic model receives the run's seed, so seed spread
includes model randomness as well as data randomness.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CLASSIC_MODELS = ("logreg", "knn", "rf", "gbdt", "svm", "mlp", "gp")
TREE_MODELS = ("rf", "gbdt")
SMOOTH_MODELS = ("svm", "mlp", "gp")


def make_classic(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "rf":
        return RandomForestClassifier(300, random_state=seed, n_jobs=-1)
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=3, random_state=seed)
    if name == "svm":
        return SVC(probability=True, gamma="scale", C=3, random_state=seed)
    if name == "mlp":
        return MLPClassifier((64, 64), max_iter=3000, random_state=seed)
    if name == "gp":
        return GaussianProcessClassifier(ConstantKernel() * RBF(), random_state=seed)
    raise ValueError(f"unknown classic model: {name}")
