"""Decision boundaries of classic models on small 2D toy datasets.

Each panel: background = model's predicted probability of the red class
(blue = sure it's blue, red = sure it's red, pale gray = unsure).
Dots = training points. Corner numbers: test accuracy and boundary
"jaggedness" (complexity score from den Breejen et al., 2024).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

SEED = 0
N = 120  # total points; half used for training -> small-data regime


def spirals(n, noise=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0, 1, n // 2)) * 3 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)]
    b = -a
    X = np.r_[a, b] + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    return X, y


def xor(n, noise=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.normal(0, noise, X.shape)
    return X, y


DATASETS = {
    "Moons": make_moons(N, noise=0.25, random_state=SEED),
    "Circles": make_circles(N, noise=0.12, factor=0.45, random_state=SEED),
    "XOR (checkerboard)": xor(N),
    "Spirals": spirals(N),
}

MODELS = {
    "Logistic regression\n(straight line)": lambda: LogisticRegression(),
    "k-nearest neighbors\n(k=5)": lambda: KNeighborsClassifier(5),
    "Random forest\n(trees)": lambda: RandomForestClassifier(300, random_state=SEED),
    "Gradient boosting\n(trees, XGBoost-style)": lambda: HistGradientBoostingClassifier(
        max_iter=200, min_samples_leaf=3, random_state=SEED),
    "SVM\n(RBF kernel)": lambda: SVC(probability=True, gamma="scale", C=3, random_state=SEED),
    "Neural net (MLP)": lambda: MLPClassifier((64, 64), max_iter=3000, random_state=SEED),
    "Gaussian process\n(textbook Bayesian)": lambda: GaussianProcessClassifier(
        ConstantKernel() * RBF(), random_state=SEED),
}

# Diverging blue <-> red with a neutral gray midpoint (unsure = 0.5).
BLUE, RED, MID = "#2a78d6", "#d8413a", "#f0efec"
CMAP = LinearSegmentedColormap.from_list("bluered", [BLUE, MID, RED])
DOT_BLUE, DOT_RED = "#1d5ea9", "#a82a24"


def complexity(P):
    """Sum of absolute probability change between neighbouring grid cells,
    normalised per cell. Higher = more jagged boundary."""
    return (np.abs(np.diff(P, axis=0)).sum() + np.abs(np.diff(P, axis=1)).sum()) / P.size


def run(extra_models=None, out="decision_boundaries.png", res=200):
    models = dict(MODELS)
    if extra_models:
        models.update(extra_models)
    nr, nc = len(DATASETS), len(models)
    fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc, 2.5 * nr + 0.9),
                             facecolor="#fcfcfb")
    for i, (dname, (X, y)) in enumerate(DATASETS.items()):
        X = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                              random_state=SEED, stratify=y)
        pad = 0.6
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
        grid = np.c_[gx.ravel(), gy.ravel()]
        for j, (mname, make) in enumerate(models.items()):
            ax = axes[i, j]
            m = make().fit(Xtr, ytr)
            P = m.predict_proba(grid)[:, 1].reshape(gx.shape)
            acc = (m.predict(Xte) == yte).mean()
            ax.imshow(P, extent=(x0, x1, y0, y1), origin="lower", cmap=CMAP,
                      vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
            ax.contour(gx, gy, P, levels=[0.5], colors="#0b0b0b", linewidths=1)
            for cls, col in ((0, DOT_BLUE), (1, DOT_RED)):
                pts = Xtr[ytr == cls]
                ax.scatter(pts[:, 0], pts[:, 1], s=16, c=col,
                           edgecolors="white", linewidths=0.8, zorder=3)
            ax.text(0.03, 0.03, f"acc {acc:.0%}  ·  jag {complexity(P) * 100:.1f}",
                    transform=ax.transAxes, fontsize=7.5, color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d6d5d0")
            if i == 0:
                ax.set_title(mname, fontsize=9, color="#0b0b0b", pad=6)
            if j == 0:
                ax.set_ylabel(dname, fontsize=10, color="#0b0b0b", labelpad=8)
    fig.suptitle("Same dots, different guesses: how each model fills in the space between points",
                 fontsize=13, color="#0b0b0b", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.955,
             f"{N // 2} training points per dataset (shown). Shading = predicted chance of red: "
             "deep blue/red = confident, pale gray = unsure. Black line = 50/50 boundary.\n"
             "acc = accuracy on 60 held-out points. jag = boundary jaggedness (higher = blockier/more complex).",
             fontsize=8.5, color="#52514e", ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", out)


if __name__ == "__main__":
    run()
