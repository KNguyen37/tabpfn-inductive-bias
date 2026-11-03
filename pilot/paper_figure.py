import sys; sys.argv=['x']
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import boundaries as B
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = {k: B.DATASETS[k] for k in ["Circles", "Spirals"]}
names = {"Logistic\nregression": "Logistic regression\n(straight line)",
         "Random\nforest": "Random forest\n(trees)",
         "Gradient\nboosting": "Gradient boosting\n(trees, XGBoost-style)",
         "MLP": "Neural net (MLP)",
         "Gaussian\nprocess": "Gaussian process\n(textbook Bayesian)"}
fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.15))
for i, (dn, (X, y)) in enumerate(data.items()):
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=B.SEED, stratify=y)
    pad = .6; x0,x1 = X[:,0].min()-pad, X[:,0].max()+pad; y0,y1 = X[:,1].min()-pad, X[:,1].max()+pad
    gx, gy = np.meshgrid(np.linspace(x0,x1,200), np.linspace(y0,y1,200)); G = np.c_[gx.ravel(), gy.ravel()]
    for j, (short, key) in enumerate(names.items()):
        ax = axes[i, j]; m = B.MODELS[key]().fit(Xtr, ytr)
        P = m.predict_proba(G)[:,1].reshape(gx.shape); acc = (m.predict(Xte)==yte).mean()
        ax.imshow(P, extent=(x0,x1,y0,y1), origin="lower", cmap=B.CMAP, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
        ax.contour(gx, gy, P, levels=[.5], colors="#0b0b0b", linewidths=.8)
        for c, col in ((0, B.DOT_BLUE), (1, B.DOT_RED)):
            p = Xtr[ytr==c]; ax.scatter(p[:,0], p[:,1], s=7, c=col, edgecolors="white", linewidths=.5, zorder=3)
        ax.text(.04,.04, f"acc {acc:.0%}  J {B.complexity(P)*100:.1f}", transform=ax.transAxes, fontsize=6.5,
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#c9c8c2")
        if i == 0: ax.set_title(short, fontsize=8)
        if j == 0: ax.set_ylabel(dn, fontsize=8.5)
fig.tight_layout(pad=.3, w_pad=.3, h_pad=.3)
fig.savefig("/home/claude/proposal/fig_boundaries.png", dpi=300)
print("ok")
