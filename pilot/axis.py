import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import boundaries as B
from prelim_gen import make
def axis_score(P):
    dx = np.diff(P, axis=1)[:-1, :]; dy = np.diff(P, axis=0)[:, :-1]
    l1 = np.abs(dx) + np.abs(dy); l2 = np.sqrt(dx**2 + dy**2)
    w = l1 > 1e-3                                   # only cells where the prediction changes
    return (l2[w].sum() / l1[w].sum())              # 1 = axis-aligned edges, 1/sqrt(2)=0.707 = diagonal

rows = []
for seed in (0, 1, 2):
    for dn in B.DATASETS:
        X, y = make(dn, seed); X = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, random_state=seed, stratify=y)
        pad = .6; x0,x1 = X[:,0].min()-pad, X[:,0].max()+pad; y0,y1 = X[:,1].min()-pad, X[:,1].max()+pad
        gx, gy = np.meshgrid(np.linspace(x0,x1,200), np.linspace(y0,y1,200)); G = np.c_[gx.ravel(), gy.ravel()]
        for mn, mk in B.MODELS.items():
            P = mk().fit(Xtr, ytr).predict_proba(G)[:,1].reshape(gx.shape)
            rows.append(dict(seed=seed, data=dn, model=mn.split("\n")[0], A=axis_score(P)))
df = pd.DataFrame(rows); df.to_csv("/home/claude/prelim_axis.csv", index=False)
print(df.groupby(["model","data"]).A.mean().unstack().round(3))
print(df.groupby(["model","seed"]).A.mean().groupby("model").agg(["mean","std"]).round(3))
