import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.datasets import make_moons, make_circles
import boundaries as B

def ece(p, y, bins=10):
    conf = np.maximum(p, 1-p); pred = (p >= .5).astype(int); e = 0
    edges = np.linspace(.5, 1, bins+1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m]==y[m]).mean() - conf[m].mean())
    return e

def make(name, seed):
    n = B.N
    if name == "Moons": return make_moons(n, noise=.25, random_state=seed)
    if name == "Circles": return make_circles(n, noise=.12, factor=.45, random_state=seed)
    if name == "XOR (checkerboard)": return B.xor(n, seed=seed)
    return B.spirals(n, seed=seed)

rows = []
for seed in (0, 1, 2):
    for dn in B.DATASETS:
        X, y = make(dn, seed); X = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, random_state=seed, stratify=y)
        pad = .6; x0,x1 = X[:,0].min()-pad, X[:,0].max()+pad; y0,y1 = X[:,1].min()-pad, X[:,1].max()+pad
        gx, gy = np.meshgrid(np.linspace(x0,x1,200), np.linspace(y0,y1,200)); G = np.c_[gx.ravel(), gy.ravel()]
        # far field: wide grid at 3x the data range, cells outside the convex hull of the training points
        cx, cy = X[:,0].mean(), X[:,1].mean(); rx, ry = np.ptp(X[:,0])*1.5, np.ptp(X[:,1])*1.5
        fx, fy = np.meshgrid(np.linspace(cx-rx, cx+rx, 120), np.linspace(cy-ry, cy+ry, 120))
        F = np.c_[fx.ravel(), fy.ravel()]; F = F[Delaunay(Xtr).find_simplex(F) < 0]
        for mn, mk in B.MODELS.items():
            m = mk().fit(Xtr, ytr)
            P = m.predict_proba(G)[:,1].reshape(gx.shape)
            pt = np.clip(m.predict_proba(Xte)[:,1], 1e-6, 1-1e-6)
            pf = m.predict_proba(F)[:,1]
            rows.append(dict(seed=seed, data=dn, model=mn.split("\n")[0], acc=((pt>=.5)==yte).mean(),
                             nll=log_loss(yte, pt), ece=ece(pt, yte), J=B.complexity(P)*100,
                             far=np.maximum(pf, 1-pf).mean()))
df = pd.DataFrame(rows); df.to_csv("/home/claude/prelim_results.csv", index=False)
g = df.groupby(["model","data"]).agg(['mean','std'])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.groupby(["model","data"])[["acc","J","far","ece","nll"]].mean().round(3))
print(df.groupby("model")[["acc","J","far","ece","nll"]].mean().round(3))
print(df.groupby(["model","data"])[["acc","J"]].std().round(3).max())
