import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import boundaries as B
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

INK, MUTED, EDGE, BG = "#0b0b0b", "#52514e", "#c9c8c2", "#fcfcfb"
X, y = B.DATASETS["Spirals"]; X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, random_state=B.SEED, stratify=y)
pad=.6; x0,x1=X[:,0].min()-pad,X[:,0].max()+pad; y0,y1=X[:,1].min()-pad,X[:,1].max()+pad
gx, gy = np.meshgrid(np.linspace(x0,x1,120), np.linspace(y0,y1,120)); G=np.c_[gx.ravel(),gy.ravel()]
keys = ["Logistic regression\n(straight line)","Random forest\n(trees)","Gradient boosting\n(trees, XGBoost-style)","Neural net (MLP)","Gaussian process\n(textbook Bayesian)"]
short = ["LogReg","RF","GBoost","MLP","GP"]
P = {s: B.MODELS[k]().fit(Xtr,ytr).predict_proba(G)[:,1].reshape(gx.shape) for k,s in zip(keys,short)}

fig = plt.figure(figsize=(7.2, 3.35), facecolor="white")
def frame(ax, label, title):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(EDGE)
    ax.set_title(f"({label}) {title}", fontsize=7.5, loc="left", color=INK, pad=3, fontweight="bold")

# (C) controls bar
axc = fig.add_axes([0.01, 0.86, 0.98, 0.1]); axc.set_xlim(0,1); axc.set_ylim(0,1); axc.axis("off")
axc.add_patch(FancyBboxPatch((0.002,0.08),0.996,0.84, boxstyle="round,pad=0,rounding_size=0.02", fc=BG, ec=EDGE, lw=.8))
axc.text(0.012,0.5,"(C)", fontsize=7.5, fontweight="bold", va="center", color=INK)
items = [("Dataset","Spirals ▾",0.045,0.06),("Model","GP ▾",0.195,0.047),("Version","v2 | 3.5",0.32,0.056),("Seed","0 ▾",0.462,0.04)]
for lab,val,x,off in items:
    axc.text(x,0.5,f"{lab}:", fontsize=6.8, va="center", color=MUTED)
    axc.add_patch(FancyBboxPatch((x+off,0.25),0.068,0.5, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=EDGE, lw=.7))
    axc.text(x+off+0.034,0.5,val, fontsize=6.6, va="center", ha="center", color=INK)
axc.text(0.565,0.5,"Train size n:", fontsize=6.8, va="center", color=MUTED)
axc.plot([0.645,0.80],[0.5,0.5], color=EDGE, lw=2.5, solid_capstyle="round")
axc.plot([0.645,0.72],[0.5,0.5], color="#2a78d6", lw=2.5, solid_capstyle="round")
axc.plot([0.72],[0.5], "o", ms=5, color="#2a78d6", mec="white", mew=1)
axc.text(0.81,0.5,"60", fontsize=6.8, va="center", color=INK)
axc.text(0.85,0.5,"Rotate ☐   Noise ☐", fontsize=6.4, va="center", color=MUTED)

# (A) main canvas
axa = fig.add_axes([0.01, 0.06, 0.36, 0.7])
axa.imshow(P["GP"], extent=(x0,x1,y0,y1), origin="lower", cmap=B.CMAP, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
axa.contour(gx,gy,P["GP"],levels=[.5],colors=INK,linewidths=.8)
for c,col in ((0,B.DOT_BLUE),(1,B.DOT_RED)):
    p=Xtr[ytr==c]; axa.scatter(p[:,0],p[:,1],s=6,c=col,edgecolors="white",linewidths=.4,zorder=3)
cx, cy = 1.15, -0.6
axa.axvline(cx, color=INK, lw=.5, ls=(0,(2,2))); axa.axhline(cy, color=INK, lw=.5, ls=(0,(2,2)))
pg = P["GP"][np.abs(gy[:,0]-cy).argmin(), np.abs(gx[0]-cx).argmin()]; ent = -(pg*np.log2(pg)+(1-pg)*np.log2(1-pg))
axa.text(cx+0.1, cy+0.12, f"p(red) = {pg:.2f}\nentropy = {ent:.2f} bits", fontsize=5.8, color=INK, bbox=dict(boxstyle="round,pad=.25", fc="white", ec=EDGE, lw=.5))
frame(axa, "A", "Main canvas")
axa.text(0.02,0.97,"View: probability | uncertainty | boundary", transform=axa.transAxes, fontsize=5.6, va="top", color=MUTED,
         bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85))

# (B) small multiples with linked crosshair
w, gap, left = 0.105, 0.012, 0.39
for i,s in enumerate(short):
    ax = fig.add_axes([left + i*(w+gap), 0.43, w, 0.33])
    ax.imshow(P[s], extent=(x0,x1,y0,y1), origin="lower", cmap=B.CMAP, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    ax.axvline(cx, color=INK, lw=.45, ls=(0,(2,2))); ax.axhline(cy, color=INK, lw=.45, ls=(0,(2,2)))
    pv = P[s][np.abs(gy[:,0]-cy).argmin(), np.abs(gx[0]-cx).argmin()]
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#2a78d6" if s=="GP" else EDGE); sp.set_linewidth(1.4 if s=="GP" else .8)
    ax.set_xlabel(f"{s}  p={pv:.2f}", fontsize=5.8, color=INK, labelpad=1.5)
    if i == 0: ax.set_title("(B) Small multiples, linked crosshair", fontsize=7.5, loc="left", color=INK, pad=3, fontweight="bold")

# (D) metrics dot plot from the pilot table
df = pd.read_csv("/home/claude/prelim_axis.csv")
S = df.groupby(["model","seed"]).A.mean().groupby("model").agg(["mean","min","max"])
r = pd.read_csv("/home/claude/prelim_results.csv")
F = r.groupby(["model","seed"]).far.mean().groupby("model").agg(["mean","min","max"])
order = ["Logistic regression","k-nearest neighbors","Random forest","Gradient boosting","SVM","Neural net (MLP)","Gaussian process"]
lab = ["LogReg","k-NN","RF","GBoost","SVM","MLP","GP"]
for j,(M,name,xl) in enumerate([(S,"Step score S",(0.75,1.01)),(F,"Far-field confidence",(0.45,1.02))]):
    ax = fig.add_axes([0.45 + j*0.28, 0.06, 0.22, 0.25])
    yy = np.arange(len(order))[::-1]
    for k,m in enumerate(order):
        hl = m=="Gaussian process"
        ax.plot([M.loc[m,"min"],M.loc[m,"max"]],[yy[k]]*2, color="#2a78d6" if hl else MUTED, lw=1)
        ax.plot(M.loc[m,"mean"], yy[k], "o", ms=3.6 if not hl else 4.5, color="#2a78d6" if hl else INK)
    ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=5.6); ax.set_xlim(*xl)
    ax.tick_params(axis="x", labelsize=5.4, length=2); ax.tick_params(axis="y", length=0)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color(EDGE)
    ax.grid(axis="x", color="#ecebe7", lw=.5); ax.set_axisbelow(True)
    ax.set_title(name, fontsize=6.4, color=INK, pad=2)
fig.text(0.39, 0.345, "(D) Metrics panel", fontsize=7.5, fontweight="bold", color=INK)
fig.savefig("/home/claude/proposal/fig_ui.png", dpi=300)
print("ok")
