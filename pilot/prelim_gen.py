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

