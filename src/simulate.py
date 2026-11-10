"""Negative-binomial simulator with planted markers, rare types, donor batch effects and
controllable cell-type-specific sequencing depth (the quantity that drives the Pearson-residual tie theorem).
Usage: python simulate.py --name sim_d2 --depth_ratio 2.0 --seed 0
"""
import argparse, os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from common import dataset_dir

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--depth_ratio", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_cells", type=int, default=10000)
ap.add_argument("--n_genes", type=int, default=6000); ap.add_argument("--n_markers", type=int, default=40)
ap.add_argument("--n_batches", type=int, default=5); ap.add_argument("--theta", type=float, default=10.0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

abund = np.array([0.28, 0.20, 0.15, 0.12, 0.10, 0.07, 0.05, 0.02, 0.01])
T = len(abund)
# per-type depth multipliers spread log-uniformly over [1/sqrt(r), sqrt(r)], assigned in a fixed shuffled order
logd = np.linspace(-0.5, 0.5, T) * np.log(a.depth_ratio)
depth_mult = np.exp(rng.permutation(logd))

G = a.n_genes
base = rng.gamma(0.3, 1.0, G); base /= base.sum()
q = np.argsort(np.argsort(base)) / (G - 1)          # baseline expression quantile per gene
fc = np.ones((T, G))
marker_rows = []
used = set()
for t in range(T):
    # markers spread over baseline quantiles so that sparse markers exist
    pool = np.array([g for g in rng.permutation(G) if g not in used])
    bins = np.array_split(pool[np.argsort(q[pool])], a.n_markers)
    for b in bins:
        g = int(rng.choice(b)); used.add(g)
        f = float(np.exp(rng.normal(np.log(4.0), 0.5)))
        fc[t, g] = f
        marker_rows.append((f"G{g}", f"T{t}", f, float(base[g]), float(q[g])))
P = base[None, :] * fc
P /= P.sum(1, keepdims=True)

be = np.exp(rng.normal(0, 0.2, (a.n_batches, G)))
types = rng.choice(T, size=a.n_cells, p=abund)
batch = rng.integers(0, a.n_batches, a.n_cells)
L = np.exp(rng.normal(np.log(3000), 0.45, a.n_cells)) * depth_mult[types]
mu = L[:, None] * P[types] * be[batch]
mu = mu / (P[types] * be[batch]).sum(1, keepdims=True) * 1.0
lam = rng.gamma(a.theta, mu / a.theta)
X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
keep_g = np.asarray((X > 0).sum(0)).ravel() >= 10
genes = np.array([f"G{g}" for g in range(G)])
X = X[:, keep_g]; genes = genes[keep_g]

o = pd.DataFrame(index=[f"c{i}" for i in range(a.n_cells)])
o["cell_type"] = [f"T{t}" for t in types]; o["batch"] = [f"B{b}" for b in batch]
o["disease"] = "sim"; o["assay"] = "sim"
o["total_counts"] = np.asarray(X.sum(1)).ravel(); o["pct_mt"] = 0.0
o["true_depth_mult"] = depth_mult[types]
d = dataset_dir(a.name); os.makedirs(d, exist_ok=True)
sp.save_npz(os.path.join(d, "counts.npz"), X)
o.to_csv(os.path.join(d, "obs.csv"))
pd.DataFrame({"gene": genes}).to_csv(os.path.join(d, "genes.csv"), index=False)
mk = pd.DataFrame(marker_rows, columns=["gene", "cell_type", "fold_change", "baseline_p", "baseline_q"])
mk = mk[mk.gene.isin(set(genes))]
mk.to_csv(os.path.join(d, "true_markers.csv"), index=False)
meta = dict(source="simulation", depth_ratio=a.depth_ratio, seed=a.seed, n_cells=int(X.shape[0]),
            n_genes=int(X.shape[1]), n_batches=a.n_batches, abundances=abund.tolist(),
            depth_mult=depth_mult.tolist(), theta=a.theta)
json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1)); print(o.groupby("cell_type").total_counts.median())
