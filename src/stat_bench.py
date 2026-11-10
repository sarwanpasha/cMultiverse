"""Marker-statistic benchmark against ground truth, on partitions sampled from the exhaustive lattice.
Statistics (gene g, population cluster c vs rest):
  auc_L        : AUC on log1p-CP10k (zeros tied)                      [nonzero: log,     zeros: tied]
  auc_DZ       : log1p-CP10k nonzeros, zeros ordered deep < shallow  [nonzero: log,     zeros: depth-ordered]
  auc_DZrev    : control, zeros ordered shallow < deep               [nonzero: log,     zeros: reversed]
  auc_Pzt      : Pearson nonzeros, zeros tied                         [nonzero: pearson, zeros: tied]
  auc_P        : Pearson residuals                                    [nonzero: pearson, zeros: depth-ordered]
  rr_z         : offset Poisson rate-ratio Wald z  log((xA/LA)/(xB/LB)) / sqrt(1/xA + 1/xB)
  qp_z         : quasi-Poisson rate-ratio z (gene-wise dispersion from Pearson chi2)
Metrics per pipeline: recall@50 of known-true markers; mean per-population AUROC (true vs other genes).
Usage: python stat_bench.py --name cite_hao --n_pipes 300
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, load_counts, auc_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--n_pipes", type=int, default=300)
a = ap.parse_args(); rng = np.random.default_rng(0)
d = dataset_dir(a.name); rdir = os.path.join(RES, a.name); vd = os.path.join(rdir, "values")
pz = np.load(os.path.join(vd, "panel.npz"), allow_pickle=True); panel = pz["panel"]; pg = pz["genes"].astype(str)
X, obs, genes, meta = load_counts(a.name)
Xp = X[:, panel].tocsr(); L = np.asarray(X.sum(1)).ravel().astype(np.float64); N, G = Xp.shape
Xd = Xp.toarray(); Z0 = Xd == 0
logv = np.load(os.path.join(vd, "values_log1p_cp10k.npy"))
dz = np.where(Z0, -(L / L.max())[:, None], logv); dzr = np.where(Z0, -(1 - L / L.max())[:, None] - 1e-6, logv)
RANKS = {"auc_L": np.load(os.path.join(vd, "ranks_log1p_cp10k.npy")), "auc_P": np.load(os.path.join(vd, "ranks_pearson.npy")),
         "auc_Pzt": np.load(os.path.join(vd, "ranks_pearson_zt.npy")),
         "auc_DZ": rankdata(dz, axis=0, method="average").astype(np.float32), "auc_DZrev": rankdata(dzr, axis=0, method="average").astype(np.float32)}
del dz, dzr
# empirical-Bayes gamma-Poisson posterior mean rate per cell: (a + x) / (b + L); gene-wise method-of-moments prior
m_g = Xd.sum(0) / L.sum()
s2 = (((Xd - np.outer(L, m_g)) ** 2).sum(0) - (L[:, None] * m_g[None, :]).sum(0)) / (L ** 2).sum()
s2 = np.clip(s2, 1e-3 * m_g ** 2 + 1e-12, None)
a_g, b_g = m_g ** 2 / s2, m_g / s2
eb = (a_g[None, :] + Xd) / (b_g[None, :] + L[:, None])
RANKS["auc_EB"] = rankdata(eb, axis=0, method="average").astype(np.float32); del eb
# depth strata (deciles) for the van Elteren stratified AUC on log1p-CP10k
nb = 10; strata = np.minimum((rankdata(L) - 1) * nb // N, nb - 1).astype(int)
SR = [(np.where(strata == s)[0], rankdata(logv[strata == s], axis=0, method="average").astype(np.float32)) for s in range(nb)]
def auc_vE(lab, K):
    U = np.zeros((K, G)); W = np.zeros((K, 1))
    for idx, Rs in SR:
        ls = lab[idx]; n = len(idx)
        oh = np.zeros((K, n), np.float32); oh[ls, np.arange(n)] = 1
        nc = oh.sum(1).astype(np.float64); mc = n - nc
        U += (oh @ Rs) - (nc * (nc + 1) / 2)[:, None]; W += (nc * mc)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return U / W
# gene-wise quasi-Poisson dispersion: mean squared Pearson residual under the global Poisson model
pgl = Xd.sum(0) / L.sum(); mu = np.outer(L, pgl)
with np.errstate(divide="ignore", invalid="ignore"):
    phi = np.nanmean((Xd - mu) ** 2 / np.where(mu > 0, mu, np.nan), axis=0)
phi = np.clip(np.nan_to_num(phi, nan=1.0), 1.0, None); del mu

# claim spec (same as claims.py): true marker sets per population
z0 = np.load(sorted(glob.glob(os.path.join(rdir, "claims", "claims_*.npz")))[0], allow_pickle=True)
set_names = list(z0["set_names"]); true_sets = [set(s.split(";")) - {""} for s in z0["true_sets"]]
ref = obs["cell_type"].astype(str).values
if a.name.startswith(("pbmc", "hlca")):
    import re
    ms = json.load(open(os.path.join(os.path.dirname(__file__), "marker_sets.json")))["pbmc" if a.name.startswith("pbmc") else "hlca"]
    masks = {}
    types = np.unique(ref)
    for nm, spec in ms.items():
        mem = [t for t in types if re.search(spec["regex"], t, flags=re.I)]
        if spec.get("exclude"): mem = [t for t in mem if not re.search(spec["exclude"], t, flags=re.I)]
        masks[nm] = np.isin(ref, mem)
else:
    masks = {s: ref == s for s in set_names}
setmat = np.stack([masks[s] for s in set_names]).astype(np.float32); ssize = setmat.sum(1)
gidx = {g: i for i, g in enumerate(pg)}
has = [i for i, t in enumerate(true_sets) if t]
cand = None
if a.name.startswith("cite"):   # protein truth: evaluate AUROC over the protein-mapped genes only
    cand = np.array(sorted({gidx[g] for g in z0["pair_gene"] if g in gidx}))

files = sorted(glob.glob(os.path.join(rdir, "lattice", "task_*.npz")))
rows = []
for f in files:
    lat = np.load(f, allow_pickle=True)
    take = rng.choice(lat["labels"].shape[0], size=max(1, a.n_pipes // len(files)), replace=False)
    for p in take:
        lab = lat["labels"][p, 0].astype(int); K = lab.max() + 1
        oh = np.zeros((K, N), np.float32); oh[lab, np.arange(N)] = 1
        inter = setmat @ oh.T; cs = oh.sum(1)
        best = (inter / (ssize[:, None] + cs[None, :] - inter)).argmax(1)
        S = {k: auc_matrix(R, lab) for k, R in RANKS.items()}
        xs = oh @ Xd; Ls = oh @ L; xT = Xd.sum(0); LT = L.sum()
        xA = xs + 0.5; xB = (xT[None, :] - xs) + 0.5; LA = Ls[:, None]; LB = LT - LA
        lrr = np.log((xA / LA) / (xB / LB)); se = np.sqrt(1 / xA + 1 / xB)
        S["rr_z"] = lrr / se; S["qp_z"] = lrr / (se * np.sqrt(phi)[None, :])
        S["auc_vE"] = auc_vE(lab, K)
        rec = {"task": os.path.basename(f), "pipe": int(p), "norm": str(lat["norm"][p]), "batch": str(lat["batch"][p]),
               "k": int(lat["k"][p]), "res": float(lat["res"][p])}
        for k, A in S.items():
            A = np.nan_to_num(A, nan=0.5 if k.startswith("auc") else 0.0)
            r50, au = [], []
            for i in has:
                sc = A[best[i]]
                top = set(pg[np.argsort(-sc)[:50]]); r50.append(len(top & true_sets[i]) / len(true_sets[i]))
                gi = cand if cand is not None else np.arange(G)
                y = np.isin(pg[gi], list(true_sets[i]))
                if 0 < y.sum() < len(y): au.append(roc_auc_score(y, sc[gi]))
            rec[k + "|recall50"] = float(np.mean(r50)); rec[k + "|auroc"] = float(np.mean(au)) if au else np.nan
        rows.append(rec)
df = pd.DataFrame(rows)
od = os.path.join(rdir, "stat_bench"); os.makedirs(od, exist_ok=True)
df.to_csv(os.path.join(od, "stat_bench.csv"), index=False)
cols = [c for c in df.columns if "|" in c]
summ = df[cols].agg(["median", "mean"]).T.round(4)
wins = {c: float((df[c] > df["auc_L|" + c.split("|")[1]]).mean()) for c in cols if not c.startswith("auc_L|")}
print(summ.to_string()); print("frac pipelines beating auc_L:", json.dumps({k: round(v, 3) for k, v in wins.items()}))
json.dump({"summary": summ.to_dict(), "wins_vs_auc_L": wins, "n_pipelines": len(df)}, open(os.path.join(od, "summary.json"), "w"), indent=1)
