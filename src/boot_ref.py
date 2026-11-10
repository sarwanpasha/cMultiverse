"""Separate statistical fragility from analytic (pipeline) fragility for disease DE claims.
For the reference pipeline's partition, resample donors with replacement within each condition B times and recompute the
pseudobulk Welch test + BH; bootstrap support = fraction of resamples where the claim stays significant with the same sign.
Joins with the exhaustive multiverse support from cond_anova.
  robust            : boot >= 0.8 and multiverse >= 0.8
  pipeline-fragile  : boot >= 0.8 and multiverse <  0.5   (stable data-wise, unstable analysis-wise)
  statistically weak: boot <  0.5
Usage: python boot_ref.py --name pbmc_covid --B 200
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, load_counts
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--B", type=int, default=200)
ap.add_argument("--min_cells", type=int, default=10); ap.add_argument("--ref", default="default"); ap.add_argument("--dsub", type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(0); SUF = ("" if a.ref == "default" else "_" + a.ref) + (f"_d{a.dsub}" if a.dsub else "")
rdir = os.path.join(RES, a.name)
X, obs, genes, meta = load_counts(a.name)
claims = pd.read_csv(os.path.join(rdir, "cond_summary" + SUF, "de_claims.csv.gz"))
z = np.load(sorted(glob.glob(os.path.join(rdir, "cond" + SUF, "cond_*.npz")))[0], allow_pickle=True)
gnames = z["genes"].astype(str); gpos = {g: i for i, g in enumerate(genes)}
gsel = np.array([gpos[g] for g in gnames]); Xc = X[:, gsel].toarray().astype(np.float32)
Ltot = np.asarray(X.sum(1)).ravel().astype(np.float64); N = Xc.shape[0]
donors = obs.batch.astype(str).values; dis = obs.disease.astype(str).values
D = sorted(set(donors)); di = np.array([D.index(d) for d in donors])
dlab = np.array([pd.Series(dis[di == i]).value_counts().index[0] for i in range(len(D))])
case = "COVID-19" if (dlab == "COVID-19").any() else [x for x in pd.unique(dlab) if x != "normal"][0]
covI = np.where(dlab == case)[0]; heaI = np.where(dlab == "normal")[0]
if a.dsub:
    rs = np.random.default_rng(100 + a.dsub)
    covI = np.sort(rs.choice(covI, a.dsub, replace=False)); heaI = np.sort(rs.choice(heaI, a.dsub, replace=False))
# populations: the reference pipeline's clusters matched to the reference populations (as in cond_claims)
rnorm = "pearson" if a.ref == "pearson" else "log1p_cp10k"
for f in sorted(glob.glob(os.path.join(rdir, "lattice", "task_*.npz"))):
    t = np.load(f, allow_pickle=True)
    if int(t["nhvg"][0]) == 2000 and str(t["norm"][0]) == rnorm: tref = dict(t); break
rb = "harmony" if meta.get("n_batches", 1) > 1 else "none"
kk, rr = (30, 1.2) if a.ref == "alt_k30r12" else (15, 0.8)
m = (tref["ncomp"] == 30) & (tref["batch"] == rb) & (tref["k"] == kk) & np.isclose(tref["res"].astype(float), rr)
lab_pipe = tref["labels"][np.where(m)[0][0], 0].astype(int)
if a.ref == "author":
    refpop = pd.Categorical(obs.cell_type.astype(str)).codes.astype(int)
else:
    refpop = lab_pipe
cl_ids = sorted(claims.cluster.unique())
oh = np.zeros((lab_pipe.max() + 1, len(lab_pipe)), np.float32); oh[lab_pipe, np.arange(len(lab_pipe))] = 1
lab = np.full(len(lab_pipe), -1)
for c in cl_ids:
    rm = (refpop == c).astype(np.float32); inter = oh @ rm; J = inter / (rm.sum() + oh.sum(1) - inter)
    lab[lab_pipe == J.argmax()] = c
from scipy import stats
def welch(x1, x0):
    m1, m0 = np.nanmean(x1, 0), np.nanmean(x0, 0); v1, v0 = np.nanvar(x1, 0, ddof=1), np.nanvar(x0, 0, ddof=1)
    n1, n0 = np.sum(~np.isnan(x1), 0), np.sum(~np.isnan(x0), 0); se = np.sqrt(v1 / n1 + v0 / n0); t = (m1 - m0) / se
    df = se ** 4 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1)); return t, 2 * stats.t.sf(np.abs(t), df)
def bh(p):
    p = np.where(np.isfinite(p), p, 1.0); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)
Dh = np.zeros((len(D), N), np.float32); Dh[di, np.arange(N)] = 1
pb = {}
for c in sorted(claims.cluster.unique()):
    mask = (lab == c).astype(np.float32); nd = Dh @ mask
    Sx = (Dh * mask[None, :]) @ Xc; Ld = (Dh * mask[None, :]) @ Ltot
    cpm = np.log2(Sx / np.maximum(Ld, 1)[:, None] * 1e6 + 1); cpm[nd < a.min_cells] = np.nan; pb[c] = cpm
hits = {k: 0 for k in range(len(claims))}; dhits = {k: 0 for k in range(len(claims))}
gi = {g: i for i, g in enumerate(gnames)}
for b in range(a.B):
    cs = rng.choice(covI, len(covI)); hs = rng.choice(heaI, len(heaI))
    for c, cpm in pb.items():
        t, p = welch(cpm[cs], cpm[hs]); q = bh(p); sig = np.sign(t) * (q < 0.1)
        for k in np.where(claims.cluster.values == c)[0]:
            if sig[gi[claims.gene.values[k]]] == claims.direction.values[k]: hits[k] += 1
            if np.sign(t[gi[claims.gene.values[k]]]) == claims.direction.values[k]: dhits[k] += 1
claims["boot_support"] = [hits[k] / a.B for k in range(len(claims))]
claims["boot_dir_consistency"] = [dhits[k] / a.B for k in range(len(claims))]
cl = np.where((claims.boot_support >= 0.8) & (claims.support >= 0.8), "robust",
     np.where((claims.boot_support >= 0.8) & (claims.support < 0.5), "pipeline-fragile",
     np.where(claims.boot_support < 0.5, "statistically-weak", "intermediate")))
claims["class"] = cl
claims["class_direction"] = np.where((claims.boot_dir_consistency >= 0.95) & (claims.dir_consistency >= 0.95), "robust",
    np.where((claims.boot_dir_consistency >= 0.95) & (claims.dir_consistency < 0.8), "pipeline-fragile",
    np.where(claims.boot_dir_consistency < 0.8, "statistically-weak", "intermediate")))
claims.to_csv(os.path.join(rdir, "cond_summary" + SUF, "de_claims_boot.csv.gz"), index=False)
s = dict(n=len(claims), class_counts=claims["class"].value_counts().to_dict(),
         class_counts_direction=claims["class_direction"].value_counts().to_dict(),
         direction_fragile_examples=claims[claims["class_direction"] == "pipeline-fragile"][["cell_type", "gene", "direction", "t_ref", "boot_dir_consistency", "dir_consistency"]].head(15).to_dict(orient="records"),
         spearman_boot_vs_multiverse=float(claims[["boot_support", "support"]].corr(method="spearman").iloc[0, 1]),
         pipeline_fragile_examples=claims[claims["class"] == "pipeline-fragile"].sort_values("boot_support", ascending=False)
             [["cell_type", "gene", "direction", "t_ref", "boot_support", "support"]].head(15).to_dict(orient="records"),
         robust_examples=claims[claims["class"] == "robust"][["cell_type", "gene", "direction", "t_ref"]].head(15).to_dict(orient="records"))
json.dump(s, open(os.path.join(rdir, "cond_summary" + SUF, "bootstrap_summary.json"), "w"), indent=1, default=str)
print(json.dumps(s, indent=1, default=str))
