"""Disease claims (COVID-19 vs healthy) for every pipeline of one lattice task (Leiden seed 0).
Clusters are the reference run's clusters, matched by best Jaccard in each pipeline.
  composition claim : population c changes in COVID (Welch t on donor-level logit fraction)
  DE claim          : gene g differs COVID vs healthy within population c (donor pseudobulk log2 CPM, Welch t, BH FDR<0.1)
Usage: python cond_claims.py --name pbmc_covid --task 3
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, load_counts

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--task", type=int, required=True)
ap.add_argument("--n_genes", type=int, default=100000); ap.add_argument("--ref", default="default")
ap.add_argument("--case", default=None, help="disease label for cases (default: COVID-19 if present, else the non-normal label)")
ap.add_argument("--dsub", type=int, default=0, help="use only this many donors per group (power analysis)"); ap.add_argument("--min_cells", type=int, default=10)
a = ap.parse_args()
rdir = os.path.join(RES, a.name)
X, obs, genes, meta = load_counts(a.name)
pz = np.load(os.path.join(rdir, "values", "panel.npz"), allow_pickle=True)
det = 1 - pz["zero_frac"]; gsel = pz["panel"][np.argsort(-det)[:a.n_genes]]; gnames = genes[gsel]
Xc = X[:, gsel].toarray().astype(np.float32); Ltot = np.asarray(X.sum(1)).ravel().astype(np.float64); N = Xc.shape[0]
donors = obs.batch.astype(str).values; dis = obs.disease.astype(str).values
D = sorted(set(donors)); dmap = {d: i for i, d in enumerate(D)}; di = np.array([dmap[d] for d in donors])
dlab = np.array([pd.Series(dis[di == i]).value_counts().index[0] for i in range(len(D))])
case = a.case or ("COVID-19" if (dlab == "COVID-19").any() else [x for x in pd.unique(dlab) if x != "normal"][0])
cov = dlab == case; hea = dlab == "normal"
if a.dsub:
    rs = np.random.default_rng(100 + a.dsub)
    keepc = rs.choice(np.where(cov)[0], a.dsub, replace=False); keeph = rs.choice(np.where(hea)[0], a.dsub, replace=False)
    cov = np.isin(np.arange(len(dlab)), keepc); hea = np.isin(np.arange(len(dlab)), keeph)
Dh = np.zeros((len(D), N), np.float32); Dh[di, np.arange(N)] = 1
dcount = Dh.sum(1)

files = sorted(glob.glob(os.path.join(rdir, "lattice", "task_*.npz")))
ref_batch = "harmony" if meta.get("n_batches", 1) > 1 else "none"
tref = None
for f in files:
    z = np.load(f, allow_pickle=True)
    if int(z["nhvg"][0]) == 2000 and str(z["norm"][0]) == "log1p_cp10k": tref = dict(z); break
def ref_labels(ref):
    """Reference populations: default / alt_k30r12 / pearson clustering, or the authors' cell-type annotation."""
    if ref == "author":
        return pd.Categorical(obs.cell_type.astype(str)).codes.astype(int)
    t = tref; nc, k, r = 30, 15, 0.8
    if ref == "alt_k30r12": k, r = 30, 1.2
    if ref == "pearson":
        for f in files:
            z = np.load(f, allow_pickle=True)
            if int(z["nhvg"][0]) == 2000 and str(z["norm"][0]) == "pearson": t = dict(z); break
    m = (t["ncomp"] == nc) & (t["batch"] == ref_batch) & (t["k"] == k) & np.isclose(t["res"].astype(float), r)
    return t["labels"][np.where(m)[0][0], 0].astype(int)
lab_ref = ref_labels(a.ref)
sizes = np.bincount(lab_ref); C = [c for c in range(len(sizes)) if sizes[c] >= 50]
cm = np.stack([lab_ref == c for c in C]).astype(np.float32); csz = cm.sum(1)

def welch(x1, x0):
    m1, m0 = np.nanmean(x1, 0), np.nanmean(x0, 0); v1, v0 = np.nanvar(x1, 0, ddof=1), np.nanvar(x0, 0, ddof=1)
    n1, n0 = np.sum(~np.isnan(x1), 0), np.sum(~np.isnan(x0), 0)
    se = np.sqrt(v1 / n1 + v0 / n0); t = (m1 - m0) / se
    df = se ** 4 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    return t, p, m1 - m0

def bh(p):
    p = np.where(np.isfinite(p), p, 1.0); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

def claims_for(lab):
    k2 = lab.max() + 1; oh = np.zeros((k2, N), np.float32); oh[lab, np.arange(N)] = 1
    inter = cm @ oh.T; J = inter / (csz[:, None] + oh.sum(1)[None, :] - inter); b = J.argmax(1)
    T = np.full((len(C), len(gnames)), np.nan, np.float32); S = np.zeros((len(C), len(gnames)), np.int8)
    LF = np.full((len(C), len(gnames)), np.nan, np.float32); comp = np.full(len(C), np.nan, np.float32)
    for ci in range(len(C)):
        mask = (lab == b[ci]).astype(np.float32)
        nd = Dh @ mask
        frac = (nd + 0.5) / (dcount + 1); lg = np.log(frac / (1 - frac))
        comp[ci] = welch(lg[cov][:, None], lg[hea][:, None])[0][0]
        Sx = (Dh * mask[None, :]) @ Xc; Ld = (Dh * mask[None, :]) @ Ltot
        ok = nd >= a.min_cells
        if ok[cov].sum() < 3 or ok[hea].sum() < 3: continue
        cpm = np.log2(Sx / np.maximum(Ld, 1)[:, None] * 1e6 + 1); cpm[~ok] = np.nan
        t, p, lf = welch(cpm[cov], cpm[hea]); q = bh(p)
        T[ci], LF[ci] = t, lf; S[ci] = np.sign(t) * (q < 0.1)
    return T, S, LF, comp, J.max(1)

lat = dict(np.load(files[a.task], allow_pickle=True)); P = lat["labels"].shape[0]
Ts, Ss, LFs, Cs, Js = [], [], [], [], []
for p in range(P):
    T, S, LF, comp, J = claims_for(lat["labels"][p, 0].astype(int))
    Ts.append(T.astype(np.float16)); Ss.append(S); LFs.append(LF.astype(np.float16)); Cs.append(comp); Js.append(J)
    if p % 100 == 0: print("pipeline", p, "/", P, flush=True)
od = os.path.join(rdir, "cond" + ("" if a.ref == "default" else "_" + a.ref) + (f"_d{a.dsub}" if a.dsub else "")); os.makedirs(od, exist_ok=True)
np.savez_compressed(os.path.join(od, f"cond_{a.task:03d}.npz"), t=np.stack(Ts), sig=np.stack(Ss), lfc=np.stack(LFs),
    comp=np.stack(Cs), jac=np.stack(Js), nhvg=lat["nhvg"], norm=lat["norm"], ncomp=lat["ncomp"], batch=lat["batch"],
    k=lat["k"], res=lat["res"], genes=gnames, clusters=np.array(C),
    cluster_label=np.array([pd.Series(obs.cell_type.values[lab_ref == c]).value_counts().index[0] for c in C]))
print("done", a.task, "donors covid/healthy", int(cov.sum()), int(hea.sum()))
