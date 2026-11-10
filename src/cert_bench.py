"""Sequential certification of claim robustness with anytime-valid confidence sequences, evaluated on exhaustive truth.
Claim support s = P(claim holds | pipeline ~ Uniform(defensible lattice)). Classes: robust s>=0.8, fragile s<0.5, middle otherwise.
Methods (pipelines shared across claims; each claim stops independently):
  fixed_n          : n random pipelines, Clopper-Pearson (valid only at the pre-chosen n)
  seq_iid          : uniform sampling, predictable-plug-in empirical-Bernstein CS (Waudby-Smith & Ramdas 2023)
  seq_strat_<S>    : Sobol-guided stratified order over strata S (proportional allocation, evenly interleaved), same CS.
                     Theory: stopping time scales with the within-stratum variance, i.e. by (1 - S_closed(S)).
Usage: python cert_bench.py --kind marker --names pbmc_s0 [--T 1024]
"""
import argparse, os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import beta
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, FACTORS

ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["marker", "disease"])
ap.add_argument("--names", required=True); ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--T", type=int, default=1024); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--K", type=int, default=2000)
ap.add_argument("--target", default="support", choices=["support", "direction"]); ap.add_argument("--ref", default="default")
a = ap.parse_args(); rng = np.random.default_rng(0); LA = np.log(2 / a.alpha)
STRATA = {"nb": ["norm", "batch"], "br": ["batch", "res"], "nbr": ["norm", "batch", "res"], "n": ["norm"], "b": ["batch"], "r": ["res"]}
SEB = ["n", "b", "r", "nb", "br"]

def load_marker(name):
    od = os.path.join(RES, name, "fragility")
    tab = pd.read_csv(os.path.join(od, "claims_table.csv.gz")); A = np.load(os.path.join(od, "pop_auc.npy"))
    pp = pd.read_csv(os.path.join(od, "pop_pipelines.csv"))
    sel = np.where(tab.auc_L.values >= 0.7)[0]
    if len(sel) > a.K: sel = rng.choice(sel, a.K, replace=False)
    return (A[:, sel].astype(np.float32) >= 0.7), pp

def load_disease(name):
    SUF = "" if a.ref == "default" else "_" + a.ref
    rdir = os.path.join(RES, name)
    cl = pd.read_csv(os.path.join(rdir, "cond_summary" + SUF, "de_claims.csv.gz"))
    if len(cl) > a.K: cl = cl.sample(n=a.K, random_state=0)
    Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(rdir, "cond" + SUF, "cond_*.npz")))]
    genes = list(Z[0]["genes"].astype(str)); clusters = list(Z[0]["clusters"])
    ci = np.array([clusters.index(c) for c in cl.cluster]); gi = np.array([genes.index(g) for g in cl.gene]); sg = cl.direction.values
    if a.target == "support": X = np.concatenate([z["sig"][:, ci, gi] == sg[None, :] for z in Z])
    else: X = np.concatenate([np.sign(np.nan_to_num(z["t"][:, ci, gi].astype(np.float32))) == sg[None, :] for z in Z])
    pp = pd.DataFrame({f: np.concatenate([z[f] for z in Z]) for f in FACTORS})
    return X, pp

def classify(L, U):
    c = np.full(L.shape, -1); c[L >= 0.8] = 2; c[U < 0.5] = 0; c[(L >= 0.5) & (U < 0.8)] = 1; return c
def true_class(s): return np.where(s >= 0.8, 2, np.where(s < 0.5, 0, 1))

def run_seq(Xseq):
    T, K = Xseq.shape
    mu = np.full(K, 0.5); s2 = np.full(K, 0.25); sumy = np.zeros(K); sq = np.zeros(K)
    sl = np.zeros(K); sly = np.zeros(K); svp = np.zeros(K); L = np.zeros(K); U = np.ones(K)
    dec = np.full(K, -1); tdec = np.full(K, T + 1)
    for t in range(1, T + 1):
        x = Xseq[t - 1].astype(np.float64)
        lam = np.minimum(np.sqrt(2 * LA / (s2 * t * np.log(1 + t))), 0.5)
        v = 4 * (x - mu) ** 2; psi = (-np.log(1 - lam) - lam) / 4
        sl += lam; sly += lam * x; svp += v * psi
        c = sly / sl; w = (LA + svp) / sl
        L = np.maximum(L, c - w); U = np.minimum(U, c + w)
        sumy += x; mu = (0.5 + sumy) / (t + 1); sq += (x - mu) ** 2; s2 = (0.25 + sq) / (t + 1)
        cls = classify(np.clip(L, 0, 1), np.clip(U, 0, 1))
        new = (dec == -1) & (cls >= 0); dec[new] = cls[new]; tdec[new] = t
    return dec, tdec

def run_sEB(X, iv, H, T):
    """Stratified empirical-Bernstein confidence sequence (new).
    Each cycle draws one pipeline from every stratum (random order, uniform within stratum); the bet lambda is fixed
    within a cycle and the variance proxy is centred on the predictable WITHIN-stratum mean. The product
    prod exp(lambda_c (X_i - mu_{h_i}) - v_i psi(lambda_c)) is a supermartingale, so at cycle ends the CS covers
    sum_i lambda_i mu_{h_i} / sum_i lambda_i = mean of stratum means (= lattice support for a balanced lattice) exactly,
    while its width tracks the within-stratum variance Var(X)(1 - S_closed)."""
    K = X.shape[1]; members = [np.where(iv == j)[0] for j in range(H)]
    sums = np.zeros((H, K)); cnt = np.zeros(H); mh = np.full((H, K), 0.5)
    sqw = np.zeros(K); s2w = np.full(K, 0.25); sl = np.zeros(K); slx = np.zeros(K); svp = np.zeros(K)
    L = np.zeros(K); U = np.ones(K); dec = np.full(K, -1); tdec = np.full(K, T + 1); t = 0
    for c_ in range(T // H):
        tt = max(t, 1)
        lam = np.minimum(np.sqrt(2 * LA / (s2w * tt * np.log(1 + tt))), 0.5); psi = (-np.log(1 - lam) - lam) / 4
        for h in rng.permutation(H):
            x = X[rng.choice(members[h])].astype(np.float64)
            v = 4 * (x - mh[h]) ** 2
            sl += lam; slx += lam * x; svp += v * psi
            sums[h] += x; cnt[h] += 1; mh[h] = (0.5 + sums[h]) / (1 + cnt[h]); sqw += (x - mh[h]) ** 2; t += 1
        s2w = (0.25 + sqw) / (1 + t)
        cc = slx / sl; w = (LA + svp) / sl
        L = np.maximum(L, cc - w); U = np.minimum(U, cc + w)
        cls = classify(np.clip(L, 0, 1), np.clip(U, 0, 1))
        new = (dec == -1) & (cls >= 0); dec[new] = cls[new]; tdec[new] = t
    return dec, tdec

def strat_order(iv, w, T):
    pos = []
    for j, wj in enumerate(w):
        n = max(1, int(round(T * wj))); u = rng.random(); pos += [((i + u) / n, j) for i in range(n)]
    pos.sort(); seq = [j for _, j in pos][:T]
    while len(seq) < T: seq.append(rng.choice(len(w), p=w))
    members = [np.where(iv == j)[0] for j in range(len(w))]
    return np.array([rng.choice(members[j]) for j in seq])

rows, perclaim = [], []
for name in a.names.split(","):
    X, pp = (load_marker if a.kind == "marker" else load_disease)(name)
    P, K = X.shape; s = X.mean(0); tc = true_class(s); Vx = s * (1 - s)
    SD = {}
    for nm, fs in STRATA.items():
        lab = pp[fs].astype(str).agg("|".join, axis=1).values
        lb, iv = np.unique(lab, return_inverse=True); wh = np.bincount(iv, minlength=len(lb)) / P
        cmh = np.stack([X[iv == j].mean(0) for j in range(len(lb))])
        SD[nm] = (np.where(Vx > 0, (wh[:, None] * (cmh - s) ** 2).sum(0) / np.maximum(Vx, 1e-12), 0), iv, wh)
    for rep in range(a.reps):
        meths = {"seq_iid": rng.integers(0, P, a.T)}
        for k in STRATA: meths[f"seq_strat_{k}"] = strat_order(SD[k][1], SD[k][2], a.T)
        for k in SEB:
            Sk, iv_k, wh_k = SD[k]; H = len(wh_k)
            cmh = np.stack([X[iv_k == j].mean(0) for j in range(H)]); s_eq = cmh.mean(0); tc_eq = true_class(s_eq)
            dec, td = run_sEB(X, iv_k, H, a.T); okk = dec >= 0
            rows.append(dict(dataset=name, method=f"seq_sEB_{k}", rep=rep, decided=float(okk.mean()),
                             error=float((dec[okk] != tc_eq[okk]).mean()) if okk.any() else np.nan,
                             mean_runs=float(td[okk].mean()) if okk.any() else np.nan, median_runs=float(np.median(np.where(okk, td, a.T + 1)))))
            if rep < 5: perclaim.append(pd.DataFrame({"dataset": name, "method": f"seq_sEB_{k}", "rep": rep, "s": s,
                                                      **{f"S_{q}": SD[q][0] for q in STRATA}, "t": td, "dec": dec, "true": tc_eq}))
        for meth, idx in meths.items():
            dec, td = run_seq(X[idx]); ok = dec >= 0
            rows.append(dict(dataset=name, method=meth, rep=rep, decided=float(ok.mean()),
                             error=float((dec[ok] != tc[ok]).mean()) if ok.any() else np.nan,
                             mean_runs=float(td[ok].mean()) if ok.any() else np.nan, median_runs=float(np.median(np.where(ok, td, a.T + 1)))))
            if rep < 5: perclaim.append(pd.DataFrame({"dataset": name, "method": meth, "rep": rep, "s": s,
                                                      **{f"S_{k}": SD[k][0] for k in STRATA}, "t": td, "dec": dec, "true": tc}))
        for n in [8, 16, 32, 64, 128, 256, 512]:
            if n > a.T: continue
            idx = rng.integers(0, P, n); kk = X[idx].sum(0)
            lo = np.where(kk > 0, beta.ppf(a.alpha / 2, kk, n - kk + 1), 0.0); hi = np.where(kk < n, beta.ppf(1 - a.alpha / 2, kk + 1, n - kk), 1.0)
            dec = classify(lo, hi); ok = dec >= 0
            rows.append(dict(dataset=name, method=f"fixed_{n}", rep=rep, decided=float(ok.mean()),
                             error=float((dec[ok] != tc[ok]).mean()) if ok.any() else np.nan, mean_runs=n, median_runs=n))
    print(name, "K", K, "P", P, "classes", np.bincount(tc, minlength=3).tolist(), "median S", {k: round(float(np.median(v[0])), 3) for k, v in SD.items()}, flush=True)
df = pd.DataFrame(rows)
tag = f"{a.kind}_{a.target}" + ("" if a.ref == "default" else "_" + a.ref) + f"_T{a.T}" + "__" + a.names.replace(",", "+")
od = os.path.join(RES, "certification"); os.makedirs(od, exist_ok=True)
df.to_csv(os.path.join(od, f"bench_{tag}.csv"), index=False); pd.concat(perclaim).to_csv(os.path.join(od, f"perclaim_{tag}.csv.gz"), index=False)
print(df.groupby(["dataset", "method"])[["decided", "error", "mean_runs", "median_runs"]].mean().round(3).to_string())
