"""Fragility prediction for disease DE claims, evaluated leave-one-cohort-out (and pooled GroupKFold by cluster).
Targets (exhaustive multiverse): significance support and direction consistency.
Predictors (cost in full pipeline runs):
  abs_t (1)            : reference |t| only
  boot (1)             : donor-bootstrap stability on the reference pipeline (no extra pipelines)
  designed4 (5)        : reference + 4 theory/Sobol-designed pipelines varying the dominant DE factors (batch correction, resolution)
  model_designed (5)   : gradient boosting on reference |t|, |lfc|, bootstrap and designed-pipeline features
  random4 / random8    : support among 4 / 8 random defensible pipelines (mini-multiverse baselines)
Usage: python cond_fragility.py --names pbmc_covid,pbmc_covid40 [--ref default]
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, levels_for, FACTORS

ap = argparse.ArgumentParser(); ap.add_argument("--names", default="pbmc_covid,pbmc_covid40"); ap.add_argument("--ref", default="default")
a = ap.parse_args(); SUF = "" if a.ref == "default" else "_" + a.ref; rng = np.random.default_rng(0)
frames = []
for name in a.names.split(","):
    rdir = os.path.join(RES, name); meta = json.load(open(os.path.join(dataset_dir(name), "meta.json"))); lv = levels_for(meta)
    Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(rdir, "cond" + SUF, "cond_*.npz")))]
    cat = lambda k: np.concatenate([z[k] for z in Z])
    D = pd.DataFrame({f: cat(f) for f in FACTORS}); T = cat("t").astype(np.float32); S = cat("sig"); LF = cat("lfc").astype(np.float32)
    cl = pd.read_csv(os.path.join(rdir, "cond_summary" + SUF, "de_claims_boot.csv.gz"))
    genes = list(Z[0]["genes"].astype(str)); clusters = list(Z[0]["clusters"])
    ci = np.array([clusters.index(c) for c in cl.cluster]); gi = np.array([genes.index(g) for g in cl.gene]); sgn = cl.direction.values
    rb = "harmony" if meta.get("n_batches", 1) > 1 else "none"
    def row(norm, b, k, r, nc=30, nh=2000):
        m = (D.nhvg == nh) & (D.norm == norm) & (D.ncomp == nc) & (D.batch == b) & (D.k == k) & np.isclose(D.res.astype(float), r)
        return int(np.where(m)[0][0])
    rn = "pearson" if a.ref == "pearson" else "log1p_cp10k"; kk, rr = (30, 1.2) if a.ref == "alt_k30r12" else (15, 0.8)
    iref = row(rn, rb, kk, rr)
    alt_b = [b for b in ["none", "combat"] if b != rb][:1] + [b for b in ["batch_scale", "regress_covariates"] if b != rb][:1]
    design = [row(rn, b, kk, rr) for b in alt_b] + [row(rn, rb, kk, 0.5), row(rn, rb, kk, 2.0)]
    st = lambda rows: (S[rows][:, ci, gi] == sgn[None, :])
    dt = lambda rows: np.nan_to_num(T[rows][:, ci, gi] * sgn[None, :], nan=0.0)
    allr = np.arange(len(D))
    r4 = np.mean([st(rng.choice(allr, 4, replace=False)).mean(0) for _ in range(1)], 0)
    r8 = st(rng.choice(allr, 8, replace=False)).mean(0)
    f = pd.DataFrame({"cohort": name, "cluster": cl.cluster, "abs_t": np.abs(cl.t_ref), "abs_lfc": np.abs(cl.lfc_ref),
        "boot": cl.boot_support, "boot_dir": cl.boot_dir_consistency,
        "designed4": st([iref] + design).mean(0), "des_tmin": dt(design).min(0), "des_tmean": dt(design).mean(0),
        "des_dir": (dt(design) > 0).mean(0), "random4": r4, "random8": r8,
        "y_support": cl.support, "y_dir": cl.dir_consistency})
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
FEAT = ["abs_t", "abs_lfc", "boot", "boot_dir", "designed4", "des_tmin", "des_tmean", "des_dir"]
res = []
for target, frag in (("y_support", lambda y: y < 0.5), ("y_dir", lambda y: y < 0.8)):
    pred_model = np.full(len(df), np.nan)
    cohorts = df.cohort.unique()
    for c in cohorts:
        tr, te = df.cohort != c, df.cohort == c
        if tr.sum() < 20: continue
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05).fit(df.loc[tr, FEAT], df.loc[tr, target])
        pred_model[te.values] = m.predict(df.loc[te, FEAT])
    preds = {"abs_t (1 run)": df.abs_t.values, "bootstrap (1 run)": (df.boot if target == "y_support" else df.boot_dir).values,
             "designed4 (5 runs)": (df.designed4 if target == "y_support" else df.des_dir).values,
             "model_designed (5 runs)": pred_model, "random4 (4 runs)": df.random4.values, "random8 (8 runs)": df.random8.values}
    for c in list(cohorts) + ["ALL"]:
        sel = (df.cohort == c).values if c != "ALL" else np.ones(len(df), bool)
        y = df[target].values[sel]; fr = frag(y).astype(int)
        for nm, p in preds.items():
            p = p[sel]; ok = np.isfinite(p)
            if ok.sum() < 10: continue
            res.append(dict(target=target, cohort=c, predictor=nm, n=int(ok.sum()), frac_fragile=float(fr[ok].mean()),
                            spearman=float(spearmanr(p[ok], y[ok]).correlation),
                            auroc_fragile=float(roc_auc_score(fr[ok], -p[ok])) if 0 < fr[ok].sum() < ok.sum() else np.nan))
res = pd.DataFrame(res)
od = os.path.join(RES, "cond_fragility" + SUF); os.makedirs(od, exist_ok=True)
res.to_csv(os.path.join(od, "evaluation.csv"), index=False); df.to_csv(os.path.join(od, "features.csv.gz"), index=False)
print(res.pivot_table(index=["target", "cohort"], columns="predictor", values="auroc_fragile").round(3).to_string())
print(res.pivot_table(index=["target", "cohort"], columns="predictor", values="spearman").round(3).to_string())
