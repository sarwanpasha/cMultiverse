"""Fragility predictor: predict a claim's multiverse support from ONE analysis, evaluated leave-one-dataset-family-out
on exhaustive multiverses. Feature tiers:
  T0 margin       : reference AUC only
  T1 descriptive  : + zero fractions, detection, size, specificity, hvg, expression
  T2 theory       : + Theorem-2 zero-tie term and factorized AUCs under Pearson / poscounts / zero-tied values (no rerun)
  T3 local        : + 4 local re-clusterings of the same embedding (no new normalization/PCA)
Baselines: mini-multiverse of 4 / 8 random full pipelines.
Usage: python fragility_model.py
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
sys.path.insert(0, os.path.dirname(__file__))
from common import RES
fam = lambda d: "sim" if d.startswith("sim") else ("pbmc" if d.startswith("pbmc") else d)
dfs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "*", "fragility", "claims_table.csv.gz")))]
df = pd.concat(dfs, ignore_index=True); df["family"] = df.dataset.map(fam)
df["dP"] = df.auc_P - df.auc_L; df["dQ"] = df.auc_Q - df.auc_L; df["dPzt"] = df.auc_Pzt - df.auc_L
df["pert_drop"] = df.auc_L - df.pert_auc_min
T0 = ["auc_L", "margin"]
T1 = T0 + ["zA", "zB", "det", "size_frac", "specificity", "second_auc", "is_hvg", "meanlog", "depth_auc"]
T2 = T1 + ["pred00", "D0", "dP", "dQ", "dPzt", "auc_P", "auc_Q"]
T3 = T2 + ["pert_auc_min", "pert_auc_mean", "pert_jac_min", "pert_drop"]
T4 = T3 + ["design_support", "design_auc_min", "design_auc_mean"]
TIERS = {"T0_margin": T0, "T1_descriptive": T1, "T2_theory": T2, "T3_theory+local": T3, "T4_theory+designed4": T4}
out = {"n_rows": int(len(df)), "families": df.family.value_counts().to_dict()}
preds = {}
for nm, cols in TIERS.items():
    p = np.zeros(len(df))
    for f in df.family.unique():
        tr, te = df.family != f, df.family == f
        sub = df[tr].sample(n=min(200000, int(tr.sum())), random_state=0)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63).fit(sub[cols], sub.support)
        p[te.values] = m.predict(df.loc[te, cols])
    preds[nm] = np.clip(p, 0, 1)
preds["mini_multiverse_4"] = df.rand4_support.values; preds["mini_multiverse_8"] = df.rand8_support.values
preds["designed4_raw"] = df.design_support.values
ev = []
for f in list(df.family.unique()) + ["ALL"]:
    sel = (df.family == f) if f != "ALL" else np.ones(len(df), bool)
    refc = sel & (df.auc_L >= 0.7)               # claims a practitioner would report
    y = df.support.values; fr = (y < 0.5).astype(int)
    for nm, p in preds.items():
        r = dict(family=f, model=nm, n=int(refc.sum()),
                 spearman=float(spearmanr(p[refc], y[refc]).correlation), mae=float(mean_absolute_error(y[refc], p[refc])))
        r["auroc_fragile"] = float(roc_auc_score(fr[refc], -p[refc])) if 0 < fr[refc].sum() < refc.sum() else np.nan
        r["frac_fragile"] = float(fr[refc].mean())
        ev.append(r)
ev = pd.DataFrame(ev)
od = os.path.join(RES, "fragility_model"); os.makedirs(od, exist_ok=True)
ev.to_csv(os.path.join(od, "evaluation.csv"), index=False)
# feature importance proxy: drop-tier gains and permutation on pooled model
pd.DataFrame({k: v for k, v in preds.items()}).assign(dataset=df.dataset, family=df.family, support=df.support,
    auc_L=df.auc_L).to_csv(os.path.join(od, "predictions.csv.gz"), index=False)
print(ev.pivot_table(index=["family"], columns="model", values="spearman").round(3).to_string())
print(ev.pivot_table(index=["family"], columns="model", values="auroc_fragile").round(3).to_string())
print(ev[ev.model == "T0_margin"][["family", "n", "frac_fragile"]].to_string())
