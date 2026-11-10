"""Aggregate disease claims over the exhaustive lattice: robustness of reference claims, exact Sobol of their t-statistics,
and a cross-validated fragility predictor using only the reference run + 4 cheap local re-clusterings.
Usage: python cond_anova.py --name pbmc_covid
"""
import argparse, os, sys, json, glob, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, levels_for, FACTORS

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ref", default="default")
ap.add_argument("--dsub", type=int, default=0); a = ap.parse_args()
SUF = ("" if a.ref == "default" else "_" + a.ref) + (f"_d{a.dsub}" if a.dsub else "")
rdir = os.path.join(RES, a.name); meta = json.load(open(os.path.join(dataset_dir(a.name), "meta.json"))); lv = levels_for(meta)
Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(rdir, "cond" + SUF, "cond_*.npz")))]
cat = lambda k: np.concatenate([z[k] for z in Z])
des = pd.DataFrame({f: cat(f) for f in FACTORS})
shape = tuple(len(lv[f]) for f in FACTORS)
idx = np.stack([pd.Index(lv[f]).get_indexer(des[f].astype(type(lv[f][0]))) for f in FACTORS], 1)
flat = np.ravel_multi_index(idx.T, shape); order = np.argsort(flat); assert len(np.unique(flat)) == np.prod(shape)
T = cat("t")[order].astype(np.float32); S = cat("sig")[order]; CO = cat("comp")[order]; LF = cat("lfc")[order].astype(np.float32)
genes = Z[0]["genes"].astype(str); clusters = Z[0]["clusters"]; clab = Z[0]["cluster_label"].astype(str)
D = des.iloc[order].reset_index(drop=True)
ref_batch = "harmony" if meta.get("n_batches", 1) > 1 else "none"
def row(nhvg, norm, nc, b, k, r):
    m = (D.nhvg == nhvg) & (D.norm == norm) & (D.ncomp == nc) & (D.batch == b) & (D.k == k) & np.isclose(D.res.astype(float), r)
    return int(np.where(m)[0][0])
iref = {"alt_k30r12": row(2000, "log1p_cp10k", 30, ref_batch, 30, 1.2), "pearson": row(2000, "pearson", 30, ref_batch, 15, 0.8)}.get(a.ref, row(2000, "log1p_cp10k", 30, ref_batch, 15, 0.8))
perts = [row(2000, "log1p_cp10k", 30, ref_batch, 15, 0.5), row(2000, "log1p_cp10k", 30, ref_batch, 15, 1.2),
         row(2000, "log1p_cp10k", 30, ref_batch, 30, 0.8), row(2000, "log1p_cp10k", 30, "none", 15, 0.8)]
Sref = S[iref]; ci, gi = np.where(Sref != 0)
print("reference DE claims:", len(ci), "in", len(set(ci)), "clusters")
sgn = Sref[ci, gi]
support = (S[:, ci, gi] == sgn[None, :]).mean(0)
any_sig = (S[:, ci, gi] != 0).mean(0)
flip = (S[:, ci, gi] == -sgn[None, :]).mean(0)
tt = T[:, ci, gi]; ok = np.isfinite(tt)
dir_cons = ((np.sign(tt) == sgn[None, :]) & ok).sum(0) / np.maximum(ok.sum(0), 1)          # same direction, any significance
lr = LF[iref, ci, gi]; ll = LF[:, ci, gi]
eff_ret = (((np.sign(ll) == sgn[None, :]) & (np.abs(ll) >= 0.5 * np.abs(lr)[None, :])) & np.isfinite(ll)).sum(0) / np.maximum(np.isfinite(ll).sum(0), 1)
nom = None
claims = pd.DataFrame({"cluster": clusters[ci], "cell_type": clab[ci], "gene": genes[gi], "direction": sgn,
                       "t_ref": T[iref, ci, gi], "lfc_ref": lr, "support": support, "any_sig": any_sig, "opposite_sig": flip,
                       "dir_consistency": dir_cons, "effect_retention": eff_ret})

def sobol_ST(Y):
    V = Y.var(); d = Y.ndim
    if V <= 1e-14: return np.full(d, np.nan), np.full(d, np.nan)
    S1 = np.array([Y.mean(axis=tuple(i for i in range(d) if i != j)).var() / V for j in range(d)])
    ST = np.array([Y.var(axis=j).mean() / V for j in range(d)])
    return S1, ST
st_rows = []
for k_, (c_, g_) in enumerate(zip(ci, gi)):
    y = np.nan_to_num(T[:, c_, g_], nan=0.0).reshape(shape)
    s1, st = sobol_ST(y)
    st_rows.append(dict(zip([f"ST_{f}" for f in FACTORS], st)) | dict(zip([f"S1_{f}" for f in FACTORS], s1)))
claims = pd.concat([claims, pd.DataFrame(st_rows)], axis=1)

# composition claims
cref = CO[iref]; csup = (np.sign(CO) == np.sign(cref)[None, :]) & (np.abs(CO) > 2.1)
comp = pd.DataFrame({"cluster": clusters, "cell_type": clab, "t_ref": cref, "sig_ref": np.abs(cref) > 2.1,
                     "support_same_sig": csup.mean(0), "frac_sig_any": (np.abs(CO) > 2.1).mean(0)})

# fragility predictor: reference + 4 local re-clusterings only
feat = pd.DataFrame({"abs_t": np.abs(claims.t_ref), "size_rank": pd.Series(claims.cluster).map(pd.Series(np.arange(len(clusters)), index=clusters)),
                     "pert_agree": np.mean([S[p, ci, gi] == sgn for p in perts], 0),
                     "pert_tmin": np.min([np.nan_to_num(T[p, ci, gi] * sgn, nan=0) for p in perts], 0),
                     "pert_tmean": np.mean([np.nan_to_num(T[p, ci, gi] * sgn, nan=0) for p in perts], 0)})
y = claims.support.values; grp = claims.cluster.values
res = {}
if len(set(grp)) >= 3 and len(y) >= 30:
    gkf = GroupKFold(n_splits=min(5, len(set(grp))))
    for nm, cols in (("abs_t_only", ["abs_t"]), ("ref+local", list(feat.columns))):
        pred = np.zeros(len(y))
        for tr, te in gkf.split(feat, y, grp):
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05).fit(feat.iloc[tr][cols], y[tr])
            pred[te] = m.predict(feat.iloc[te][cols])
        fr = (y < 0.5).astype(int)
        res[nm] = dict(spearman=float(spearmanr(pred, y).correlation),
                       auroc_fragile=float(roc_auc_score(fr, -pred)) if 0 < fr.sum() < len(fr) else None)
od = os.path.join(rdir, "cond_summary" + SUF); os.makedirs(od, exist_ok=True)
claims.to_csv(os.path.join(od, "de_claims.csv.gz"), index=False); comp.to_csv(os.path.join(od, "composition_claims.csv"), index=False)
summ = dict(n_ref_de_claims=int(len(claims)), frac_robust_ge_0_9=float((claims.support >= 0.9).mean()),
            frac_fragile_lt_0_5=float((claims.support < 0.5).mean()), frac_ever_opposite_sig=float((claims.opposite_sig > 0).mean()),
            median_support=float(claims.support.median()),
            median_dir_consistency=float(claims.dir_consistency.median()), frac_dir_consistency_ge_0_95=float((claims.dir_consistency >= 0.95).mean()),
            frac_dir_consistency_lt_0_8=float((claims.dir_consistency < 0.8).mean()),
            median_effect_retention=float(claims.effect_retention.median()), frac_effect_retention_ge_0_8=float((claims.effect_retention >= 0.8).mean()),
            median_ST=claims[[f"ST_{f}" for f in FACTORS]].median().round(3).to_dict(),
            median_S1=claims[[f"S1_{f}" for f in FACTORS]].median().round(3).to_dict(),
            by_cell_type=claims.groupby("cell_type").support.agg(["count", "median"]).round(3).to_dict(),
            composition=comp.round(3).to_dict(orient="records"), fragility_prediction=res)
json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in summ.items() if k not in ("composition", "by_cell_type")}, indent=1, default=str))
