"""Ground-truth evaluation of single pipelines vs multiverse consensus, the zero-tied Pearson fix,
and finite-sample robustness certificates.
Usage: python ground_truth.py --name cite_pbmc10k
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, FACTORS

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tau", type=float, default=0.7); ap.add_argument("--quick", action="store_true")
a = ap.parse_args(); rng = np.random.default_rng(0)
rdir = os.path.join(RES, a.name); meta = json.load(open(os.path.join(dataset_dir(a.name), "meta.json")))
SUF = "_quick" if a.quick else ""
Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(rdir, "claims" + SUF, "claims_*.npz")))]
z0 = Z[0]; vn = list(z0["vnorms"]); set_names = list(z0["set_names"]); pg = z0["panel_genes"].astype(str)
cat = lambda k: np.concatenate([z[k] for z in Z])
D = pd.DataFrame({f: cat(f) for f in FACTORS})
P = len(D)
vi_nat = np.array([vn.index(n) for n in D.norm])
rec = cat("disc_prec")                     # P x S x V x sets (recall@50 of known-true markers)
has_truth = [i for i, s in enumerate(z0["true_sets"]) if len(s) > 0]
nat_rec = np.nanmean(rec[np.arange(P), :, vi_nat][:, :, has_truth], axis=(1, 2))   # P
out = {"n_pipelines": int(P)}

# default scanpy-like pipeline
dflt = dict(nhvg=2000, norm="log1p_cp10k", ncomp=30, batch=("harmony" if meta["n_batches"] > 1 else "none"), k=15, res=0.8)
m = np.ones(P, bool)
for f, v in dflt.items(): m &= (D[f].astype(str).values == str(v))
out["default_pipeline_recall50"] = float(nat_rec[m][0]) if m.any() else None
out["single_pipeline_recall50"] = dict(median=float(np.median(nat_rec)), q10=float(np.quantile(nat_rec, .1)),
                                       q90=float(np.quantile(nat_rec, .9)), best=float(nat_rec.max()))
out["recall50_by_norm"] = D.assign(r=nat_rec).groupby("norm").r.median().round(4).to_dict()

# zero-tied fix: Pearson pipelines re-scored with pearson_zt values at the same partition
if "pearson_zt" in vn:
    pm = D.norm.values == "pearson"
    r_zt = np.nanmean(rec[pm][:, :, vn.index("pearson_zt")][:, :, has_truth], axis=(1, 2))
    out["pearson_pipelines_recall50"] = dict(original=float(np.median(nat_rec[pm])), zero_tied=float(np.median(r_zt)),
                                             frac_improved=float(np.mean(r_zt > nat_rec[pm])))

# multiverse consensus: mean AUC over all pipelines (natural value normalization), top-50 per population
tasks_norm = [str(z["norm"][0]) for z in Z]
cons = sum(z["auc_sum"][vn.index(n)] for z, n in zip(Z, tasks_norm)) / sum(int(z["n_acc"]) for z in Z)
cons_freq = sum(z["top_cnt"][vn.index(n)] for z, n in zip(Z, tasks_norm))
true_sets = [set(s.split(";")) - {""} for s in z0["true_sets"]]
gidx = {g: i for i, g in enumerate(pg)}
def recall_from_scores(S):
    r = []
    for i in has_truth:
        top = set(pg[np.argsort(-S[i])[:50]]); r.append(len(top & true_sets[i]) / len(true_sets[i]))
    return float(np.mean(r))
out["consensus_recall50_meanAUC"] = recall_from_scores(cons)
out["consensus_recall50_topfreq"] = recall_from_scores(cons_freq.astype(float))
out["consensus_beats_frac_of_pipelines"] = float(np.mean(out["consensus_recall50_meanAUC"] >= nat_rec))

# marker-claim certificates on the pair list
_m = cat("marker_auc"); _m[~np.isfinite(_m)] = np.nan
mauc = np.nan_to_num(np.nanmean(_m[np.arange(P), :, vi_nat], axis=1), nan=0.5)      # P x pairs (seed-avg, natural)
support = (mauc >= a.tau).mean(0)
pairs = pd.DataFrame({"gene": z0["pair_gene"], "set": z0["pair_set"], "support": support,
                      "mean_auc": mauc.mean(0), "default_auc": (mauc[m][0] if m.any() else np.nan)})
if len(z0["pair_truth"]):
    pairs["truth_auc"] = z0["pair_truth"]; y = (pairs.truth_auc > 0.75).astype(int).values
    if 0 < y.sum() < len(y):
        per_pipe = np.array([roc_auc_score(y, mauc[p]) for p in range(P)])
        out["cite_truth_AUROC"] = dict(single_median=float(np.median(per_pipe)), single_q10=float(np.quantile(per_pipe, .1)),
            default=float(roc_auc_score(y, pairs.default_auc)) if m.any() else None,
            consensus_meanAUC=float(roc_auc_score(y, pairs.mean_auc)), certificate_support=float(roc_auc_score(y, support)))
# finite-sample certificate coverage: Clopper-Pearson lower bound from n random pipelines vs exhaustive support
cov = {}
for n in [50, 100, 200, 500]:
    hits = []
    for rep in range(200):
        s = rng.choice(P, n, replace=False); k = (mauc[s] >= a.tau).sum(0)
        lo = np.where(k > 0, beta.ppf(0.025, k, n - k + 1), 0.0); hi = np.where(k < n, beta.ppf(0.975, k + 1, n - k), 1.0)
        hits.append(np.mean((support >= lo) & (support <= hi)))
    cov[n] = float(np.mean(hits))
out["certificate_coverage_95"] = cov
od = os.path.join(rdir, "ground_truth" + SUF); os.makedirs(od, exist_ok=True)
pairs.to_csv(os.path.join(od, "pair_certificates.csv"), index=False)
D.assign(recall50=nat_rec).to_csv(os.path.join(od, "pipeline_recall.csv.gz"), index=False)
json.dump(out, open(os.path.join(od, "summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
