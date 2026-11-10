"""Exact functional-ANOVA (Sobol) indices on the complete factorial lattice, plus the
direct / mediated decomposition of normalization effects on marker claims.
Usage: python anova.py --name pbmc_s0
"""
import argparse, os, sys, json, glob, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, levels_for, FACTORS, NORMS, NORM_CLASS, apply_quick

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--quick", action="store_true")
ap.add_argument("--extra_batch", default=None, help="merge claims_<extra> lattice as an additional batch level")
a = ap.parse_args()
SUF = "_quick" if a.quick else ("_" + a.extra_batch if a.extra_batch else "")
rdir = os.path.join(RES, a.name); meta = json.load(open(os.path.join(dataset_dir(a.name), "meta.json")))
lv = levels_for(meta)
if a.quick: lv = apply_quick(lv)
if a.extra_batch:
    lv["batch"] = lv["batch"] + [a.extra_batch]
    files = sorted(glob.glob(os.path.join(rdir, "claims", "claims_*.npz"))) + sorted(glob.glob(os.path.join(rdir, "claims_" + a.extra_batch, "claims_*.npz")))
else:
    files = sorted(glob.glob(os.path.join(rdir, "claims" + SUF, "claims_*.npz")))
Z = [np.load(f, allow_pickle=True) for f in files]
def cat(key): return np.concatenate([z[key] for z in Z])
design = pd.DataFrame({f: cat(f) for f in FACTORS})
shape = tuple(len(lv[f]) for f in FACTORS)
idx = np.stack([pd.Index(lv[f]).get_indexer(design[f].astype(type(lv[f][0]))) for f in FACTORS], 1)
assert (idx >= 0).all(), "design levels mismatch"
flat = np.ravel_multi_index(idx.T, shape)
assert len(np.unique(flat)) == np.prod(shape), f"incomplete lattice: {len(np.unique(flat))} of {np.prod(shape)}"
order = np.argsort(flat)
z0 = Z[0]; set_names = list(z0["set_names"]); vnorms = list(z0["vnorms"])
pair_gene, pair_set, pair_truth = z0["pair_gene"], z0["pair_set"], z0["pair_truth"]

def grid(arr):  # arr: pipelines x ... -> factor grid x ...
    arr = arr[order]
    return arr.reshape(shape + arr.shape[1:])

def sobol(Y, names):
    """Exact first-order, total-order and pure pairwise indices of a full-factorial grid Y (uniform design)."""
    V = Y.var()
    if not np.isfinite(V) or V <= 1e-14: return None
    d = len(names); res = {}
    for j in range(d):
        others = tuple(i for i in range(d) if i != j)
        S1 = Y.mean(axis=others).var() / V
        ST = Y.var(axis=j).mean() / V
        res[names[j]] = (S1, ST)
    pw = {}
    for i, j in itertools.combinations(range(d), 2):
        others = tuple(t for t in range(d) if t not in (i, j))
        closed = Y.mean(axis=others).var() / V
        pw[(names[i], names[j])] = closed - res[names[i]][0] - res[names[j]][0]
    return V, res, pw

n_cl = grid(cat("n_clusters").astype(float)); ari = grid(cat("ari")); nmi = grid(cat("nmi"))
jac = grid(cat("jaccard")); _m = cat("marker_auc"); _m[~np.isfinite(_m)] = np.nan; mauc = grid(_m); dprec = grid(cat("disc_prec"))
S = n_cl.shape[6]
norm_ax = FACTORS.index("norm")
# natural marker claim: value normalization == pipeline normalization
vpos = [vnorms.index(n) for n in lv["norm"]]
def natural(X):  # X grid (... , S, V, C) -> (..., S, C)
    out = np.empty(X.shape[:7] + X.shape[8:], X.dtype)
    for ni, vi in enumerate(vpos):
        sl = [slice(None)] * 7; sl[norm_ax] = ni
        out[tuple(sl)] = X[tuple(sl) + (vi,)]
    return out
mauc_nat = np.nan_to_num(natural(mauc), nan=0.5); dprec_nat = natural(dprec)

claims = [("n_clusters", "partition", n_cl), ("ari", "partition", ari), ("nmi", "partition", nmi)]
for i, s in enumerate(set_names): claims.append((f"recover:{s}", "recovery", jac[..., i]))
for i in range(len(pair_gene)): claims.append((f"marker:{pair_gene[i]}|{pair_set[i]}", "marker", mauc_nat[..., i]))
if np.isfinite(dprec_nat).any():
    for i, s in enumerate(set_names): claims.append((f"discovery:{s}", "discovery", dprec_nat[..., i]))

VARIANTS = {"full": {}}
for n in lv["norm"]: VARIANTS[f"drop_{n}"] = {"norm": [x for x in lv["norm"] if x != n]}
VARIANTS["defensible"] = {"k": [x for x in lv["k"] if x <= 50], "res": [x for x in lv["res"] if x <= 1.2],
                          "nhvg": [x for x in lv["nhvg"] if 1000 <= x <= 4000]}
def subgrid(Y, spec):
    for f, keep in spec.items():
        ax = FACTORS.index(f); ii = [lv[f].index(x) for x in keep]
        Y = np.take(Y, ii, axis=ax)
    return Y

rows, prow, noise = [], [], []
for cname, ctype, Y in claims:
    Yseed = Y  # factors..., S
    nf = Yseed.var(axis=-1).mean() / max(Yseed.var(), 1e-14)   # Leiden-seed noise share
    noise.append((cname, ctype, nf))
    for vname, spec in VARIANTS.items():
        Ym = subgrid(Yseed.mean(-1), spec)
        out = sobol(Ym, FACTORS)
        if out is None: continue
        V, res, pw = out
        for f, (s1, st) in res.items(): rows.append((cname, ctype, vname, f, s1, st, V))
        if vname == "full":
            for (f1, f2), v in pw.items(): prow.append((cname, ctype, f1, f2, v))
    if True:
        for s in range(S):  # per-seed indices for uncertainty
            out = sobol(Yseed[..., s], FACTORS)
            if out:
                for f, (s1, st) in out[1].items(): rows.append((cname, ctype, f"seed{s}", f, s1, st, out[0]))

od = os.path.join(rdir, "anova" + SUF); os.makedirs(od, exist_ok=True)
sob = pd.DataFrame(rows, columns=["claim", "type", "variant", "factor", "S1", "ST", "var"])
sob.to_csv(os.path.join(od, "sobol.csv"), index=False)
pd.DataFrame(prow, columns=["claim", "type", "f1", "f2", "Vpure"]).to_csv(os.path.join(od, "pairwise.csv"), index=False)
pd.DataFrame(noise, columns=["claim", "type", "seed_noise_share"]).to_csv(os.path.join(od, "noise.csv"), index=False)

# ---------------- mediation: extended grid with value-normalization axis
lattice_norms = lv["norm"]
vsel = [vnorms.index(n) for n in lattice_norms]
med = []
ext_names = FACTORS + ["vnorm"]
for i in range(len(pair_gene)):
    Y = np.nan_to_num(mauc[..., i], nan=0.5).mean(axis=6)[..., vsel]   # factors..., V(lattice norms)
    out = sobol(Y, ext_names)
    if out is None: continue
    V, res, pw = out
    nat = sobol(mauc_nat[..., i].mean(-1), FACTORS)
    med.append(dict(claim=f"marker:{pair_gene[i]}|{pair_set[i]}", direct_S1=res["vnorm"][0], direct_ST=res["vnorm"][1],
                    mediated_S1=res["norm"][0], mediated_ST=res["norm"][1], interaction=pw[("norm", "vnorm")],
                    natural_norm_ST=(nat[1]["norm"][1] if nat else np.nan), truth=(pair_truth[i] if len(pair_truth) else np.nan)))
pd.DataFrame(med).to_csv(os.path.join(od, "mediation.csv"), index=False)

# ---------------- Theorem 1 check: class-L value normalizations give identical AUCs at a fixed partition
Lcls = [vnorms.index(n) for n in vnorms if NORM_CLASS.get(n) == "L"]
dev = float(np.nanmax(np.abs(mauc[..., Lcls, :] - mauc[..., Lcls[:1], :]))) if len(Lcls) > 1 else np.nan
json.dump({"max_abs_AUC_diff_within_classL": dev, "n_pipelines": int(np.prod(shape)), "n_claims": len(claims)},
          open(os.path.join(od, "theorem1.json"), "w"), indent=1)
summ = sob[sob.variant == "full"].groupby(["type", "factor"])[["S1", "ST"]].median().unstack().round(3)
print(summ.to_string()); print("theorem1 max dev", dev)
print(pd.DataFrame(med).drop(columns="claim").median(numeric_only=True).round(3).to_string() if med else "")
