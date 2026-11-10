"""Claim universe for the fragility predictor.
Claims: (gene g, cluster c of the REFERENCE run) -- 'g is a marker of c' -- exactly what a practitioner holds after one analysis.
Targets: support across M pipelines sampled uniformly from the exhaustive lattice (c matched by best Jaccard in each pipeline).
Features: computed from the reference run only (+ factorized re-scoring under other value normalizations, + 4 cheap local
re-clusterings of the same embedding).
Usage: python claims_universe.py --name pbmc_s0 --M 600
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, load_counts, auc_matrix, levels_for, NORMS, VALUE_NORMS

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--M", type=int, default=600)
ap.add_argument("--tau", type=float, default=0.7); ap.add_argument("--min_size", type=int, default=30)
a = ap.parse_args(); rng = np.random.default_rng(0)
rdir = os.path.join(RES, a.name); vd = os.path.join(rdir, "values")
X, obs, genes, meta = load_counts(a.name); lv = levels_for(meta)
pz = np.load(os.path.join(vd, "panel.npz"), allow_pickle=True); panel = pz["panel"]; pg = pz["genes"].astype(str)
L = pz["depth"].astype(float); N = len(L); G = len(panel)
R = {v: np.load(os.path.join(vd, f"ranks_{v}.npy")) for v in VALUE_NORMS}
logv = np.load(os.path.join(vd, "values_log1p_cp10k.npy"))
Z0 = (X[:, panel].toarray() == 0)

files = sorted(glob.glob(os.path.join(rdir, "lattice", "task_*.npz")))
TASKS = [dict(np.load(f, allow_pickle=True)) for f in files]   # materialize once (npz members re-decompress on access)
ref_batch = "harmony" if meta.get("n_batches", 1) > 1 else "none"
def find(t, nc, b, k, r):
    m = (t["ncomp"] == nc) & (t["batch"] == b) & (t["k"] == k) & (np.isclose(t["res"].astype(float), r))
    return int(np.where(m)[0][0])
tref = [t for t in TASKS if int(t["nhvg"][0]) == 2000 and str(t["norm"][0]) == "log1p_cp10k"][0]
iref = find(tref, 30, ref_batch, 15, 0.8)
lab_ref = tref["labels"][iref, 0].astype(int)
pert = [tref["labels"][find(tref, 30, ref_batch, 15, 0.5), 0], tref["labels"][find(tref, 30, ref_batch, 15, 1.2), 0],
        tref["labels"][find(tref, 30, ref_batch, 30, 0.8), 0], tref["labels"][iref, 1]]
hvg_ref = set(tref["hvg"].tolist())
# theory-designed mini-multiverse: vary only the steps theory/Sobol flag (normalization class, batch correction)
def task_of(nhvg, norm): return [t for t in TASKS if int(t["nhvg"][0]) == nhvg and str(t["norm"][0]) == norm][0]
design = []
for nv in ["pearson", "log1p_poscounts"]:
    tt = task_of(2000, nv); design.append((nv, tt["labels"][find(tt, 30, ref_batch, 15, 0.8), 0], nv))
alt_b = [b for b in ["none", "combat", "regress_covariates", "batch_scale", "harmony"] if b in lv["batch"] and b != ref_batch][:2]
for b in alt_b:
    design.append(("batch_" + b, tref["labels"][find(tref, 30, b, 15, 0.8), 0], "log1p_cp10k"))
K = lab_ref.max() + 1; sizes = np.bincount(lab_ref, minlength=K)
C = [c for c in range(K) if sizes[c] >= a.min_size]
cm = np.stack([lab_ref == c for c in C]).astype(np.float32); csz = cm.sum(1)

def match(lab):
    k2 = lab.max() + 1; oh = np.zeros((k2, N), np.float32); oh[lab, np.arange(N)] = 1
    inter = cm @ oh.T; J = inter / (csz[:, None] + oh.sum(1)[None, :] - inter)
    return J.argmax(1), J.max(1)

# ---------------- reference-run features
A_ref = {v: np.nan_to_num(auc_matrix(R[v], lab_ref), nan=0.5) for v in VALUE_NORMS}
AL = A_ref["log1p_cp10k"]
second = np.sort(AL, axis=0)[-2]                       # second-best cluster AUC per gene
zA = np.stack([Z0[lab_ref == c].mean(0) for c in C]); zall = Z0.mean(0)
nA = csz[:, None]; zB = (zall[None, :] * N - zA * nA) / (N - nA)
D0 = np.full((len(C), G), 0.5)
for j in range(G):
    zz = Z0[:, j]
    if zz.sum() < 2: continue
    r = rankdata(-L[zz]); lz = lab_ref[zz]
    for ci, c in enumerate(C):
        m = lz == c; n1 = m.sum(); n0 = len(lz) - n1
        if n1 and n0: D0[ci, j] = (r[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
pred00 = zA * zB * (D0 - 0.5)
meanlog = np.stack([logv[lab_ref == c].mean(0) for c in C])
depth_auc = np.array([(rankdata(L)[lab_ref == c].sum() - s * (s + 1) / 2) / (s * (N - s)) for c, s in zip(C, csz)])
pa, pj = [], []
for pl in pert:
    b, J = match(pl.astype(int)); Ap = np.nan_to_num(auc_matrix(R["log1p_cp10k"], pl.astype(int)), nan=0.5)
    pa.append(Ap[b]); pj.append(J)
pa = np.stack(pa); pj = np.stack(pj)
des_auc = {}
for nm, dl, vn in design:
    b, J = match(dl.astype(int)); des_auc[nm] = np.nan_to_num(auc_matrix(R[vn], dl.astype(int)), nan=0.5)[b]
des_stack = np.stack(list(des_auc.values()))

# ---------------- multiverse targets (uniform sample of the exhaustive lattice)
allrows = [(ti, ri, s) for ti, t in enumerate(TASKS) for ri in range(t["labels"].shape[0]) for s in range(t["labels"].shape[1])]
pick = rng.choice(len(allrows), size=min(a.M, len(allrows)), replace=False)
aucs = np.zeros((len(pick), len(C), G), np.float16); jacs = np.zeros((len(pick), len(C)), np.float32)
for i, p in enumerate(pick):
    ti, ri, s = allrows[p]; t = TASKS[ti]; lab = t["labels"][ri, s].astype(int)
    b, J = match(lab); A = np.nan_to_num(auc_matrix(R[str(t["norm"][ri])], lab), nan=0.5)
    aucs[i] = A[b]; jacs[i] = J
    if i % 100 == 0: print("pipe", i, flush=True)
aucs = aucs.astype(np.float32)
rand4 = (aucs[:4] >= a.tau).mean(0); rand8 = (aucs[:8] >= a.tau).mean(0)
aucs = aucs[8:]   # held-out pipelines define the target
support = (aucs >= a.tau).mean(0); mean_auc = aucs.mean(0); sd_auc = aucs.std(0)

rows = []
ci_idx, g_idx = np.meshgrid(np.arange(len(C)), np.arange(G), indexing="ij")
df = pd.DataFrame({"dataset": a.name, "cluster": np.array(C)[ci_idx.ravel()], "gene": pg[g_idx.ravel()],
    "auc_L": AL[C][ci_idx, g_idx].ravel() if False else AL[np.array(C)][:, :].ravel(),
    "auc_P": A_ref["pearson"][np.array(C)].ravel(), "auc_Q": A_ref["log1p_poscounts"][np.array(C)].ravel(),
    "auc_Pzt": A_ref["pearson_zt"][np.array(C)].ravel(),
    "second_auc": np.broadcast_to(second, (len(C), G)).ravel(), "zA": zA.ravel(), "zB": zB.ravel(), "D0": D0.ravel(),
    "pred00": pred00.ravel(), "meanlog": meanlog.ravel(), "det": np.broadcast_to(1 - zall, (len(C), G)).ravel(),
    "size": np.repeat(csz, G), "size_frac": np.repeat(csz / N, G), "depth_auc": np.repeat(depth_auc, G),
    "is_hvg": np.broadcast_to(np.isin(panel, list(hvg_ref)), (len(C), G)).ravel().astype(int),
    "pert_auc_min": pa.min(0).ravel(), "pert_auc_mean": pa.mean(0).ravel(), "pert_jac_min": np.repeat(pj.min(0), G),
    "support": support.ravel(), "mean_auc": mean_auc.ravel(), "sd_auc": sd_auc.ravel(),
    "rand4_support": rand4.ravel(), "rand8_support": rand8.ravel(),
    "design_support": ((np.concatenate([des_stack, AL[np.array(C)][None]]) >= a.tau).mean(0)).ravel(),
    "design_auc_min": des_stack.min(0).ravel(), "design_auc_mean": des_stack.mean(0).ravel(),
    **{f"des_{k}": v.ravel() for k, v in des_auc.items()}})
df["margin"] = df.auc_L - a.tau; df["specificity"] = df.auc_L - df.second_auc
keep = ((df.auc_L >= 0.6) | (df.mean_auc >= 0.6)).values
df = df[keep].reset_index(drop=True)
od = os.path.join(rdir, "fragility"); os.makedirs(od, exist_ok=True)
df.to_csv(os.path.join(od, "claims_table.csv.gz"), index=False)
np.save(os.path.join(od, "sample_jaccard.npy"), jacs)
# per-pipeline claim values for the held-out population (certification benchmark)
np.save(os.path.join(od, "pop_auc.npy"), aucs.reshape(aucs.shape[0], -1)[:, keep].astype(np.float16))
strat = pd.DataFrame([dict(zip(["nhvg", "norm", "ncomp", "batch", "k", "res"],
        [TASKS[allrows[p][0]][f][allrows[p][1]] for f in ["nhvg", "norm", "ncomp", "batch", "k", "res"]])) for p in pick[8:]])
strat.to_csv(os.path.join(od, "pop_pipelines.csv"), index=False)
ref_claims = df[df.auc_L >= a.tau]
print(json.dumps(dict(n_rows=len(df), n_ref_claims=len(ref_claims), n_clusters=len(C),
    frac_ref_claims_support_lt_0_5=float((ref_claims.support < 0.5).mean()),
    frac_ref_claims_support_ge_0_9=float((ref_claims.support >= 0.9).mean())), indent=1))
