"""Extract claims for every pipeline in one lattice task.
For each pipeline p, Leiden seed s and value-normalization v (factorized design):
  - partition claims: n_clusters, ARI, NMI vs reference labels, best-match Jaccard per reference population
  - marker claims: AUC of gene g in the cluster matched to its population (matched-population marker, not 'any cluster')
  - discovery precision (simulations): fraction of top-K AUC genes of the matched cluster that are planted markers
Usage: python claims.py --name sim_d2 --task 3
"""
import argparse, os, sys, json, re, glob
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, load_counts, auc_matrix, VALUE_NORMS as NORMS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--task", type=int, required=True)
ap.add_argument("--lattice_dir", default="lattice"); a = ap.parse_args()
d = dataset_dir(a.name); rdir = os.path.join(RES, a.name)
obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
meta = json.load(open(os.path.join(d, "meta.json")))
pz = np.load(os.path.join(rdir, "values", "panel.npz"), allow_pickle=True)
pgenes = pz["genes"].astype(str); gpos = {g: i for i, g in enumerate(pgenes)}
ranks = {v: np.load(os.path.join(rdir, "values", f"ranks_{v}.npy"), mmap_mode="r") for v in NORMS}

# ---------------- reference populations and marker pairs
ref = obs["cell_type"].astype(str).values
sets, pairs, truth = {}, [], []
def add_set(name, members):
    m = np.isin(ref, list(members))
    if m.sum() >= 20: sets[name] = m
if a.name.startswith("sim"):
    for t in np.unique(ref): add_set(t, [t])
    tm = pd.read_csv(os.path.join(d, "true_markers.csv"))
    for _, r in tm.iterrows():
        if r.gene in gpos and r.cell_type in sets: pairs.append((r.gene, r.cell_type))
elif a.name.startswith("cite"):
    for t in np.unique(ref): add_set(t, [t])
    clr = np.load(os.path.join(d, "protein_clr.npy")); prot = pd.read_csv(os.path.join(d, "proteins.csv")).protein.tolist()
    GP = {"CD3": "CD3E", "CD4": "CD4", "CD8a": "CD8A", "CD14": "CD14", "CD15": "FUT4", "CD16": "FCGR3A",
          "CD56": "NCAM1", "CD19": "CD19", "CD25": "IL2RA", "PD-1": "PDCD1", "TIGIT": "TIGIT", "CD127": "IL7R"}
    mf = os.path.join(d, "protein_gene_map.csv")
    if os.path.exists(mf): GP = dict(pd.read_csv(mf).values.tolist())
    from scipy.stats import rankdata
    for p, g in GP.items():
        if p not in prot or g not in gpos: continue
        pr = rankdata(clr[:, prot.index(p)])
        for s, m in sets.items():
            n1, n0 = m.sum(), (~m).sum()
            pauc = (pr[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            pairs.append((g, s)); truth.append(pauc)
else:
    key = "pbmc" if a.name.startswith("pbmc") else "hlca"
    ms = json.load(open(os.path.join(os.path.dirname(__file__), "marker_sets.json")))[key]
    types = np.unique(ref)
    for name, spec in ms.items():
        mem = [t for t in types if re.search(spec["regex"], t, flags=re.I)]
        if spec.get("exclude"): mem = [t for t in mem if not re.search(spec["exclude"], t, flags=re.I)]
        add_set(name, mem)
        if name in sets:
            for g in spec["genes"]:
                if g in gpos: pairs.append((g, name))
set_names = list(sets)
set_idx = {s: i for i, s in enumerate(set_names)}
pair_g = np.array([gpos[g] for g, _ in pairs], dtype=int); pair_s = np.array([set_idx[s] for _, s in pairs], dtype=int)
setmat = sp.csr_matrix(np.stack([sets[s] for s in set_names]).astype(np.float32))  # n_sets x N
set_sizes = np.asarray(setmat.sum(1)).ravel()
# known-true marker sets per population: planted (sims), protein-confirmed AUC>0.75 (CITE), curated (real)
truth_arr = np.array(truth, dtype=float)
if a.name.startswith("cite"):
    true_sets = [set(pair_g[(pair_s == i) & (truth_arr > 0.75)]) for i in range(len(set_names))]
else:
    true_sets = [set(pair_g[pair_s == i]) for i in range(len(set_names))]
KTOP = 50
G = len(pgenes)

lat = np.load(os.path.join(rdir, a.lattice_dir, f"task_{a.task:03d}.npz"), allow_pickle=True)
L = lat["labels"]; P, S, N = L.shape
nv = len(NORMS)
ncl = np.zeros((P, S), np.int16); ari = np.zeros((P, S), np.float32); nmi = np.zeros((P, S), np.float32)
jac = np.zeros((P, S, len(set_names)), np.float32)
mauc = np.full((P, S, nv, len(pairs)), np.nan, np.float32)
dprec = np.full((P, S, nv, len(set_names)), np.nan, np.float32)   # recall@50 of known-true markers
auc_sum = np.zeros((nv, len(set_names), G)); top_cnt = np.zeros((nv, len(set_names), G), np.int32); n_acc = 0
R = {v: np.asarray(ranks[v]) for v in NORMS}
for p in range(P):
    for s in range(S):
        lab = L[p, s].astype(int)
        K = lab.max() + 1
        ncl[p, s] = K
        ari[p, s] = adjusted_rand_score(ref, lab); nmi[p, s] = normalized_mutual_info_score(ref, lab)
        oh = sp.csr_matrix((np.ones(N, np.float32), (lab, np.arange(N))), shape=(K, N))
        inter = np.asarray((setmat @ oh.T).todense())               # n_sets x K
        csize = np.asarray(oh.sum(1)).ravel()
        J = inter / (set_sizes[:, None] + csize[None, :] - inter)
        best = J.argmax(1); jac[p, s] = J.max(1)
        for vi, v in enumerate(NORMS):
            A = auc_matrix(R[v], lab)                                 # K x G
            mauc[p, s, vi] = A[best[pair_s], pair_g]
            Ab = np.nan_to_num(A[best], nan=0.5)                            # n_sets x G
            top = np.argpartition(-Ab, KTOP, axis=1)[:, :KTOP]
            auc_sum[vi] += Ab
            np.add.at(top_cnt[vi], (np.repeat(np.arange(len(set_names)), KTOP), top.ravel()), 1)
            for i in range(len(set_names)):
                if true_sets[i]:
                    dprec[p, s, vi, i] = len(true_sets[i].intersection(top[i])) / len(true_sets[i])
        n_acc += 1
    if p % 100 == 0: print("pipeline", p, "/", P, flush=True)

out_dir = os.path.join(rdir, a.lattice_dir.replace("lattice", "claims")); os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(os.path.join(out_dir, f"claims_{a.task:03d}.npz"),
    nhvg=lat["nhvg"], norm=lat["norm"], ncomp=lat["ncomp"], batch=lat["batch"], k=lat["k"], res=lat["res"],
    n_clusters=ncl, ari=ari, nmi=nmi, jaccard=jac, marker_auc=mauc, disc_prec=dprec,
    set_names=np.array(set_names), set_sizes=set_sizes, pair_gene=np.array([g for g, _ in pairs]),
    pair_set=np.array([s for _, s in pairs]), pair_truth=np.array(truth, dtype=np.float32), vnorms=np.array(NORMS),
    auc_sum=auc_sum.astype(np.float32), top_cnt=top_cnt, n_acc=n_acc, panel_genes=pgenes,
    true_sets=np.array([";".join(pgenes[sorted(t)]) for t in true_sets]))
print("done", a.task)
