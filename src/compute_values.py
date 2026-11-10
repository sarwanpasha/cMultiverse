"""Per normalization: average within-gene ranks (and values) for a gene panel, used for the factorized
marker analysis (values from normalization n_v, partition from any pipeline).
Usage: python compute_values.py --name pbmc_s0
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, load_counts, normalize, avg_ranks, VALUE_NORMS as NORMS

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
ap.add_argument("--n_panel", type=int, default=4000); a = ap.parse_args()
X, obs, genes, meta = load_counts(a.name)
d = dataset_dir(a.name)
det = np.asarray((X > 0).mean(0)).ravel()
must = set()
if a.name.startswith("cite"):
    must |= {"CD3E","CD4","CD8A","CD14","FUT4","FCGR3A","NCAM1","CD19","IL2RA","PDCD1","TIGIT","IL7R"}
    mf = os.path.join(d, "protein_gene_map.csv")
    if os.path.exists(mf): must |= set(pd.read_csv(mf).gene.astype(str))
for fn in ["true_markers.csv"]:
    p = os.path.join(d, fn)
    if os.path.exists(p): must |= set(pd.read_csv(p).gene.astype(str))
mk = os.path.join(os.path.dirname(__file__), "marker_sets.json")
if os.path.exists(mk):
    key = "pbmc" if a.name.startswith("pbmc") else ("hlca" if a.name.startswith(("hlca", "lung")) else "")
    for grp in json.load(open(mk)).get(key, {}).values():
        must |= set(grp["genes"])
gidx = {g: i for i, g in enumerate(genes)}
if X.shape[1] <= 8000:
    panel = np.arange(X.shape[1])
else:
    top = np.argsort(-det)[:a.n_panel]
    panel = np.unique(np.concatenate([top, [gidx[g] for g in must if g in gidx]])).astype(int)
out_dir = os.path.join(RES, a.name, "values"); os.makedirs(out_dir, exist_ok=True)
L = np.asarray(X.sum(1)).ravel()
np.savez_compressed(os.path.join(out_dir, "panel.npz"), panel=panel, genes=genes[panel],
                    depth=L, zero_frac=1 - det[panel])
for n in NORMS:
    Z = normalize(X, n)[:, panel]
    R = avg_ranks(Z)
    np.save(os.path.join(out_dir, f"ranks_{n}.npy"), R)
    np.save(os.path.join(out_dir, f"values_{n}.npy"), Z.astype(np.float32))
    print(n, "done", flush=True)
print("panel size", len(panel))
