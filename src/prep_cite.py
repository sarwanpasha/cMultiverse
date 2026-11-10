"""10x Genomics 10k PBMC CITE-seq (TotalSeq-B, 17 antibodies). RNA for the lattice, protein as ground truth.
Usage: python prep_cite.py --name cite_pbmc10k
"""
import argparse, os, sys, json, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from common import dataset_dir

URL = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5"
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="cite_pbmc10k"); a = ap.parse_args()
d = dataset_dir(a.name); os.makedirs(d, exist_ok=True)
raw = os.path.join(d, "raw_10x.h5")
if not os.path.exists(raw):
    subprocess.check_call(["curl", "-fsSL", "-o", raw, URL])

import scanpy as sc
ad = sc.read_10x_h5(raw, gex_only=False)
ad.var_names_make_unique()
ft = ad.var["feature_types"].astype(str).values
rna = ad[:, ft == "Gene Expression"].copy()
adt = ad[:, ft == "Antibody Capture"].copy()
X = sp.csr_matrix(rna.X).astype(np.float32)
genes = rna.var_names.astype(str).values
L = np.asarray(X.sum(1)).ravel()
mt = np.array([g.startswith("MT-") for g in genes])
pct_mt = 100 * np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(L, 1)
ngenes = np.asarray((X > 0).sum(1)).ravel()
keep_c = (L >= 1000) & (ngenes >= 300) & (pct_mt < 20)
X = X[keep_c]; A = np.asarray(adt.X[keep_c].todense() if sp.issparse(adt.X) else adt.X[keep_c], dtype=np.float32)
keep_g = np.asarray((X > 0).sum(0)).ravel() >= 10
X = X[:, keep_g]; genes = genes[keep_g]
prot = [p.split("_")[0] for p in adt.var_names]
# CLR across antibodies within each cell
lA = np.log1p(A); clr = lA - lA.mean(1, keepdims=True)

# protein-defined populations: Leiden on CLR of non-isotype antibodies
use = [i for i, p in enumerate(prot) if not p.lower().startswith(("igg", "mouse", "rat"))]
pa = sc.AnnData(clr[:, use]); sc.pp.neighbors(pa, n_neighbors=30, use_rep="X", random_state=0)
sc.tl.leiden(pa, resolution=0.6, flavor="igraph", n_iterations=2, random_state=0, key_added="pop")
pops = pa.obs["pop"].astype(int).values
means = pd.DataFrame(clr[:, use], columns=[prot[i] for i in use]).groupby(pops).mean()
labels = {}
for c, r in means.iterrows():
    top = r.sort_values(ascending=False).index[:3].tolist()
    labels[c] = f"P{c}:" + "+".join(top)
o = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
o["cell_type"] = [labels[p] for p in pops]
o["batch"] = "s1"; o["disease"] = "normal"; o["assay"] = "10x 3' v3"
o["total_counts"] = np.asarray(X.sum(1)).ravel()
o["pct_mt"] = 100 * np.asarray(X[:, np.array([g.startswith("MT-") for g in genes])].sum(1)).ravel() / o.total_counts.values

sp.save_npz(os.path.join(d, "counts.npz"), X)
o.to_csv(os.path.join(d, "obs.csv"))
pd.DataFrame({"gene": genes}).to_csv(os.path.join(d, "genes.csv"), index=False)
np.save(os.path.join(d, "protein_clr.npy"), clr.astype(np.float32))
pd.DataFrame({"protein": prot}).to_csv(os.path.join(d, "proteins.csv"), index=False)
meta = dict(source="10x pbmc_10k_protein_v3", url=URL, n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
            n_batches=1, n_proteins=len(prot))
json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1)); print(o.cell_type.value_counts().to_string()); print(means.round(2).to_string())
