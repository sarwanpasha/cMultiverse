"""Pull ONE named CELLxGENE dataset (filtered by dataset_id) and draw a donor-balanced subsample.
Usage: python prep_census.py --name pbmc_s0 --which pbmc --seed 0 --n_donors 20 --per_donor 500
"""
import argparse, os, json, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from common import dataset_dir

PATTERNS = {
    # Stephenson et al. 2021 Nat Med (Haniffa lab) collection
    "pbmc": dict(collection=r"immune response in COVID-19", title=None),
    # Sikkema et al. 2023 Nat Med, HLCA core
    # Perez et al. 2022 Science, SLE PBMC
    "lupus": dict(collection=r"lupus", title=None),
    # Adams et al. 2020 Sci Adv / Habermann et al. 2020 Sci Adv -- idiopathic pulmonary fibrosis lung atlases
    "ipf": dict(collection=r"pulmonary fibrosis", title=None),
    "hlca": dict(collection=r"lung", title=r"(?i)human lung cell atlas.*core|core.*lung cell atlas|integrated cell atlas of the human lung.*core"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--which", required=True, choices=list(PATTERNS))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_donors", type=int, default=20)
ap.add_argument("--per_donor", type=int, default=500)
ap.add_argument("--census_version", default="2025-01-30")
ap.add_argument("--dataset_id", default=None)
ap.add_argument("--balance", default=None, help="'|'-separated disease labels; draw n_donors/len equally from each")
ap.add_argument("--balance_re", default=None, help="';'-separated regexes over disease labels (case-insensitive)")
a = ap.parse_args()

import cellxgene_census
try:
    census = cellxgene_census.open_soma(census_version=a.census_version)
    cv = a.census_version
except Exception as e:
    print("pinned census version failed:", e, "-> using stable")
    census = cellxgene_census.open_soma(census_version="stable"); cv = "stable"

ds = census["census_info"]["datasets"].read().concat().to_pandas()
pat = PATTERNS[a.which]
if a.dataset_id:
    cand = ds[ds.dataset_id == a.dataset_id]
else:
    cand = ds[ds.collection_name.str.contains(pat["collection"], case=False, regex=True)]
    if pat["title"]:
        cand = cand[cand.dataset_title.str.contains(pat["title"], regex=True)]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print("candidate datasets:\n", cand[["dataset_id", "collection_name", "dataset_title", "dataset_total_cell_count"]])
assert len(cand) > 0, "no dataset matched; pass --dataset_id"
row = cand.sort_values("dataset_total_cell_count", ascending=False).iloc[0]
did = row.dataset_id
print("USING", did, "|", row.dataset_title, "|", row.collection_name)

obs = census["census_data"]["homo_sapiens"].obs.read(
    value_filter=f"dataset_id == '{did}'",
    column_names=["soma_joinid", "donor_id", "cell_type", "disease", "assay", "tissue", "sex"]).concat().to_pandas()
print("cells in dataset:", len(obs), "donors:", obs.donor_id.nunique())

rng = np.random.default_rng(a.seed)
vc = obs.donor_id.value_counts()
elig = vc[vc >= a.per_donor].index.values
if a.balance_re:
    import re
    dd = obs.groupby("donor_id")["disease"].agg(lambda s: s.value_counts().index[0])
    print("donor disease labels:", dd.value_counts().to_dict())
    pats = a.balance_re.split(";"); per = a.n_donors // len(pats); donors = []
    for pt in pats:
        pool = np.sort([x for x in elig if re.search(pt, str(dd.get(x)), flags=re.I)])
        print(pt, "eligible donors", len(pool))
        donors += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    donors = np.array(donors)
elif a.balance:
    labs = a.balance.split("|"); dd = obs.groupby("donor_id")["disease"].agg(lambda s: s.value_counts().index[0])
    per = a.n_donors // len(labs); donors = []
    for lb in labs:
        pool = np.sort([x for x in elig if dd.get(x) == lb])
        print(lb, "eligible donors", len(pool))
        donors += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    donors = np.array(donors)
else:
    if len(elig) < a.n_donors:
        print(f"only {len(elig)} donors with >= {a.per_donor} cells; using them all")
    donors = rng.choice(np.sort(elig), size=min(a.n_donors, len(elig)), replace=False)
pick = []
for d in donors:
    ids = obs.loc[obs.donor_id == d, "soma_joinid"].values
    pick.append(rng.choice(ids, size=a.per_donor, replace=False))
pick = np.sort(np.concatenate(pick))

adata = cellxgene_census.get_anndata(census, organism="Homo sapiens", measurement_name="RNA",
                                     X_name="raw", obs_coords=pick)
census.close()
X = sp.csr_matrix(adata.X).astype(np.float32)
genes = adata.var["feature_name"].astype(str).values
keep_g = np.asarray((X > 0).sum(0)).ravel() >= 10
X = X[:, keep_g]; genes = genes[keep_g]
L = np.asarray(X.sum(1)).ravel()
keep_c = L > 0
X = X[keep_c]; o = adata.obs.loc[keep_c].copy()
mt = np.array([g.startswith("MT-") for g in genes])
o["total_counts"] = np.asarray(X.sum(1)).ravel()
o["pct_mt"] = 100 * np.asarray(X[:, mt].sum(1)).ravel() / o["total_counts"].values
o["batch"] = o["donor_id"].astype(str)
o = o[["cell_type", "batch", "disease", "assay", "total_counts", "pct_mt"]]
o.index = [f"c{i}" for i in range(X.shape[0])]

d = dataset_dir(a.name); os.makedirs(d, exist_ok=True)
sp.save_npz(os.path.join(d, "counts.npz"), X)
o.to_csv(os.path.join(d, "obs.csv"))
pd.DataFrame({"gene": genes}).to_csv(os.path.join(d, "genes.csv"), index=False)
meta = dict(source="cellxgene_census", census_version=cv, dataset_id=did, dataset_title=row.dataset_title,
            collection=row.collection_name, seed=a.seed, n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
            n_batches=int(o.batch.nunique()), donors=list(map(str, donors)))
json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
print(o.cell_type.value_counts().to_string())
