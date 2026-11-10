"""Hao et al. 2021 (Cell) PBMC CITE-seq, 3' kit: RNA + 228 antibodies, 8 donors (GSE164378).
Donor-balanced subsample; RNA for the lattice; protein (CLR) as ground truth via an antibody -> gene map.
Usage: python prep_hao.py --name cite_hao --per_donor 1250 --seed 0
"""
import argparse, os, sys, json, re, subprocess, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io
sys.path.insert(0, os.path.dirname(__file__))
from common import dataset_dir

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="cite_hao")
ap.add_argument("--per_donor", type=int, default=1250); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
d = dataset_dir(a.name); raw = os.path.join(d, "raw"); os.makedirs(raw, exist_ok=True)
BASE = "https://ftp.ncbi.nlm.nih.gov/geo"
def listdir(url):
    html = urllib.request.urlopen(url, timeout=120).read().decode()
    return sorted(set(re.findall(r'href="([^"?/][^"]*)"', html)))
def get(url, dst):
    if not os.path.exists(dst): subprocess.check_call(["curl", "-fsSL", "--retry", "5", "-o", dst, url])
files = {}
for gsm in ["GSM5008737", "GSM5008738"]:
    u = f"{BASE}/samples/GSM5008nnn/{gsm}/suppl/"
    for f in listdir(u):
        if f.endswith(".gz"): get(u + f, os.path.join(raw, f)); files[f] = os.path.join(raw, f)
meta_f = os.path.join(raw, "meta_3P.csv.gz")
get(f"{BASE}/series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz", meta_f)
print("files:", list(files))
def trio(tag):
    b = [v for k, v in files.items() if tag in k and "barcodes" in k][0]
    f = [v for k, v in files.items() if tag in k and ("features" in k or "genes" in k)][0]
    m = [v for k, v in files.items() if tag in k and "matrix" in k][0]
    return b, f, m
meta = pd.read_csv(meta_f, index_col=0)
print(meta.columns.tolist()); print(meta.head(2).to_string())
rng = np.random.default_rng(a.seed)
pick = []
for dn, g in meta.groupby("donor"):
    pick.append(rng.choice(g.index.values, size=min(a.per_donor, len(g)), replace=False))
pick = np.concatenate(pick)

def load(tag):
    b, f, m = trio(tag)
    bc = pd.read_csv(b, header=None)[0].astype(str).values
    ft = pd.read_csv(f, header=None, sep="\t")[0].astype(str).values
    M = scipy.io.mmread(m).tocsc()            # features x cells
    pos = pd.Series(np.arange(len(bc)), index=bc)
    idx = pos.reindex(pick).values
    assert not np.isnan(idx).any(), "barcode mismatch with metadata"
    return ft, sp.csr_matrix(M[:, idx.astype(int)].T).astype(np.float32)
genes, X = load("RNA_3P"); prot, A = load("ADT_3P")
o = meta.loc[pick].copy()
keep_g = np.asarray((X > 0).sum(0)).ravel() >= 10
X = X[:, keep_g]; genes = genes[keep_g]
mt = np.array([g.startswith("MT-") for g in genes])
L = np.asarray(X.sum(1)).ravel()
out = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
out["cell_type"] = o["celltype.l2"].astype(str).values; out["batch"] = o["donor"].astype(str).values
out["disease"] = "normal"; out["assay"] = "10x 3' v3 CITE"; out["total_counts"] = L
out["pct_mt"] = 100 * np.asarray(X[:, mt].sum(1)).ravel() / L
Ad = A.toarray(); lA = np.log1p(Ad); clr = lA - lA.mean(1, keepdims=True)

CD = {"CD1a": "CD1A", "CD1c": "CD1C", "CD1d": "CD1D", "CD2": "CD2", "CD3": "CD3E", "CD4": "CD4", "CD5": "CD5", "CD7": "CD7",
 "CD8": "CD8A", "CD8a": "CD8A", "CD9": "CD9", "CD11a": "ITGAL", "CD11b": "ITGAM", "CD11c": "ITGAX", "CD13": "ANPEP", "CD14": "CD14",
 "CD15": "FUT4", "CD16": "FCGR3A", "CD18": "ITGB2", "CD19": "CD19", "CD20": "MS4A1", "CD21": "CR2", "CD22": "CD22", "CD24": "CD24",
 "CD25": "IL2RA", "CD26": "DPP4", "CD27": "CD27", "CD28": "CD28", "CD29": "ITGB1", "CD31": "PECAM1", "CD32": "FCGR2A", "CD33": "CD33",
 "CD34": "CD34", "CD35": "CR1", "CD36": "CD36", "CD38": "CD38", "CD39": "ENTPD1", "CD40": "CD40", "CD41": "ITGA2B", "CD42b": "GP1BA",
 "CD44": "CD44", "CD47": "CD47", "CD48": "CD48", "CD49a": "ITGA1", "CD49b": "ITGA2", "CD49d": "ITGA4", "CD49f": "ITGA6",
 "CD52": "CD52", "CD54": "ICAM1", "CD56": "NCAM1", "CD57": "B3GAT1", "CD58": "CD58", "CD59": "CD59", "CD61": "ITGB3", "CD62L": "SELL",
 "CD62P": "SELP", "CD63": "CD63", "CD64": "FCGR1A", "CD69": "CD69", "CD70": "CD70", "CD71": "TFRC", "CD73": "NT5E", "CD79b": "CD79B",
 "CD80": "CD80", "CD81": "CD81", "CD82": "CD82", "CD83": "CD83", "CD85j": "LILRB1", "CD86": "CD86", "CD88": "C5AR1", "CD94": "KLRD1",
 "CD95": "FAS", "CD96": "CD96", "CD100": "SEMA4D", "CD101": "CD101", "CD103": "ITGAE", "CD105": "ENG", "CD107a": "LAMP1",
 "CD112": "NECTIN2", "CD115": "CSF1R", "CD119": "IFNGR1", "CD122": "IL2RB", "CD123": "IL3RA", "CD124": "IL4R", "CD127": "IL7R",
 "CD134": "TNFRSF4", "CD137": "TNFRSF9", "CD141": "THBD", "CD146": "MCAM", "CD152": "CTLA4", "CD154": "CD40LG", "CD155": "PVR",
 "CD158b": "KIR2DL3", "CD158e1": "KIR3DL1", "CD161": "KLRB1", "CD163": "CD163", "CD169": "SIGLEC1", "CD172a": "SIRPA",
 "CD183": "CXCR3", "CD185": "CXCR5", "CD186": "CXCR6", "CD192": "CCR2", "CD194": "CCR4", "CD195": "CCR5", "CD196": "CCR6",
 "CD197": "CCR7", "CD200": "CD200", "CD223": "LAG3", "CD226": "CD226", "CD244": "CD244", "CD253": "TNFSF10", "CD268": "TNFRSF13C",
 "CD270": "TNFRSF14", "CD272": "BTLA", "CD273": "PDCD1LG2", "CD274": "CD274", "CD278": "ICOS", "CD279": "PDCD1", "CD303": "CLEC4C",
 "CD304": "NRP1", "CD314": "KLRK1", "CD319": "SLAMF7", "CD335": "NCR1", "CD337": "NCR3", "CD352": "SLAMF6", "CD357": "TNFRSF18",
 "CD360": "IL21R", "HLA-DR": "HLA-DRA", "IgD": "IGHD", "IgM": "IGHM", "FceRIa": "FCER1A", "TIM-3": "HAVCR2", "PD-1": "PDCD1",
 "PD-L1": "CD274", "LAG-3": "LAG3", "TCR-V-alpha-7.2": "TRAV1-2", "TCR-alpha-beta": "TRAC"}
gset = set(genes); pairs = []
for p in prot:
    base = re.sub(r"-\d+$", "", p)
    g = CD.get(base) or CD.get(p) or (base if base in gset else None)
    if g and g in gset and not base.startswith(("CD45", "Rat", "Mouse", "Ig-", "IgG")): pairs.append((p, g))
pm = pd.DataFrame(pairs, columns=["protein", "gene"]).drop_duplicates("gene")
sp.save_npz(os.path.join(d, "counts.npz"), X)
out.to_csv(os.path.join(d, "obs.csv")); pd.DataFrame({"gene": genes}).to_csv(os.path.join(d, "genes.csv"), index=False)
np.save(os.path.join(d, "protein_clr.npy"), clr.astype(np.float32))
pd.DataFrame({"protein": prot}).to_csv(os.path.join(d, "proteins.csv"), index=False)
pm.to_csv(os.path.join(d, "protein_gene_map.csv"), index=False)
m = dict(source="Hao 2021 GSE164378 3P", n_cells=int(X.shape[0]), n_genes=int(X.shape[1]), n_batches=int(out.batch.nunique()),
         n_proteins=len(prot), n_protein_gene_pairs=len(pm), seed=a.seed)
json.dump(m, open(os.path.join(d, "meta.json"), "w"), indent=1)
print(json.dumps(m, indent=1)); print(out.cell_type.value_counts().to_string())
