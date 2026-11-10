"""Pair-type decomposition of normalization effects on rank-based marker claims (generalizes Theorems 1-2).
For population A vs rest B and transform v:  AUC_v = pi00*A00_v + pi_mix*Amix_v + pi_pp*App_v, where the three terms are the
Mann-Whitney AUCs restricted to zero-zero, zero/nonzero and nonzero-nonzero cell pairs. For two transforms v1, v2:
  Delta = Delta_00 + Delta_mix + Delta_pp.
Theory: library-size scale family (L) vs poscounts (Q): Delta_00 = Delta_mix = 0 (zeros tied and below nonzeros in both);
all effect sits in Delta_pp and grows with the dispersion of log(s_i/L_i). L vs Pearson: Delta_00 = Theorem-2 term.
L vs Pearson-zero-tied: Delta_00 = 0.
Usage: python theory_general.py --name pbmc_s0
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, load_counts, poscounts_size_factors

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); a = ap.parse_args()
rdir = os.path.join(RES, a.name); vd = os.path.join(rdir, "values")
X, obs, genes, meta = load_counts(a.name)
pz = np.load(os.path.join(vd, "panel.npz"), allow_pickle=True); panel = pz["panel"]; L = pz["depth"].astype(float)
Z0 = (X[:, panel].toarray() == 0); N, G = Z0.shape
V = {v: np.load(os.path.join(vd, f"values_{v}.npy")) for v in ["log1p_cp10k", "acosh_vst", "log1p_poscounts", "pearson", "pearson_zt"]}
ref = obs.cell_type.astype(str).values
types = [t for t, c in pd.Series(ref).value_counts().items() if c >= 20]
T = np.stack([ref == t for t in types]).astype(np.float32); nT = T.sum(1)

def terms(vals):
    """per (type, gene): U-statistics restricted to 00, pp pairs (mix = total - 00 - pp), and total U."""
    U00 = np.zeros((len(types), G)); Upp = np.zeros((len(types), G)); Utot = np.zeros((len(types), G))
    n0A = np.zeros((len(types), G)); npA = np.zeros((len(types), G))
    rall = rankdata(vals, axis=0, method="average")
    Utot = T @ rall - (nT * (nT + 1) / 2)[:, None]
    for j in range(G):
        z = Z0[:, j]; nz = ~z
        for mask, out, cnt in ((z, U00, n0A), (nz, Upp, npA)):
            k = mask.sum()
            if k == 0: continue
            r = rankdata(vals[mask, j]); Ts = T[:, mask]; n1 = Ts.sum(1)
            out[:, j] = Ts @ r - n1 * (n1 + 1) / 2; cnt[:, j] = n1
    return Utot, U00, Upp, n0A, npA

base = terms(V["log1p_cp10k"])
den = (nT * (N - nT))[:, None]
rows = []
for v in ["acosh_vst", "log1p_poscounts", "pearson", "pearson_zt"]:
    Ut, U0, Up, _, _ = terms(V[v])
    d_tot = (Ut - base[0]) / den; d00 = (U0 - base[1]) / den; dpp = (Up - base[2]) / den; dmix = d_tot - d00 - dpp
    for ti, t in enumerate(types):
        rows.append(pd.DataFrame({"pair": f"log1p_cp10k_vs_{v}", "type": t, "gene": pz["genes"].astype(str),
                                  "d_total": d_tot[ti], "d_00": d00[ti], "d_mix": dmix[ti], "d_pp": dpp[ti]}))
df = pd.concat(rows)
sf = poscounts_size_factors(X)
disp = float(np.var(np.log(sf / (L / L.mean()))))
summ = {}
for pr, g in df.groupby("pair"):
    tot2 = (g.d_total ** 2).sum()
    summ[pr] = dict(median_abs_total=float(g.d_total.abs().median()), max_abs_total=float(g.d_total.abs().max()),
                    share_00=float((g.d_00 * g.d_total).sum() / tot2) if tot2 > 0 else 0.0,
                    share_mix=float((g.d_mix * g.d_total).sum() / tot2) if tot2 > 0 else 0.0,
                    share_pp=float((g.d_pp * g.d_total).sum() / tot2) if tot2 > 0 else 0.0)
summ["var_log_sizefactor_ratio_Q_over_L"] = disp
od = os.path.join(rdir, "theory"); os.makedirs(od, exist_ok=True)
df.to_csv(os.path.join(od, "pairtype_decomposition.csv.gz"), index=False)
json.dump(summ, open(os.path.join(od, "pairtype_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
