"""Theorem checks on the reference partition (author / true / protein labels).

Theorem 1 (rank invariance): if T(x_ig) = h(x_ig / s_i) with h strictly increasing, within-gene cell ranks
  depend only on x/s; all library-size scale transforms (log1p_cp10k, log1p_median, acosh_vst) give identical AUCs.
Theorem 2 (zero-tie decomposition): under analytic Pearson residuals a zero count maps to
  r = -sqrt(mu/(1+mu/theta)), mu = L_i p_g, which is strictly decreasing in depth L_i. Hence, for population A vs rest B,
    AUC_P - AUC_L = pi00 * (D0 - 1/2) + Delta_mixed + Delta_nz,
  where pi00 = z_A z_B (zero fractions), D0 = P(L_a < L_b | both zero). The first term is exact for zero-zero pairs.
Usage: python theory_check.py --name sim_d4
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, load_counts

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); a = ap.parse_args()
d = dataset_dir(a.name); rdir = os.path.join(RES, a.name); vd = os.path.join(rdir, "values")
obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
pz = np.load(os.path.join(vd, "panel.npz"), allow_pickle=True)
genes = pz["genes"].astype(str); depth = pz["depth"].astype(float)
X, _, allgenes, _ = load_counts(a.name)
Xp = X[:, pz["panel"]].toarray()
RL = np.load(os.path.join(vd, "ranks_log1p_cp10k.npy")); RP = np.load(os.path.join(vd, "ranks_pearson.npy"))
RZ = np.load(os.path.join(vd, "ranks_pearson_zt.npy"))
ref = obs.cell_type.astype(str).values
types = [t for t, c in pd.Series(ref).value_counts().items() if c >= 20]
N = len(ref)

def auc_from_ranks(R, m):
    n1 = m.sum(); n0 = N - n1
    return (R[m].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)

rows = []
zero = Xp == 0
for t in types:
    m = ref == t; nA, nB = m.sum(), (~m).sum()
    aL, aP, aZ = auc_from_ranks(RL, m), auc_from_ranks(RP, m), auc_from_ranks(RZ, m)
    zA = zero[m].mean(0); zB = zero[~m].mean(0)
    depth_auc = (rankdata(depth)[m].sum() - nA * (nA + 1) / 2) / (nA * nB)
    for j in range(len(genes)):
        zz = zero[:, j]; n0A = (zz & m).sum(); n0B = (zz & ~m).sum()
        if n0A == 0 or n0B == 0:
            d00 = 0.0
        else:
            r = rankdata(-depth[zz])            # larger rank = shallower = larger residual
            d0 = (r[m[zz]].sum() - n0A * (n0A + 1) / 2) / (n0A * n0B)
            d00 = (n0A * n0B) / (nA * nB) * (d0 - 0.5)
        rows.append((t, genes[j], aL[j], aP[j], aZ[j], zA[j], zB[j], d00, depth_auc))
df = pd.DataFrame(rows, columns=["type", "gene", "auc_L", "auc_P", "auc_Pzt", "zero_A", "zero_B", "pred_zero_tie", "type_depth_auc"])
df["delta"] = df.auc_P - df.auc_L
od = os.path.join(rdir, "theory"); os.makedirs(od, exist_ok=True)
df.to_csv(os.path.join(od, "zero_tie_decomposition.csv.gz"), index=False)
inf = df[(df.auc_L > 0.6) | (df.auc_P > 0.6)]
def r2(y, x): return 1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
summ = dict(
    n_pairs=len(df), n_informative=len(inf),
    R2_delta_vs_zero_tie_term_all=float(r2(df.delta, df.pred_zero_tie)),
    R2_delta_vs_zero_tie_term_informative=float(r2(inf.delta, inf.pred_zero_tie)),
    corr_delta_pred=float(np.corrcoef(df.delta, df.pred_zero_tie)[0, 1]),
    median_abs_delta_P=float(df.delta.abs().median()),
    median_abs_delta_Pzt=float((df.auc_Pzt - df.auc_L).abs().median()),
    per_type=df.groupby("type").apply(lambda g: dict(depth_auc=float(g.type_depth_auc.iloc[0]),
        mean_delta=float(g.delta.mean()), mean_pred=float(g.pred_zero_tie.mean()))).to_dict())
json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "per_type"}, indent=1))
print(pd.DataFrame(summ["per_type"]).T.round(4).to_string())
