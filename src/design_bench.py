"""Cost-aware design benchmark against the exhaustive lattice (ground truth).
Designs at equal compute cost (prefix-cached cost model: embedding c_E, kNN graph c_G, Leiden c_L):
  iid         : uniformly random pipelines
  splitplot   : random embeddings (nhvg,norm,ncomp,batch) x random m of the (k,res) sub-plots
  splitplot+F : split-plot + factorized augmentation for marker claims (each partition re-used under all value norms)
Estimators: ridge functional-ANOVA (main + pairwise) -> exact Sobol of the fitted grid; one-hot GP (legacy) for reference.
Usage: python design_bench.py --name sim_d2 --cE 30 --cG 6 --cL 1.5
"""
import argparse, os, sys, json, glob, itertools, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, dataset_dir, levels_for, FACTORS, apply_quick

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
ap.add_argument("--cE", type=float, default=30.0); ap.add_argument("--cG", type=float, default=6.0)
ap.add_argument("--cL", type=float, default=1.5); ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--fracs", default="0.01,0.02,0.05,0.1,0.2"); ap.add_argument("--max_claims", type=int, default=60)
ap.add_argument("--gp", action="store_true"); ap.add_argument("--quick", action="store_true")
ap.add_argument("--only_pf", action="store_true", help="only pick-freeze estimators (jansen, jansen_cv)")
a = ap.parse_args()
rng = np.random.default_rng(0)
rdir = os.path.join(RES, a.name); meta = json.load(open(os.path.join(dataset_dir(a.name), "meta.json")))
lv = levels_for(meta)
SUF = "_quick" if a.quick else ""
if a.quick: lv = apply_quick(lv)
Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(rdir, "claims" + SUF, "claims_*.npz")))]
cat = lambda k: np.concatenate([z[k] for z in Z])
design = pd.DataFrame({f: cat(f) for f in FACTORS})
shape = tuple(len(lv[f]) for f in FACTORS)
idx = np.stack([pd.Index(lv[f]).get_indexer(design[f].astype(type(lv[f][0]))) for f in FACTORS], 1)
order = np.argsort(np.ravel_multi_index(idx.T, shape))
grid = lambda arr: arr[order].reshape(shape + arr.shape[1:])
vnorms = list(Z[0]["vnorms"]); vsel = [vnorms.index(n) for n in lv["norm"]]
norm_ax = FACTORS.index("norm")

claims = {}
claims["n_clusters"] = ("partition", grid(cat("n_clusters").astype(float)).mean(-1), None)
claims["ari"] = ("partition", grid(cat("ari")).mean(-1), None)
jac = grid(cat("jaccard")).mean(6)
for i, s in enumerate(Z[0]["set_names"]): claims[f"recover:{s}"] = ("recovery", jac[..., i], None)
mauc = np.nan_to_num(grid(cat("marker_auc")), nan=0.5).mean(6)[..., vsel, :]   # factors x V x C
pg, ps = Z[0]["pair_gene"], Z[0]["pair_set"]
mi = rng.permutation(len(pg))[:a.max_claims]
for i in mi:
    ext = mauc[..., i]
    nat = np.stack([np.take(np.take(ext, j, axis=norm_ax), j, axis=-1) for j in range(len(vsel))], axis=norm_ax)
    claims[f"marker:{pg[i]}|{ps[i]}"] = ("marker", nat, ext)

def sobol_ST(Y):
    V = Y.var(); d = Y.ndim
    if V <= 1e-14: return np.zeros(d), np.zeros(d)
    S1 = np.array([Y.mean(axis=tuple(i for i in range(d) if i != j)).var() / V for j in range(d)])
    ST = np.array([Y.var(axis=j).mean() / V for j in range(d)])
    return S1, ST

truth = {c: sobol_ST(Y) for c, (t, Y, _) in claims.items()}
all_idx = np.array(list(np.ndindex(*shape)))
emb_shape = shape[:4]; sub_shape = shape[4:]
n_sub = int(np.prod(sub_shape))
full_cost = np.prod(emb_shape) * (a.cE + shape[4] * a.cG + n_sub * a.cL)

def cost(sel):
    s = all_idx[sel]
    ne = len({tuple(r[:4]) for r in s}); ng = len({tuple(r[:5]) for r in s})
    return ne * a.cE + ng * a.cG + len(s) * a.cL

def design_iid(budget):
    perm = rng.permutation(len(all_idx)); sel = []; embs = set(); graphs = set(); c = 0.0
    for p in perm:
        r = all_idx[p]; e = tuple(r[:4]); g = tuple(r[:5])
        dc = (a.cE if e not in embs else 0) + (a.cG if g not in graphs else 0) + a.cL
        if c + dc > budget: break
        c += dc; sel.append(p); embs.add(e); graphs.add(g)
    return np.array(sel)

def design_split(budget, m):
    sel = []; c = 0.0
    for e in rng.permutation(int(np.prod(emb_shape))):
        ei = np.unravel_index(e, emb_shape)
        subs = rng.permutation(n_sub)[:m]
        ks = {np.unravel_index(s, sub_shape)[0] for s in subs}
        dc = a.cE + len(ks) * a.cG + m * a.cL
        if c + dc > budget: break
        c += dc
        for s in subs:
            si = np.unravel_index(s, sub_shape)
            sel.append(np.ravel_multi_index(tuple(ei) + tuple(si), shape))
    return np.array(sel)

def onehot(rows, sh):
    cols = []; offs = np.cumsum([0] + list(sh))
    M = np.zeros((len(rows), offs[-1]), np.float32)
    for j in range(len(sh)): M[np.arange(len(rows)), offs[j] + rows[:, j]] = 1
    return M

def ridge_anova(rows, y, sh, pred_rows, lam=1e-2):
    """Main effects + all pairwise interactions (one-hot), ridge-regularized."""
    def feats(R):
        B = onehot(R, sh); offs = np.cumsum([0] + list(sh)); F = [np.ones((len(R), 1), np.float32), B]
        for i, j in itertools.combinations(range(len(sh)), 2):
            Bi = B[:, offs[i]:offs[i + 1]]; Bj = B[:, offs[j]:offs[j + 1]]
            F.append((Bi[:, :, None] * Bj[:, None, :]).reshape(len(R), -1))
        return np.hstack(F)
    X = feats(rows); A = X.T @ X + lam * len(rows) * np.eye(X.shape[1]); A[0, 0] -= lam * len(rows)
    w = np.linalg.solve(A, X.T @ y)
    out = np.empty(len(pred_rows), np.float32)
    for s in range(0, len(pred_rows), 4000): out[s:s + 4000] = feats(pred_rows[s:s + 4000]) @ w
    return out

def gp_fit(rows, y, sh, pred_rows):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    if len(rows) > 1500: idx = rng.permutation(len(rows))[:1500]; rows, y = rows[idx], y[idx]
    gp = GaussianProcessRegressor(ConstantKernel() * RBF(1.0) + WhiteKernel(), normalize_y=True, random_state=0)
    gp.fit(onehot(rows, sh), y); return gp.predict(onehot(pred_rows, sh))

# compute-optimal sub-plot size (two-stage sampling): m* = sqrt((cE + cG*~) / cL * sigma_sub^2 / sigma_whole^2)
def opt_m(Y):
    Yw = Y.reshape(int(np.prod(emb_shape)), n_sub)
    s_w = Yw.mean(1).var(); s_s = Yw.var(1).mean()
    if s_w <= 1e-14: return n_sub
    return int(np.clip(round(np.sqrt((a.cE + a.cG * shape[4] / 2) / a.cL * s_s / s_w)), 2, n_sub))

def design_pickfreeze(budget):
    """Jansen pick-freeze: base matrices A, B and A_B^(j) for every factor j; grow N until the prefix-cached cost hits budget."""
    d = len(shape); N = 2; best = None
    while True:
        A = rng.integers(0, shape, size=(N, d)); B = rng.integers(0, shape, size=(N, d))
        ABs = []
        for j in range(d):
            M = A.copy(); M[:, j] = B[:, j]; ABs.append(M)
        flat = lambda R: np.ravel_multi_index(R.T, shape)
        sel = np.unique(np.concatenate([flat(A), flat(B)] + [flat(M) for M in ABs]))
        if cost(sel) > budget: return best
        best = (flat(A), flat(B), [flat(M) for M in ABs], sel); N = int(N * 1.25) + 1

def jansen(Yf, pf):
    fa, fb, fabs, _ = pf; yA, yB = Yf[fa], Yf[fb]
    V = np.concatenate([yA, yB]).var()
    if V <= 1e-14: return np.zeros(len(fabs)), np.zeros(len(fabs))
    ST = np.array([np.mean((yA - Yf[f]) ** 2) / 2 / V for f in fabs])
    S1 = np.array([(V - np.mean((yB - Yf[f]) ** 2) / 2) / V for f in fabs])
    return np.clip(S1, 0, 1), np.clip(ST, 0, 1)

def jansen_cv(Y, pf):
    """Cross-fitted control-variate Jansen estimator. A ridge functional-ANOVA surrogate fitted on one half of the
    pick-freeze base rows supplies exact grid moments (V, E_-j Var_j, E_j Var_-j); the other half supplies an unbiased
    Monte Carlo correction of each moment by the difference f - f_hat. Halves are swapped and averaged."""
    fa, fb, fabs, _ = pf; Yf = Y.reshape(-1); n = len(fa); d = len(fabs)
    if n < 8: return jansen(Yf, pf)
    perm = rng.permutation(n); halves = [perm[: n // 2], perm[n // 2:]]
    ests = []
    for k in (0, 1):
        tr, te = halves[k], halves[1 - k]
        tr_idx = np.unique(np.concatenate([fa[tr], fb[tr]] + [f[tr] for f in fabs]))
        if Yf[tr_idx].var() < 1e-14: return jansen(Yf, pf)
        fh = ridge_anova(all_idx[tr_idx], Yf[tr_idx], shape, all_idx).reshape(-1)
        G = fh.reshape(shape); Vh = G.var()
        Th = np.array([G.var(axis=j).mean() for j in range(d)])                       # E_-j Var_j f_hat
        Uh = np.array([Vh - G.mean(axis=tuple(i for i in range(d) if i != j)).var() for j in range(d)])  # E_j Var_-j
        rA, rB = Yf[fa[te]] - fh[fa[te]], Yf[fb[te]] - fh[fb[te]]
        yA, yB, hA, hB = Yf[fa[te]], Yf[fb[te]], fh[fa[te]], fh[fb[te]]
        V = Vh + (np.concatenate([yA, yB]).var() - np.concatenate([hA, hB]).var())
        T = np.array([Th[j] + (np.mean((yA - Yf[fabs[j][te]]) ** 2) - np.mean((hA - fh[fabs[j][te]]) ** 2)) / 2 for j in range(d)])
        U = np.array([Uh[j] + (np.mean((yB - Yf[fabs[j][te]]) ** 2) - np.mean((hB - fh[fabs[j][te]]) ** 2)) / 2 for j in range(d)])
        ests.append((V, T, U))
    V = np.mean([e[0] for e in ests]); T = np.mean([e[1] for e in ests], 0); U = np.mean([e[2] for e in ests], 0)
    if V <= 1e-14: return np.zeros(d), np.zeros(d)
    return np.clip((V - U) / V, 0, 1), np.clip(T / V, 0, 1)

ext_shape = shape + (len(vsel),)
res = []
fracs = [float(x) for x in a.fracs.split(",")]
for frac in fracs:
    budget = frac * full_cost
    for rep in range(a.reps):
        D = {"iid": design_iid(budget), "splitplot_full": design_split(budget, n_sub),
             "splitplot_m6": design_split(budget, 6)}
        pf = design_pickfreeze(budget)
        if pf is not None:
            for c, (ctype, Y, ext) in claims.items():
                tS1, tST = truth[c]
                for ename, fn in (("jansen", lambda: jansen(Y.reshape(-1), pf)), ("jansen_cv", lambda: jansen_cv(Y, pf))):
                    S1, ST = fn()
                    res.append(dict(frac=frac, rep=rep, design="pickfreeze", estimator=ename, claim=c, type=ctype,
                                    n_pipelines=len(pf[3]), mae_ST=float(np.abs(ST - tST).mean()),
                                    mae_S1=float(np.abs(S1 - tS1).mean()), top_ok=int(ST.argmax() == tST.argmax())))
        if a.only_pf: D = {}
        for dname, sel in D.items():
            rows = all_idx[sel]
            for c, (ctype, Y, ext) in claims.items():
                y = Y.reshape(-1)[sel]
                if y.var() < 1e-12: continue
                ests = {"ridge": ridge_anova(rows, y, shape, all_idx)}
                if a.gp and dname == "iid" and rep < 5: ests["gp_onehot"] = gp_fit(rows, y, shape, all_idx)
                if ext is not None and dname.startswith("splitplot"):
                    erows = np.concatenate([np.column_stack([rows, np.full(len(rows), v)]) for v in range(len(vsel))])
                    ey = np.concatenate([ext.reshape(-1, len(vsel))[sel, v] for v in range(len(vsel))])
                    pe = ridge_anova(erows, ey, ext_shape, np.column_stack([all_idx, all_idx[:, norm_ax]]))
                    ests["ridge+factorized"] = pe
                for ename, pred in ests.items():
                    S1, ST = sobol_ST(pred.reshape(shape)); tS1, tST = truth[c]
                    res.append(dict(frac=frac, rep=rep, design=dname, estimator=ename, claim=c, type=ctype,
                                    n_pipelines=len(sel), mae_ST=float(np.abs(ST - tST).mean()),
                                    mae_S1=float(np.abs(S1 - tS1).mean()), top_ok=int(ST.argmax() == tST.argmax())))
        print(f"frac {frac} rep {rep} done", flush=True)
df = pd.DataFrame(res)
od = os.path.join(rdir, "design_bench" + SUF); os.makedirs(od, exist_ok=True)
df.to_csv(os.path.join(od, "bench_pf.csv.gz" if a.only_pf else "bench.csv.gz"), index=False)
optm = {c: opt_m(Y) for c, (t, Y, _) in claims.items()}
json.dump(dict(full_cost=full_cost, cE=a.cE, cG=a.cG, cL=a.cL, optimal_m=optm), open(os.path.join(od, "meta.json"), "w"), indent=1)
print(df.groupby(["type", "frac", "design", "estimator"])[["mae_ST", "top_ok", "n_pipelines"]].mean().round(3).to_string())
