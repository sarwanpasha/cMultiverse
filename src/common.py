"""Shared definitions for the scMultiverse experiments.

Normalizations are chosen to span three rank-equivalence classes (see theory):
  class L  (library-size scale family, T(x)=h(x/L) with h increasing): log1p_cp10k, log1p_median, acosh_vst
  class Q  (poscounts size factors, T(x)=h(x/s), s != L):               log1p_poscounts
  class P  (analytic Pearson residuals, depth-dependent offset):          pearson
"""
import os, json
import numpy as np
import scipy.sparse as sp

ROOT = os.environ.get("SCMULTIVERSE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

NORMS = ["log1p_cp10k", "log1p_median", "acosh_vst", "log1p_poscounts", "pearson"]
NORM_CLASS = {"log1p_cp10k": "L", "log1p_median": "L", "acosh_vst": "L",
              "log1p_poscounts": "Q", "pearson": "P"}

DEFAULT_LEVELS = {
    "nhvg":  [500, 1000, 2000, 4000, 8000],
    "norm":  NORMS,
    "ncomp": [10, 20, 30, 50, 100],
    "batch": ["none", "combat", "harmony", "batch_scale", "regress_covariates"],
    "k":     [10, 15, 30, 50, 100, 200],
    "res":   [0.2, 0.5, 0.8, 1.2, 2.0],
}
VALUE_NORMS = NORMS + ["pearson_zt"]  # pearson_zt: Pearson residuals with zero counts tied (theory-derived fix)
FACTORS = ["nhvg", "norm", "ncomp", "batch", "k", "res"]
LEIDEN_SEEDS = [0, 1]


def levels_for(meta):
    lv = {f: list(v) for f, v in DEFAULT_LEVELS.items()}
    if meta.get("n_batches", 0) < 2:
        lv["batch"] = ["none", "regress_covariates"]
    return lv


def apply_quick(lv):
    lv.update(ncomp=[10, 30], k=[15, 50], res=[0.5, 1.2], batch=lv["batch"][:2])
    return lv


def dataset_dir(name):
    return os.path.join(DATA, name)


def load_counts(name):
    d = dataset_dir(name)
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr().astype(np.float32)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    import pandas as pd
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    genes = pd.read_csv(os.path.join(d, "genes.csv"))["gene"].astype(str).values
    return X, obs, genes, meta


# ---------------------------------------------------------------- normalization
def poscounts_size_factors(X):
    """DESeq2 'poscounts' size factors on a sparse cells x genes matrix."""
    X = X.tocsc()
    n = X.shape[0]
    lgm = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        v = X.data[X.indptr[j]:X.indptr[j + 1]]
        lgm[j] = np.log(v).sum() / n if v.size else -np.inf
    X = X.tocsr()
    sf = np.ones(n)
    for i in range(n):
        idx = X.indices[X.indptr[i]:X.indptr[i + 1]]
        v = X.data[X.indptr[i]:X.indptr[i + 1]]
        ok = np.isfinite(lgm[idx]) & (v > 0)
        if ok.any():
            sf[i] = np.exp(np.median(np.log(v[ok]) - lgm[idx][ok]))
    sf = sf / np.exp(np.mean(np.log(sf)))
    return sf


def pearson_residuals(X, theta=100.0):
    """Analytic Pearson residuals (Lause et al. 2021), clipped at sqrt(n). Dense float32."""
    n = X.shape[0]
    L = np.asarray(X.sum(1)).ravel()
    pg = np.asarray(X.sum(0)).ravel() / L.sum()
    mu = np.outer(L, pg).astype(np.float32)
    Xd = X.toarray()
    R = (Xd - mu) / np.sqrt(mu + mu * mu / theta)
    clip = np.sqrt(n)
    np.clip(R, -clip, clip, out=R)
    return R.astype(np.float32)


def normalize(X, norm):
    """Return dense float32 cells x genes transformed matrix."""
    L = np.asarray(X.sum(1)).ravel().astype(np.float64)
    if norm == "pearson":
        return pearson_residuals(X)
    if norm == "pearson_zt":
        R = pearson_residuals(X)
        Z0 = (X.toarray() == 0)
        Rnz = np.where(Z0, np.inf, R)
        floor = Rnz.min(0) - 1.0
        floor[~np.isfinite(floor)] = 0.0
        R[Z0] = np.broadcast_to(floor, R.shape)[Z0]
        return R
    if norm == "log1p_poscounts":
        s = poscounts_size_factors(X)
        Y = sp.diags(1.0 / s) @ X
        return np.log1p(Y.toarray()).astype(np.float32)
    if norm == "log1p_cp10k":
        Y = sp.diags(1e4 / L) @ X
        return np.log1p(Y.toarray()).astype(np.float32)
    if norm == "log1p_median":
        Y = sp.diags(np.median(L) / L) @ X
        return np.log1p(Y.toarray()).astype(np.float32)
    if norm == "acosh_vst":
        alpha = 0.05
        Y = sp.diags(L.mean() / L) @ X
        return (np.arccosh(2 * alpha * Y.toarray() + 1) / np.sqrt(alpha)).astype(np.float32)
    raise ValueError(norm)


def select_hvg(Z, norm, n_top, n_bins=20):
    """Top genes by variance on the transformed scale; for non-Pearson transforms the
    variance is z-scored within mean-expression bins (Seurat-style dispersion)."""
    mean = Z.mean(0)
    var = Z.var(0)
    if norm == "pearson":
        score = var
    else:
        q = np.quantile(mean, np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.searchsorted(q, mean, side="right") - 1, 0, n_bins - 1)
        score = np.zeros_like(var)
        for k in range(n_bins):
            m = b == k
            if m.sum() > 1:
                sd = var[m].std()
                score[m] = (var[m] - var[m].mean()) / (sd if sd > 0 else 1.0)
    score = np.where(var > 0, score, -np.inf)
    return np.sort(np.argsort(-score)[:n_top])


# ---------------------------------------------------------------- batch handling
def batch_matrix(Xh, method, obs, batch_key="batch"):
    """Feature-level correction applied before PCA. Harmony is applied after PCA."""
    if method in ("none", "harmony"):
        return Xh - Xh.mean(0)
    if method == "batch_scale":
        out = np.empty_like(Xh)
        for b in np.unique(obs[batch_key].values):
            m = (obs[batch_key].values == b)
            sub = Xh[m]
            sd = sub.std(0); sd[sd == 0] = 1.0
            out[m] = (sub - sub.mean(0)) / sd
        return out
    if method == "regress_covariates":
        D = np.column_stack([np.ones(len(obs)), np.log1p(obs["total_counts"].values),
                             obs["pct_mt"].values])
        beta, *_ = np.linalg.lstsq(D, Xh, rcond=None)
        return (Xh - D @ beta).astype(np.float32)
    if method == "combat":
        import anndata as ad, scanpy as sc
        a = ad.AnnData(Xh.copy(), obs=obs[[batch_key]].astype(str).copy())
        sc.pp.combat(a, key=batch_key)
        Y = np.asarray(a.X, dtype=np.float32)
        return Y - Y.mean(0)
    raise ValueError(method)


def scvi_embed(Xcounts, obs, n_latent, batch_key="batch", seed=0, max_epochs=200, threads=8):
    """scVI latent space trained on raw counts of the pipeline's HVGs (normalization only enters via the HVG set)."""
    import torch, anndata as ad, scvi
    torch.set_num_threads(threads); scvi.settings.seed = seed
    A = ad.AnnData(Xcounts.copy(), obs=obs[[batch_key]].astype(str).copy())
    scvi.model.SCVI.setup_anndata(A, batch_key=batch_key)
    m = scvi.model.SCVI(A, n_latent=int(n_latent), n_layers=1, n_hidden=128)
    m.train(max_epochs=max_epochs, early_stopping=True, accelerator="cpu", enable_progress_bar=False)
    return np.asarray(m.get_latent_representation(), dtype=np.float32)


def harmony(pcs, obs, batch_key="batch", seed=0):
    import harmonypy
    import pandas as pd
    meta = pd.DataFrame({"batch": obs[batch_key].astype(str).values})
    ho = harmonypy.run_harmony(pcs, meta, ["batch"], max_iter_harmony=10, random_state=seed, verbose=False)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != pcs.shape[0]:
        Z = Z.T
    return Z.astype(np.float32)


# ---------------------------------------------------------------- rank statistics
def avg_ranks(V):
    """Column-wise average ranks (ties -> mean rank). V: cells x genes."""
    from scipy.stats import rankdata
    return rankdata(V, axis=0, method="average").astype(np.float32)


def auc_matrix(ranks, labels):
    """Mann-Whitney AUC of every gene for every cluster (one-vs-rest).
    ranks: cells x genes average ranks; labels: int cluster ids. Returns K x genes."""
    labels = np.asarray(labels)
    K = labels.max() + 1
    N = len(labels)
    oh = np.zeros((K, N), dtype=np.float32)
    oh[labels, np.arange(N)] = 1.0
    Rsum = (oh @ ranks).astype(np.float64)   # dense BLAS (multithreaded)
    nc = np.bincount(labels, minlength=K).astype(np.float64)
    mc = N - nc
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (Rsum - nc[:, None] * (nc[:, None] + 1) / 2) / (nc[:, None] * mc[:, None])
    A[~np.isfinite(A)] = np.nan   # a cluster containing every cell has no defined one-vs-rest AUC
    return A
