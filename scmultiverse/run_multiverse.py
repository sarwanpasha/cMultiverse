#!/usr/bin/env python3
"""
run_multiverse.py — scMultiverse Stage 1 multiverse runner.

Runs a Latin-hypercube-sampled set of pipeline configurations through the
6-step specification lattice and writes per-configuration claim fingerprints,
the lattice definition, a run report, and per-panel-gene expression coverage.

The lattice is dumped to JSON alongside outputs so downstream analysis
scripts can read it from disk rather than importing this file.

Usage:
    python run_multiverse.py --dataset covid_pbmc --n-specs 500 --n-cells 10000 \\
        --seed 42 --output results/pbmc/

    # audit run (forces published-marker genes into the panel):
    python run_multiverse.py --dataset covid_pbmc --n-specs 500 --n-cells 50000 \\
        --seed 42 --force-include-genes C1QA,C1QB,C1QC,FCGR3A,GATA1,MPO,PF4,CD34,CD38 \\
        --output results/pbmc_audit/

    # synthetic dry run for testing:
    python run_multiverse.py --synthetic --n-specs 100 --output results/synth/
"""

from __future__ import annotations
import argparse, json, time, hashlib, warnings, logging, sys
from pathlib import Path
from typing import Any
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multiverse")

# =====================================================================
# Lattice — also dumped to JSON alongside outputs for downstream analysis
# scripts, so they read the lattice from disk rather than importing this file.
# =====================================================================

LATTICE = {
    "n_hvg":            ["h500", "h1000", "h2000", "h4000", "h8000"],
    "normalization":    ["log1p_cp10k", "pearson_residuals", "scran_like", "raw_z", "log1p_median"],
    "n_components":     ["c10", "c20", "c30", "c50", "c100"],
    "batch_correction": ["none", "combat", "harmony", "scaling_per_batch", "regress_out"],
    "k_neighbors":      ["k10", "k15", "k30", "k50", "k100", "k200"],
    "cluster_res":      ["r02", "r05", "r08", "r12", "r20"],
}
STEP_NAMES = list(LATTICE.keys())

LATTICE_INT = {
    "n_hvg":            {"h500": 500, "h1000": 1000, "h2000": 2000, "h4000": 4000, "h8000": 8000},
    "n_components":     {"c10": 10, "c20": 20, "c30": 30, "c50": 50, "c100": 100},
    "k_neighbors":      {"k10": 10, "k15": 15, "k30": 30, "k50": 50, "k100": 100, "k200": 200},
    "cluster_res":      {"r02": 0.2, "r05": 0.5, "r08": 0.8, "r12": 1.2, "r20": 2.0},
}


def latin_hypercube_specs(n: int, rng: np.random.Generator) -> list[dict[str, str]]:
    cols = {}
    for step, options in LATTICE.items():
        n_opts = len(options)
        per_opt = n // n_opts
        remainder = n - per_opt * n_opts
        values = []
        for i, opt in enumerate(options):
            count = per_opt + (1 if i < remainder else 0)
            values.extend([opt] * count)
        rng.shuffle(values)
        cols[step] = values
    return [{step: cols[step][i] for step in STEP_NAMES} for i in range(n)]


def spec_to_hash(spec: dict[str, str]) -> str:
    return hashlib.md5(repr(sorted(spec.items())).encode()).hexdigest()[:12]


# =====================================================================
# Pipeline execution
# =====================================================================

def materialize_and_run(spec: dict[str, str], adata, panel_genes: list[str]) -> dict[str, Any]:
    """Run one full pipeline spec and return a claim fingerprint."""
    import scanpy as sc
    sc.settings.verbosity = 0

    a = adata.copy()

    n_hvg = LATTICE_INT["n_hvg"][spec["n_hvg"]]
    norm = spec["normalization"]

    # ----- Normalization -----
    if norm == "log1p_cp10k":
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    elif norm == "log1p_median":
        med = float(np.median(np.asarray(a.X.sum(axis=1)).ravel()))
        med = max(med, 1.0)  # avoid zero
        sc.pp.normalize_total(a, target_sum=med)
        sc.pp.log1p(a)
    elif norm == "pearson_residuals":
        try:
            sc.experimental.pp.normalize_pearson_residuals(a)
        except Exception:
            sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    elif norm == "scran_like":
        libsize = np.asarray(a.X.sum(axis=1)).ravel().astype(float)
        libsize = np.where(libsize > 0, libsize, 1.0)
        sf = libsize / np.median(libsize)
        if sp.issparse(a.X):
            X = a.X.toarray() / sf[:, None]
            a.X = sp.csr_matrix(X)
        else:
            a.X = a.X / sf[:, None]
        sc.pp.log1p(a)
    elif norm == "raw_z":
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        mu = X.mean(axis=0); sigma = X.std(axis=0) + 1e-6
        a.X = (X - mu) / sigma

    # ----- HVG selection -----
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=min(n_hvg, a.n_vars - 1), flavor="seurat", subset=False)
    except Exception:
        # Fallback by variance
        X = a.X.toarray() if sp.issparse(a.X) else a.X
        var = X.var(axis=0)
        top_idx = np.argsort(-var)[:min(n_hvg, a.n_vars)]
        a.var["highly_variable"] = False
        a.var.iloc[top_idx, a.var.columns.get_loc("highly_variable")] = True

    # KEEP a reference to the pre-HVG-subset normalized data — we score markers
    # on the FULL gene set, not just HVGs. The HVG subset is for PCA/clustering.
    a_full = a.copy()  # normalized, all genes

    a = a[:, a.var["highly_variable"]].copy()
    hvg_set = set(a.var_names)

    # ----- Batch correction -----
    bc = spec["batch_correction"]
    if "batch" not in a.obs.columns:
        a.obs["batch"] = "0"
    a.obs["batch"] = a.obs["batch"].astype("category")

    if bc == "none":
        pass
    elif bc == "regress_out":
        try:
            sc.pp.regress_out(a, ["batch"])
        except Exception:
            pass
    elif bc == "scaling_per_batch":
        try:
            for b in a.obs["batch"].cat.categories:
                mask = (a.obs["batch"] == b).values
                if mask.sum() < 2:
                    continue
                X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
                mu = X[mask].mean(axis=0)
                X[mask] -= mu
                a.X = sp.csr_matrix(X) if sp.issparse(a.X) else X
        except Exception:
            pass
    elif bc == "combat":
        try:
            sc.pp.combat(a, key="batch")
        except Exception:
            pass

    # ----- PCA -----
    n_comp = LATTICE_INT["n_components"][spec["n_components"]]
    n_comp_eff = min(n_comp, a.n_vars - 1, a.n_obs - 1)
    if n_comp_eff < 2:
        raise ValueError(f"n_comp_eff={n_comp_eff} (n_vars={a.n_vars}, n_obs={a.n_obs})")
    try:
        if bc not in ("scaling_per_batch", "regress_out"):
            sc.pp.scale(a, max_value=10)
    except Exception:
        pass
    try:
        sc.tl.pca(a, n_comps=n_comp_eff, svd_solver="arpack")
    except Exception:
        sc.tl.pca(a, n_comps=min(10, n_comp_eff))

    # ----- Harmony -----
    if bc == "harmony":
        try:
            import harmonypy as hm
            ho = hm.run_harmony(a.obsm["X_pca"], a.obs, "batch", max_iter_harmony=10)
            a.obsm["X_pca"] = ho.Z_corr.T
        except Exception as e:
            log.debug(f"Harmony failed: {e}")

    # ----- Neighbors + Leiden -----
    k = LATTICE_INT["k_neighbors"][spec["k_neighbors"]]
    k_eff = min(k, a.n_obs - 1)
    sc.pp.neighbors(a, n_neighbors=k_eff, use_rep="X_pca")
    res = LATTICE_INT["cluster_res"][spec["cluster_res"]]
    sc.tl.leiden(a, resolution=res, flavor="leidenalg")

    # ----- Fingerprint -----
    fp = {}
    cluster_labels = a.obs["leiden"].astype(str).values
    n_clust = len(set(cluster_labels))
    fp["n_clusters"] = int(n_clust)

    sizes = pd.Series(cluster_labels).value_counts().values.astype(float)
    log_sizes = np.log10(sizes + 1)
    hist, _ = np.histogram(log_sizes, bins=10, range=(0, 4))
    for i, h in enumerate(hist):
        fp[f"cluster_size_bin_{i}"] = int(h)

    panel_set = set(panel_genes)
    if len(hvg_set) > 0:
        fp["hvg_panel_jaccard"] = len(hvg_set & panel_set) / len(hvg_set | panel_set)
    else:
        fp["hvg_panel_jaccard"] = 0.0

    # Score markers using Wilcoxon AUC, then call markers by WITHIN-SPEC quantile.
    #
    # We computed three previous attempts (z-score, LFC, absolute-AUC) and each
    # produced different per-normalization biases on real data:
    #   - z-score scales with cluster count
    #   - LFC needs non-negative input
    #   - absolute-AUC threshold of 0.70 gives 22-178 markers depending on
    #     normalization (pearson_residuals inflates dramatically)
    #
    # Quantile-based marker calling sidesteps this: flag the top X% of panel
    # genes by max-cluster AUC, WITHIN EACH SPEC. Every spec has the same
    # marker count by construction, so the fingerprint measures which genes
    # get flagged, not how many.
    #
    # This is the correct semantics for the multiverse: we want to know which
    # genes flip in/out of marker status across pipeline choices, not whether
    # one pipeline produces more markers in absolute terms.
    panel_in_var = [g for g in panel_genes if g in a_full.var_names]
    if len(panel_in_var) > 0 and n_clust >= 2:
        sub = a_full[:, panel_in_var]
        X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
        # Rank cells per gene (rankdata supports axis= for vectorization)
        from scipy.stats import rankdata
        n_cells, n_genes = X.shape
        ranks = rankdata(X, axis=0)  # shape (n_cells, n_genes), vectorized
        # AUC per cluster per gene
        max_auc = np.zeros(n_genes)
        for c in sorted(set(cluster_labels)):
            mask = (cluster_labels == c)
            n_in = mask.sum()
            n_out = n_cells - n_in
            if n_in < 3 or n_out < 3:
                continue
            rank_sum_in = ranks[mask].sum(axis=0)
            U = rank_sum_in - n_in * (n_in + 1) / 2
            auc_c = U / (n_in * n_out)
            max_auc = np.maximum(max_auc, auc_c)
        # Quantile-based marker call: flag top 20% of panel genes by max-cluster AUC.
        # 20% chosen as ~38 of 190 markers per spec — stable count for Sobol,
        # variable enough across specs.
        n_flag = max(5, int(0.20 * n_genes))
        threshold = np.partition(max_auc, -n_flag)[-n_flag]
        # Use strict > to avoid flagging large numbers of tied AUCs at the
        # threshold value (which happens when many genes have AUC=0.5 exactly).
        is_marker_any = (max_auc > threshold).astype(int)
        # If fewer than n_flag flagged due to ties at threshold, top-up with
        # genes at exactly the threshold (deterministic by gene order)
        n_actual = int(is_marker_any.sum())
        if n_actual < n_flag:
            tie_indices = np.where(max_auc == threshold)[0]
            need = n_flag - n_actual
            for idx in tie_indices[:need]:
                is_marker_any[idx] = 1
        for gi, g in enumerate(panel_in_var):
            fp[f"is_marker__{g}"] = int(is_marker_any[gi])
    for g in panel_genes:
        fp.setdefault(f"is_marker__{g}", 0)

    return fp


# =====================================================================
# Dataset loading — robust to CELLxGENE renames
# =====================================================================

def load_panel_genes(adata, n_panel: int = 200, force_include=None) -> list[str]:
    """Pick a marker-like panel by running ONE baseline pipeline and selecting
    genes whose marker-test z-score is in a "borderline" range — strong enough
    to be called a marker under some pipeline choices, weak enough to be
    un-flagged under others.

    Why this matters: previous panel selectors picked genes by CV across cells.
    On heterogeneous tissue (e.g. HLCA with epithelial + endothelial + stromal +
    immune cells), the CV-selected genes are *cell-type-specific* and trivially
    always-called regardless of pipeline choices. The marker fingerprint becomes
    constant across specs and provides no signal.

    The fix: pick genes from the BORDERLINE of the marker-significance
    distribution. These are exactly the claims whose status depends on
    pipeline choices — which is what the multiverse is supposed to be testing.

    Method:
      1. Normalize, HVG-select, PCA, neighbors, Leiden with a fixed-baseline spec.
      2. For each cluster, compute per-gene z-score vs the rest.
      3. Pool genes with max-across-clusters z-score in [1.5, 4.0].
      4. Return the top n_panel.
    """
    import scanpy as sc
    sc.settings.verbosity = 0

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"Empty dataset: {adata.n_obs} cells × {adata.n_vars} genes")

    log.info(f"      computing baseline panel from {adata.n_obs} cells × {adata.n_vars} genes")
    a = adata.copy()

    # Baseline pipeline: a reasonable middle-of-the-road choice
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat", subset=False)
    a_full = a.copy()  # keep all genes after normalization for marker scoring
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(a, max_value=10)
    n_comp = min(30, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comp, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), use_rep="X_pca")
    sc.tl.leiden(a, resolution=0.8, flavor="leidenalg")
    cluster_labels = a.obs["leiden"].astype(str).values
    n_clust = len(set(cluster_labels))
    log.info(f"      baseline pipeline produced {n_clust} clusters")
    if n_clust < 2:
        log.warning("      <2 clusters from baseline; cannot compute marker panel; falling back to CV")
        return _fallback_cv_panel(adata, n_panel)

    # Score every gene's marker LFC on the FULL gene set (not just HVGs).
    # We use log fold change per cluster, NOT a z-score. LFC is the standard
    # scRNA-seq marker metric and is unbounded, giving a real gradient between
    # "weak" and "strong" markers.
    X = a_full.X.toarray() if sp.issparse(a_full.X) else np.asarray(a_full.X)
    cluster_means = np.zeros((n_clust, a_full.n_vars))
    for ci, c in enumerate(sorted(set(cluster_labels))):
        mask = cluster_labels == c
        cluster_means[ci] = X[mask].mean(axis=0)
    # log fold change: log((mean_in_cluster + eps) / (mean_in_others + eps))
    # We compute "mean in others" by removing the focal cluster and averaging
    total_sum = cluster_means.sum(axis=0)  # sum across clusters per gene
    # mean_others[c,g] = (total_sum[g] - cluster_means[c,g]) / (n_clust - 1)
    mean_others = (total_sum[None, :] - cluster_means) / max(n_clust - 1, 1)
    eps = 1e-3
    lfc = np.log2((cluster_means + eps) / (mean_others + eps))  # (n_clust, n_genes)
    max_lfc = lfc.max(axis=0)  # per-gene best LFC across clusters

    log.info(f"      LFC distribution: "
             f"p25={np.percentile(max_lfc, 25):.2f}, "
             f"p50={np.percentile(max_lfc, 50):.2f}, "
             f"p75={np.percentile(max_lfc, 75):.2f}, "
             f"p95={np.percentile(max_lfc, 95):.2f}, "
             f"max={max_lfc.max():.2f}")

    # Per-cluster median-rank panel selection.
    #
    # For each baseline cluster, the gene LFC scores span a wide range. Top-N
    # genes are saturated markers (always flagged); bottom-N are noise. The
    # *median-rank* genes (per cluster) are the ones whose marker status could
    # plausibly flip across pipeline choices.
    #
    # Algorithm:
    #   For each cluster c:
    #     Find genes where c is the max-LFC cluster
    #     Within those, sort by LFC[c]
    #     Take ~n_panel/n_clust genes from the MIDDLE of that ranking

    n_per_cluster = max(3, n_panel // n_clust)
    log.info(f"      panel selection: ~{n_per_cluster} genes from middle ranks of each "
             f"of {n_clust} clusters")

    panel_ids = []
    for ci in range(n_clust):
        # Genes where cluster ci has the highest LFC
        is_max_cluster = (lfc.argmax(axis=0) == ci)
        # Only consider genes that are at least mildly upregulated (LFC > 0.5)
        candidate = np.where(is_max_cluster & (lfc[ci] > 0.5))[0]
        if len(candidate) < 5:
            continue
        # Sort by LFC[ci] ASCENDING within this cluster
        sorted_cand = candidate[np.argsort(lfc[ci][candidate])]
        n_take = min(n_per_cluster, len(sorted_cand))
        if n_take < 1:
            continue
        # Take MIDDLE n_take genes
        mid = len(sorted_cand) // 2
        start = max(0, mid - n_take // 2)
        end = min(len(sorted_cand), start + n_take)
        panel_ids.extend(sorted_cand[start:end].tolist())

    panel_ids = list(dict.fromkeys(panel_ids))

    if len(panel_ids) < 20:
        log.warning(f"      only {len(panel_ids)} genes selected; falling back to CV panel")
        return _fallback_cv_panel(adata, n_panel)

    if len(panel_ids) > n_panel:
        rng = np.random.default_rng(0)
        panel_ids = rng.choice(panel_ids, size=n_panel, replace=False).tolist()

    panel = [a_full.var_names[i] for i in panel_ids]
    selected_lfc = max_lfc[np.array(panel_ids)]
    log.info(f"      selected panel of {len(panel)} per-cluster middle-rank genes "
             f"(LFC range: {selected_lfc.min():.2f} to {selected_lfc.max():.2f}, "
             f"median {np.median(selected_lfc):.2f})")

    # Force-include audit genes (e.g. published marker claims). These are
    # prepended and guaranteed present in the panel regardless of their
    # baseline ranking, so their reproducibility can be audited across specs.
    if force_include:
        present = [g for g in force_include if g in set(a_full.var_names)]
        missing = [g for g in force_include if g not in set(a_full.var_names)]
        if missing:
            log.warning(f"      force_include: {len(missing)} requested genes not in "
                        f"dataset and will be skipped: {missing}")
        if present:
            # Drop any already in the auto panel to avoid duplicates, then
            # prepend forced genes and trim the auto panel so total == n_panel.
            auto = [g for g in panel if g not in set(present)]
            keep_auto = max(0, n_panel - len(present))
            panel = present + auto[:keep_auto]
            log.info(f"      force-included {len(present)} audit genes; "
                     f"panel now {len(panel)} genes ({len(present)} forced + "
                     f"{len(panel) - len(present)} auto)")
    return panel


def _fallback_cv_panel(adata, n_panel: int = 200) -> list[str]:
    """Fallback panel by CV when the baseline-marker approach doesn't work
    (e.g. baseline clustering fails). Keeps the previous CV-based logic."""
    X = adata.X
    if sp.issparse(X):
        n_expr = np.asarray((X > 0).sum(axis=0)).ravel()
        means = np.asarray(X.mean(axis=0)).ravel()
        sqmeans = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = sqmeans - means ** 2
    else:
        X = np.asarray(X)
        n_expr = (X > 0).sum(axis=0)
        means = X.mean(axis=0)
        var = X.var(axis=0)
    expr_frac = n_expr / adata.n_obs
    cv = np.sqrt(var) / (means + 1e-6)
    mask = (expr_frac >= 0.05) & (expr_frac <= 0.50) & (means > 0.1)
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) < n_panel:
        mask = (expr_frac > 0.01) & (means > 0)
        candidate_idx = np.where(mask)[0]
    if len(candidate_idx) < 10:
        raise ValueError(f"Too few usable panel candidates: {len(candidate_idx)}")
    ranked = candidate_idx[np.argsort(-cv[candidate_idx])]
    top = ranked[:min(n_panel, len(ranked))]
    return [adata.var_names[i] for i in top]


def make_synthetic_anndata(n_cells: int, n_genes: int = 3000, n_groups: int = 8, seed: int = 0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, size=n_cells)
    batch = rng.integers(0, 3, size=n_cells)
    X = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(np.float32)
    for g in range(n_groups):
        mask = group == g
        marker_idx = rng.choice(n_genes, size=10, replace=False)
        for mi in marker_idx:
            X[mask, mi] += rng.negative_binomial(20, 0.4, size=mask.sum()).astype(np.float32)
    var = pd.DataFrame(index=[f"GENE{i:04d}" for i in range(n_genes)])
    obs = pd.DataFrame({
        "true_group": group.astype(str),
        "batch": batch.astype(str),
    }, index=[f"cell_{i}" for i in range(n_cells)])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


# =====================================================================
# Dataset registry — defines how to load each supported dataset
# =====================================================================

DATASET_REGISTRY = {
    "covid_pbmc": {
        "description": "COVID-19 PBMC (immune cells from blood, COVID patients)",
        "strategies": [
            ("disease=COVID-19 + tissue=blood", "disease == 'COVID-19' and tissue_general == 'blood'"),
            ("disease=COVID-19", "disease == 'COVID-19'"),
            ("tissue=blood (healthy fallback)", "tissue_general == 'blood' and disease == 'normal'"),
        ],
        "preferred_batch_keys": ["donor_id", "dataset_id", "assay"],
        "stratify_by": None,  # No stratification needed; relatively uniform
    },
    "hlca_lung": {
        "description": "Human Lung Cell Atlas — healthy lung, multi-study integration",
        "strategies": [
            ("tissue=lung + disease=normal", "tissue_general == 'lung' and disease == 'normal'"),
            ("tissue=lung", "tissue_general == 'lung'"),
        ],
        # HLCA's dominant batch effect is across the studies that were integrated.
        # dataset_id captures this; donor_id is too granular.
        "preferred_batch_keys": ["dataset_id", "donor_id", "assay"],
        "stratify_by": "dataset_id",  # Subsample across studies to retain batch structure
    },
    "tabula_immune": {
        "description": "Tabula Sapiens immune cells",
        "strategies": [
            ("Tabula Sapiens blood + immune", "tissue_general == 'blood' and dataset_id contains 'Tabula Sapiens'"),
            ("tissue=blood healthy", "tissue_general == 'blood' and disease == 'normal'"),
        ],
        "preferred_batch_keys": ["donor_id", "assay"],
        "stratify_by": "donor_id",
    },
}


def _select_batch_key(adata, preferred_keys, min_cells_per_batch=30,
                      max_batches=600):
    """Choose a batch key with a healthy cells-per-batch ratio.

    Scores each candidate column by how well it lands in the target regime:
      - at least `min_cells_per_batch` cells per batch (so ComBat/Harmony can
        estimate batch parameters), AND
      - no more than `max_batches` total levels.

    Preference order:
      1. Among keys meeting BOTH constraints, pick the one EARLIEST in
         preferred_keys (the dataset author's intended primary batch variable).
      2. If none meet both, pick the key whose cells-per-batch is closest to a
         target of ~130 (the HLCA value we know works), but log a warning.
      3. If no key has >1 level, return None (single batch).
    """
    candidates = []
    for col in preferred_keys:
        if col not in adata.obs.columns:
            continue
        n_lev = adata.obs[col].nunique()
        if n_lev < 2:
            continue
        cpb = adata.n_obs / n_lev
        candidates.append((col, n_lev, cpb))

    if not candidates:
        return None

    # Tier 1: keys meeting both constraints, in preference order
    for col, n_lev, cpb in candidates:
        if cpb >= min_cells_per_batch and n_lev <= max_batches:
            return col

    # Tier 2: none ideal. Pick closest to target cells-per-batch = 130, warn.
    target = 130.0
    best = min(candidates, key=lambda c: abs(c[2] - target))
    col, n_lev, cpb = best
    log.warning(f"    -> no batch key in ideal range; best available is "
                f"'{col}' with {n_lev} levels ({cpb:.0f} cells/batch). "
                f"batch_correction methods may behave suboptimally.")
    return col


def load_cellxgene_dataset(dataset_name: str, n_cells: int, seed: int = 0):
    """Generic CELLxGENE loader driven by the registry.

    Returns AnnData on success, None on failure. Logs every attempt for
    diagnostic transparency.
    """
    if dataset_name not in DATASET_REGISTRY:
        log.error(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY)}")
        return None

    spec = DATASET_REGISTRY[dataset_name]
    log.info(f"Dataset: {dataset_name} — {spec['description']}")

    try:
        import cellxgene_census
    except ImportError:
        log.warning("cellxgene_census not installed; cannot fetch from CELLxGENE")
        return None

    rng = np.random.default_rng(seed)

    for label, value_filter in spec["strategies"]:
        try:
            log.info(f"  CELLxGENE strategy: {label}")
            with cellxgene_census.open_soma(census_version="2025-11-08") as census:
                obs_meta = cellxgene_census.get_obs(
                    census,
                    organism="Homo sapiens",
                    value_filter=value_filter,
                    column_names=["soma_joinid", "dataset_id", "donor_id",
                                  "tissue_general", "disease", "assay"],
                )
                n_matching = len(obs_meta)
                log.info(f"    -> matching cells in census: {n_matching}")
                if n_matching < 1000:
                    log.warning(f"    -> too few matching cells; trying next strategy")
                    continue

                # CRITICAL: cellxgene_census.get_obs may return a TileDB-backed
                # dataframe where boolean masking on string columns doesn't
                # fully materialize. Force conversion to a real pandas DataFrame
                # before subsampling.
                if not isinstance(obs_meta, pd.DataFrame) or not all(
                    isinstance(obs_meta[c], pd.Series) for c in obs_meta.columns
                ):
                    log.info(f"    -> materializing obs_meta as pandas (was {type(obs_meta).__name__})")
                obs_meta = pd.DataFrame({c: np.asarray(obs_meta[c]) for c in obs_meta.columns})
                # Sanity check: column lengths should match the row count
                assert len(obs_meta) == n_matching, (
                    f"obs_meta length mismatch after materialization: "
                    f"{len(obs_meta)} != {n_matching}"
                )
                log.info(f"    -> obs_meta materialized: {len(obs_meta)} rows × {len(obs_meta.columns)} cols")

                # Stratified subsampling if requested
                if spec["stratify_by"] is not None and spec["stratify_by"] in obs_meta.columns:
                    strat_key = spec["stratify_by"]
                    log.info(f"    -> stratifying subsample by {strat_key} "
                             f"({obs_meta[strat_key].nunique()} levels)")
                    log.info(f"    -> obs_meta type={type(obs_meta).__name__}, "
                             f"len={len(obs_meta)}, "
                             f"soma_joinid dtype={obs_meta['soma_joinid'].dtype}")
                    chosen = _stratified_subsample(obs_meta, strat_key, n_cells, rng)
                else:
                    joinids = obs_meta["soma_joinid"].values
                    if len(joinids) > n_cells * 20:
                        prelim = rng.choice(joinids, size=n_cells * 20, replace=False)
                    else:
                        prelim = joinids
                    if len(prelim) > n_cells:
                        chosen = rng.choice(prelim, size=n_cells, replace=False)
                    else:
                        chosen = prelim

                log.info(f"    -> pulling {len(chosen)} cells from census")
                adata = cellxgene_census.get_anndata(
                    census,
                    organism="Homo sapiens",
                    obs_coords=np.sort(chosen),
                )
                log.info(f"    -> got {adata.n_obs} cells × {adata.n_vars} genes")

                # Use symbol names if available
                if "feature_name" in adata.var.columns:
                    adata.var_names = adata.var["feature_name"].astype(str)
                    adata.var_names_make_unique()

                # Drop all-zero genes
                if sp.issparse(adata.X):
                    gene_sum = np.asarray(adata.X.sum(axis=0)).ravel()
                else:
                    gene_sum = adata.X.sum(axis=0)
                keep = gene_sum > 0
                log.info(f"    -> keeping {keep.sum()} non-zero genes")
                adata = adata[:, keep].copy()

                # Pick the best batch key. Previous logic took the FIRST preferred
                # key with >1 level, which on COVID PBMC resolved to donor_id with
                # 525 levels (~19 cells/batch) — too granular for ComBat/Harmony to
                # estimate batch parameters, silently neutering the batch_correction
                # step. We now score each candidate by cells-per-batch and pick the
                # one in a healthy range (target >= 30 cells/batch, ideally 50-500
                # batches total).
                chosen_batch_key = _select_batch_key(adata, spec["preferred_batch_keys"])
                if chosen_batch_key is not None:
                    adata.obs["batch"] = adata.obs[chosen_batch_key].astype(str)
                    n_lev = adata.obs["batch"].nunique()
                    cpb = adata.n_obs / max(n_lev, 1)
                    log.info(f"    -> using {chosen_batch_key} as batch key "
                             f"({n_lev} levels, {cpb:.0f} cells/batch)")
                else:
                    adata.obs["batch"] = "single_batch"
                    log.warning(f"    -> no usable batch column; using single batch")

                if adata.n_obs >= max(500, int(0.5 * n_cells)):
                    return adata
                else:
                    log.warning(f"    -> got only {adata.n_obs} cells "
                                f"(requested {n_cells}); trying next strategy")
                    continue

        except Exception as e:
            log.warning(f"    -> strategy failed: {e}")
            continue

    log.error(f"All CELLxGENE strategies failed for {dataset_name}")
    return None


def _stratified_subsample(obs_meta, strat_key: str, n_cells: int, rng):
    """Subsample n_cells across the levels of strat_key, guaranteeing every
    level is represented by at least min_per_level cells.

    Algorithm:
      1. Reserve `min_per_level * n_levels` cells for the floor.
      2. Allocate the remaining budget proportionally to level size.
      3. Sample without replacement per level.

    This is correct in expectation and guarantees every batch appears in the
    output, which is essential for downstream batch-correction methods.
    """
    levels = obs_meta[strat_key].value_counts()
    n_levels = len(levels)
    min_per_level = max(20, n_cells // (n_levels * 5))

    # Defensive sanity check: value_counts sum must equal row count.
    # If it doesn't, the dataframe is not fully materialized and we cannot
    # proceed safely.
    if int(levels.sum()) != len(obs_meta):
        log.error(f"      VALUE_COUNTS MISMATCH: levels sum to {int(levels.sum())} "
                  f"but obs_meta has {len(obs_meta)} rows. "
                  f"The dataframe may not be fully materialized. Aborting subsample.")
        raise ValueError(
            f"obs_meta value_counts mismatch: sum={int(levels.sum())}, "
            f"len={len(obs_meta)}. Likely a TileDB/SOMA lazy-evaluation issue."
        )

    log.info(f"      _stratified_subsample: n_cells={n_cells}, "
             f"n_levels={n_levels}, min_per_level={min_per_level}, "
             f"total_in_meta={len(obs_meta)}")
    log.info(f"      level size range: min={int(levels.min())}, "
             f"max={int(levels.max())}, median={int(levels.median())}")

    floor_total = min_per_level * n_levels
    if floor_total >= n_cells:
        # Budget too tight; just give equal slices, capped at level size
        per_level = n_cells // n_levels
        log.info(f"      floor_total={floor_total} >= n_cells={n_cells}; "
                 f"falling back to equal slices ({per_level}/level)")
        chosen_ids = []
        for level in levels.index:
            sub_pool = obs_meta.loc[obs_meta[strat_key] == level, "soma_joinid"].values
            take = min(per_level, len(sub_pool))
            sel = rng.choice(sub_pool, size=take, replace=False)
            chosen_ids.extend(sel.tolist())
        return np.array(chosen_ids)

    remaining_budget = n_cells - floor_total
    total_cells = len(obs_meta)
    chosen_ids = []
    leftover = 0  # cells we couldn't allocate to a small pool, redistribute
    per_level_taken = {}
    for level, count in levels.items():
        sub_pool = obs_meta.loc[obs_meta[strat_key] == level, "soma_joinid"].values
        prop_extra = int(round(remaining_budget * count / total_cells))
        target = min_per_level + prop_extra
        take = min(target, len(sub_pool))
        if take < target:
            leftover += (target - take)
        sel = rng.choice(sub_pool, size=take, replace=False)
        chosen_ids.extend(sel.tolist())
        per_level_taken[level] = take

    log.info(f"      after proportional pass: total_chosen={len(chosen_ids)}, "
             f"leftover_to_redistribute={leftover}")

    # Redistribute leftover into the largest underfilled batches
    if leftover > 0:
        # Map level -> already chosen size
        chosen_arr = np.array(chosen_ids)
        chosen_set = set(chosen_arr.tolist())
        for level, count in levels.items():
            if leftover <= 0:
                break
            sub_pool = obs_meta.loc[obs_meta[strat_key] == level, "soma_joinid"].values
            # Vectorized set-difference is much faster than list comprehension
            sub_pool_arr = np.asarray(sub_pool)
            mask = ~np.isin(sub_pool_arr, list(chosen_set))
            available = sub_pool_arr[mask]
            take_extra = min(leftover, len(available))
            if take_extra > 0:
                sel = rng.choice(available, size=take_extra, replace=False)
                chosen_ids.extend(sel.tolist())
                chosen_set.update(sel.tolist())
                leftover -= take_extra

    chosen_arr = np.array(chosen_ids)
    log.info(f"      _stratified_subsample returning {len(chosen_arr)} cells "
             f"(requested {n_cells}, hit-rate {100*len(chosen_arr)/n_cells:.0f}%)")

    # Hard sanity check: if we got much fewer than requested, something is wrong.
    if len(chosen_arr) < 0.5 * n_cells:
        log.error(f"      STRATIFIED SUBSAMPLE FAILED: got {len(chosen_arr)} cells "
                  f"out of {n_cells} requested. Investigate before continuing.")
        log.error(f"      Sample of per-level takes (first 10): "
                  f"{dict(list(per_level_taken.items())[:10])}")

    return chosen_arr


# Keep the old name as an alias for backwards compatibility
def load_cellxgene_covid_pbmc(n_cells: int, seed: int = 0):
    return load_cellxgene_dataset("covid_pbmc", n_cells, seed)


def load_dataset(n_cells: int, synthetic: bool, dataset_name: str = "covid_pbmc",
                 seed: int = 0, allow_synthetic_fallback: bool = False):
    if synthetic:
        log.info(f"Loading synthetic dataset: {n_cells} cells × 3000 genes")
        return make_synthetic_anndata(n_cells=n_cells, n_genes=3000, n_groups=8, seed=seed)

    log.info(f"Attempting CELLxGENE Census load for {dataset_name}...")
    adata = load_cellxgene_dataset(dataset_name, n_cells, seed=seed)
    if adata is None or adata.n_obs == 0:
        if allow_synthetic_fallback:
            log.warning("CELLxGENE load failed; falling back to synthetic data "
                        "(allow_synthetic_fallback=True)")
            return make_synthetic_anndata(n_cells=n_cells, n_genes=3000, n_groups=8, seed=seed)
        else:
            log.error(f"CELLxGENE load FAILED for {dataset_name} and synthetic "
                      f"fallback is disabled. Pass --synthetic to use synthetic "
                      f"data explicitly, or fix the CELLxGENE access issue.")
            raise RuntimeError(f"Failed to load real {dataset_name} dataset")
    return adata


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specs", type=int, default=500)
    ap.add_argument("--n-cells", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="results")
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip CELLxGENE and use synthetic data")
    ap.add_argument("--dataset", type=str, default="covid_pbmc",
                    choices=list(DATASET_REGISTRY.keys()),
                    help=f"Which dataset to load. Available: {list(DATASET_REGISTRY.keys())}")
    ap.add_argument("--allow-synthetic-fallback", action="store_true",
                    help="If CELLxGENE load fails, fall back to synthetic data "
                         "instead of aborting. Default OFF for safety.")
    ap.add_argument("--force-include-genes", type=str, default=None,
                    help="Comma-separated gene symbols to force into the marker "
                         "panel (e.g. published marker claims for an audit). "
                         "Genes absent from the dataset are skipped with a warning.")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    log.info("=" * 70)
    log.info("Step 0: dump lattice definition for downstream scripts")
    log.info("=" * 70)
    lattice_json = {
        "LATTICE": LATTICE,
        "STEP_NAMES": STEP_NAMES,
        "LATTICE_INT": LATTICE_INT,
        "dataset_name": args.dataset,
        "n_cells_requested": args.n_cells,
        "seed": args.seed,
    }
    (out / "lattice.json").write_text(json.dumps(lattice_json, indent=2))

    log.info(f"Step 1: load dataset ({args.dataset})")
    adata = load_dataset(args.n_cells, synthetic=args.synthetic,
                         dataset_name=args.dataset, seed=args.seed,
                         allow_synthetic_fallback=args.allow_synthetic_fallback)
    log.info(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    if adata.n_obs < 500 or adata.n_vars < 500:
        log.error("Dataset too small to run the multiverse; aborting")
        sys.exit(1)

    force_genes = None
    if args.force_include_genes:
        force_genes = [g.strip() for g in args.force_include_genes.split(",") if g.strip()]
        log.info(f"Force-include audit genes requested ({len(force_genes)}): {force_genes}")
    panel_genes = load_panel_genes(adata, n_panel=200, force_include=force_genes)
    log.info(f"Panel: {len(panel_genes)} genes")

    # Record the fraction of cells expressing each panel gene (raw, pre-pipeline).
    # The audit uses this to gate claims: a gene expressed in too few cells cannot
    # be reliably called a marker, and its low reproducibility reflects coverage,
    # not pipeline fragility.
    panel_in_var = [g for g in panel_genes if g in set(adata.var_names)]
    sub = adata[:, panel_in_var]
    Xp = sub.X
    if sp.issparse(Xp):
        n_expr = np.asarray((Xp > 0).sum(axis=0)).ravel()
    else:
        n_expr = (np.asarray(Xp) > 0).sum(axis=0)
    expr_frac = {g: float(n_expr[i] / adata.n_obs) for i, g in enumerate(panel_in_var)}
    expr_count = {g: int(n_expr[i]) for i, g in enumerate(panel_in_var)}
    coverage = {"n_cells_total": int(adata.n_obs),
                "expressing_cells": expr_count,
                "expressing_fraction": expr_frac}
    (out / "gene_expression_fraction.json").write_text(json.dumps(coverage, indent=2))
    log.info(f"  Wrote {out / 'gene_expression_fraction.json'} "
             f"({len(expr_frac)} panel genes, {adata.n_obs} cells)")

    log.info(f"Step 2: sample {args.n_specs} specs via Latin hypercube")
    specs = latin_hypercube_specs(args.n_specs, rng)
    specs_df = pd.DataFrame(specs)
    specs_df.to_parquet(out / "specs.parquet")
    log.info(f"  Wrote {out / 'specs.parquet'}")

    log.info("Step 3: run each spec and compute fingerprint")
    fingerprints = []
    fail_count = 0
    t_start = time.time()
    for i, spec in enumerate(specs):
        try:
            fp = materialize_and_run(spec, adata, panel_genes)
            fp["spec_id"] = i
            fp["spec_hash"] = spec_to_hash(spec)
            fp["status"] = "ok"
            fingerprints.append(fp)
        except Exception as e:
            fail_count += 1
            log.warning(f"  spec {i} failed: {e}")
            fingerprints.append({"spec_id": i, "spec_hash": spec_to_hash(spec), "status": f"fail: {type(e).__name__}: {e}"})

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (args.n_specs - i - 1) / rate
            log.info(f"  [{i+1}/{args.n_specs}] {rate:.2f} specs/s, ETA {eta/60:.1f} min, fails: {fail_count}")

    # Save fingerprints — pad with NaN to ensure all rows have all columns
    all_keys = set()
    for fp in fingerprints:
        all_keys.update(fp.keys())
    for fp in fingerprints:
        for k in all_keys:
            fp.setdefault(k, None)
    fp_df = pd.DataFrame(fingerprints)
    fp_df.to_parquet(out / "fingerprints.parquet")
    log.info(f"  Wrote {out / 'fingerprints.parquet'}")

    # Report summary
    is_synthetic = args.synthetic or (
        adata.n_vars == 3000 and "true_group" in adata.obs.columns
    )
    report = {
        "n_specs_requested": args.n_specs,
        "n_specs_succeeded": len(fingerprints) - fail_count,
        "n_specs_failed": fail_count,
        "fail_fraction": fail_count / max(args.n_specs, 1),
        "dataset_name": args.dataset,
        "dataset_n_cells": int(adata.n_obs),
        "dataset_n_genes": int(adata.n_vars),
        "n_batches": int(adata.obs["batch"].nunique()) if "batch" in adata.obs else 1,
        "used_synthetic": bool(is_synthetic),
        "wall_time_minutes": (time.time() - t_start) / 60,
        "lattice_steps": STEP_NAMES,
        "seed": args.seed,
    }
    (out / "run_report.json").write_text(json.dumps(report, indent=2))
    log.info("=" * 70)
    log.info(f"DONE: {report['n_specs_succeeded']} ok, {fail_count} failed")
    log.info(f"  -> {out / 'specs.parquet'}")
    log.info(f"  -> {out / 'fingerprints.parquet'}")
    log.info(f"  -> {out / 'lattice.json'}")
    log.info(f"  -> {out / 'run_report.json'}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
