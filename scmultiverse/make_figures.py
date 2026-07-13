#!/usr/bin/env python3
"""
make_figures.py — generate the four publication figures for the scMultiverse
manuscript, with large, legible text suitable for print.

Produces, in ./Figures/:
  figure_sampler_regimes.png       Fig. 1  three-panel active-sampler benchmark
  figure_specification_recipe.png  Fig. 2  PBMC vs HLCA specification recipe
  figure_pairwise_interactions.png Fig. 3  PBMC vs HLCA pairwise interactions
  figure_published_audit.png       Fig. 4  published-claim audit heatmap

Figure 1 is plotted from the raw per-budget CSVs written by the active-sampler
benchmark. Figures 2-4 use the reconciled numbers reported in the manuscript
tables (Sobol indices, pairwise Vjk, audit reproducibility fractions).

Usage:
    python make_figures.py --outdir Figures --sampler-csv-dir results/sampler/
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# ----------------------------------------------------------------------
# Global style: large, legible type for print.
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
    "axes.linewidth": 1.1,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

STEP_LABELS = ["n_hvg", "normalization", "n_components",
               "batch_correction", "k_neighbors", "cluster_res"]
STEP_LABELS_PRETTY = ["HVG count", "Normalization", "PCs",
                      "Batch corr.", "k neighbors", "Resolution"]


# ======================================================================
# FIGURE 1 — active-sampler regimes (Exp A / B / C), from real CSVs
# ======================================================================
def _load_sampler_csv(results_dir, fname):
    """Aggregate one experiment CSV (sampler,n,err,replicate) across replicates.
    Returns (budgets, {sampler: (mean, lo, hi)}) where lo/hi are the
    across-replicate min/max."""
    import pandas as pd
    df = pd.read_csv(Path(results_dir) / fname)
    budgets = sorted(df["n"].unique())
    out = {}
    for s in sorted(df["sampler"].unique()):
        d = df[df["sampler"] == s]
        mean = np.array([d[d["n"] == n]["err"].mean() for n in budgets])
        lo = np.array([d[d["n"] == n]["err"].min() for n in budgets])
        hi = np.array([d[d["n"] == n]["err"].max() for n in budgets])
        out[s] = (mean, lo, hi)
    return budgets, out


def figure_sampler_regimes(outdir: Path, results_dir="results/sampler"):
    files = {
        "A": ("experiment_A_first_order.csv",
              "A. First-order Sobol\n(additive response)",
              "Negligible advantage"),
        "B": ("experiment_B_total_order.csv",
              "B. Total-order Sobol\n(interacting response)",
              "9\u201328% lower error"),
        "C": ("experiment_C_rare_events.csv",
              "C. Rare-event stability\n(binary claims)",
              "20\u201349% lower error\n(small\u2013moderate budgets)"),
    }
    # color/marker/label per sampler
    style = {
        "lhs":          ("#777777", "o", "Latin hypercube"),
        "bald":         ("#e08214", "^", "BALD (active)"),
        "scmultiverse": ("#1b6ca8", "s", "scMultiverse (active)"),
    }
    order = ["lhs", "bald", "scmultiverse"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6), constrained_layout=True)

    for ax, key in zip(axes, ["A", "B", "C"]):
        fname, title, note = files[key]
        budgets, data = _load_sampler_csv(results_dir, fname)
        for s in order:
            if s not in data:
                continue
            mean, lo, hi = data[s]
            col, mk, lab = style[s]
            ax.fill_between(budgets, lo, hi, color=col, alpha=0.13)
            ax.plot(budgets, mean, mk + "-", color=col, lw=2.6, ms=9, label=lab)
        ax.set_title(title, pad=12)
        ax.set_xlabel("Evaluation budget")
        ax.set_ylabel("Estimation error")
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, lw=0.8)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95)
        is_win = "lower" in note
        ax.text(0.04, 0.06, note, transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="#1b6ca8" if is_win else "#444444",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))

    fig.suptitle("Active sampling improves estimation only for interactions and rare events",
                 fontweight="bold", y=1.04)
    path = outdir / "figure_sampler_regimes.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ======================================================================
# FIGURE 2 — specification recipe (PBMC vs HLCA)
# ======================================================================
def figure_specification_recipe(outdir: Path):
    categories = ["Marker", "Cluster size", "n_clusters", "HVG overlap"]

    # median total-order Sobol S_T, [category x step], from run logs.
    # column order = STEP_LABELS
    pbmc = np.array([
        [0.117, 0.927, 0.017, 0.023, 0.016, 0.022],   # Marker
        [0.171, 0.121, 0.163, 0.115, 0.108, 0.583],   # Cluster size
        [0.097, 0.068, 0.089, 0.053, 0.149, 0.725],   # n_clusters
        [0.946, 0.461, 0.000, 0.000, 0.000, 0.000],   # HVG overlap
    ])
    hlca = np.array([
        [0.014, 0.958, 0.008, 0.017, 0.010, 0.010],   # Marker
        [0.112, 0.174, 0.153, 0.118, 0.203, 0.580],   # Cluster size
        [0.052, 0.135, 0.101, 0.052, 0.199, 0.617],   # n_clusters
        [0.665, 0.462, 0.000, 0.000, 0.000, 0.000],   # HVG overlap
    ])

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), constrained_layout=True)
    for ax, mat, name in [(axes[0], pbmc, "PBMC (blood)"),
                          (axes[1], hlca, "HLCA (lung)")]:
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(STEP_LABELS_PRETTY)))
        ax.set_xticklabels(STEP_LABELS_PRETTY, rotation=40, ha="right")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_title(name, pad=12, fontweight="bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=14,
                        color="white" if v < 0.55 else "black",
                        fontweight="bold" if v >= 0.5 else "normal")
        ax.set_xlabel("Analysis step")
        # thin separators
        ax.set_xticks(np.arange(-0.5, len(STEP_LABELS_PRETTY), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(categories), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("Claim category")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Median total-order Sobol index $S_\\mathrm{T}$", fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle("Normalization dominates marker calls; resolution dominates cluster counts \u2014 in both tissues",
                 fontweight="bold", y=1.06)
    path = outdir / "figure_specification_recipe.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ======================================================================
# FIGURE 3 — pairwise interactions (PBMC vs HLCA), marker claims
# ======================================================================
def figure_pairwise_interactions(outdir: Path):
    # Symmetric 6x6 median V_jk matrices for MARKER claims, from run logs.
    # Only the leading pairs carry appreciable mass; the rest are near zero.
    # Indices follow STEP_LABELS order:
    # 0 n_hvg, 1 normalization, 2 n_components, 3 batch_correction,
    # 4 k_neighbors, 5 cluster_res
    def sym(pairs, n=6):
        m = np.zeros((n, n))
        for (i, j), v in pairs.items():
            m[i, j] = v
            m[j, i] = v
        np.fill_diagonal(m, np.nan)
        return m

    # PBMC marker pairwise medians: n_hvg x normalization dominant (0.042),
    # normalization x batch_correction next (0.008), others ~0.005 or below.
    pbmc = sym({
        (0, 1): 0.042,   # n_hvg x normalization (leading)
        (1, 3): 0.008,   # normalization x batch_correction
        (1, 5): 0.006,   # normalization x cluster_res
        (1, 2): 0.005,   # normalization x n_components
        (0, 5): 0.005,   # n_hvg x cluster_res
        (4, 5): 0.005,   # k_neighbors x cluster_res
        (0, 2): 0.004,
        (2, 5): 0.004,
        (3, 5): 0.003,
        (0, 3): 0.003,
    })
    # HLCA marker pairwise medians: all weak, normalization x batch_correction
    # leads (~0.003); n_hvg x batch_correction ~0.003.
    hlca = sym({
        (1, 3): 0.0031,  # normalization x batch_correction (leading)
        (0, 3): 0.0030,  # n_hvg x batch_correction
        (2, 3): 0.0021,  # n_components x batch_correction
        (0, 1): 0.0020,  # n_hvg x normalization
        (1, 5): 0.0018,
        (4, 5): 0.0018,
        (1, 2): 0.0015,
        (0, 5): 0.0014,
        (2, 5): 0.0012,
        (3, 5): 0.0010,
    })

    vmax = 0.05  # shared scale; PBMC leading pair = 0.042
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, mat, name in [(axes[0], pbmc, "PBMC (blood)"),
                          (axes[1], hlca, "HLCA (lung)")]:
        masked = np.ma.masked_invalid(mat)
        cmap = plt.cm.magma.copy()
        cmap.set_bad("#e8e8e8")
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(STEP_LABELS_PRETTY, rotation=40, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels(STEP_LABELS_PRETTY)
        ax.set_title(name, pad=12, fontweight="bold")
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                v = mat[i, j]
                if v >= 0.0005:
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=12,
                            color="white" if v < vmax * 0.55 else "black")
        ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Median pairwise interaction $V_{jk}$ (marker claims)", fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle("HVG\u00d7normalization is the strongest interaction in blood; interactions are uniformly weak in lung",
                 fontweight="bold", y=1.05)
    path = outdir / "figure_pairwise_interactions.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ======================================================================
# FIGURE 4 — published-claim audit (Stephenson 2021, 50k-cell PBMC run)
# ======================================================================
def figure_published_audit(outdir: Path):
    # Adequately-covered genes (>= 500 expressing cells), from the 50k audit.
    # Rows ordered by overall reproducibility (ascending), genes annotated with
    # expressing-cell count. Columns are the five normalizations.
    norm_names = ["log1p_cp10k", "log1p_median", "scran_like", "raw_z", "pearson_residuals"]
    genes = ["MPO", "C1QB", "C1QA", "FCGR3A", "CD38", "PF4"]
    cells = {"MPO": 617, "C1QB": 607, "C1QA": 1006,
             "FCGR3A": 8981, "CD38": 5272, "PF4": 2691}
    overall = {"MPO": 0.08, "C1QB": 0.30, "C1QA": 0.50,
               "FCGR3A": 0.86, "CD38": 0.87, "PF4": 0.93}
    # reproducibility by normalization (fraction of pipelines), from the audit log
    repro = {
        "MPO":    {"log1p_cp10k": 0.19, "log1p_median": 0.08, "scran_like": 0.12, "raw_z": 0.00, "pearson_residuals": 0.00},
        "C1QB":   {"log1p_cp10k": 0.48, "log1p_median": 0.43, "scran_like": 0.53, "raw_z": 0.08, "pearson_residuals": 0.00},
        "C1QA":   {"log1p_cp10k": 0.71, "log1p_median": 0.71, "scran_like": 0.72, "raw_z": 0.36, "pearson_residuals": 0.00},
        "FCGR3A": {"log1p_cp10k": 1.00, "log1p_median": 1.00, "scran_like": 0.98, "raw_z": 0.94, "pearson_residuals": 0.36},
        "CD38":   {"log1p_cp10k": 1.00, "log1p_median": 1.00, "scran_like": 1.00, "raw_z": 1.00, "pearson_residuals": 0.34},
        "PF4":    {"log1p_cp10k": 1.00, "log1p_median": 1.00, "scran_like": 1.00, "raw_z": 1.00, "pearson_residuals": 0.67},
    }
    norm_pretty = ["log1p-CP10k", "log1p-median", "scran", "raw z-score", "Pearson resid."]

    mat = np.array([[repro[g][n] for n in norm_names] for g in genes])
    ylabels = [f"{g}  (n={cells[g]:,})" for g in genes]

    fig, ax = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(norm_pretty)))
    ax.set_xticklabels(norm_pretty, rotation=35, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(ylabels)
    for i in range(len(genes)):
        for j in range(len(norm_names)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=14,
                    color="black" if 0.25 < v < 0.85 else "white",
                    fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(norm_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(genes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Normalization")
    ax.set_ylabel("Published marker (Stephenson et al. 2021)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Fraction of pipelines reproducing the marker call", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle("Reproducibility of published PBMC markers collapses under Pearson residuals",
                 fontweight="bold", y=1.04)
    path = outdir / "figure_published_audit.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="Figures")
    ap.add_argument("--sampler-csv-dir", default="results/sampler",
                    help="Directory with experiment_{A,B,C}_*.csv from the "
                         "active-sampler benchmark. Figure 1 is plotted from "
                         "these per-budget results.")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figure_sampler_regimes(outdir, results_dir=args.sampler_csv_dir)
    figure_specification_recipe(outdir)
    figure_pairwise_interactions(outdir)
    figure_published_audit(outdir)
    print("All figures written to", outdir.resolve())


if __name__ == "__main__":
    main()
