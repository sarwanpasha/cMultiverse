#!/usr/bin/env python3
"""
audit_published_claims.py — audit of published marker claims.

Takes specific marker-gene claims made by a published scRNA-seq study and
measures, across the pipelines in an scMultiverse run, what fraction reproduce
each claim overall and stratified by normalization. A claim reproduced by
most pipelines is robust; one reproduced by a minority is contingent on the
original authors' particular pipeline choices.

Because normalization is the dominant driver of marker-call variability, the
audit stratifies reproducibility by normalization to expose the mechanism.

An absolute coverage gate is applied: genes expressed in fewer than
--min-expressing-cells cells are reported as insufficient-coverage rather
than pipeline-fragile, because at such low counts a marker test is unstable
for reasons unrelated to analytical choice.

Requires a multiverse run produced with --force-include-genes so that the
audited genes are guaranteed present in the panel:

    python run_multiverse.py --dataset covid_pbmc \\
        --n-specs 500 --n-cells 50000 --seed 42 \\
        --force-include-genes C1QA,C1QB,C1QC,FCGR3A,GATA1,MPO,PF4,CD34,CD38 \\
        --output results/pbmc_audit/

    python audit_published_claims.py \\
        --specs results/pbmc_audit/specs.parquet \\
        --fingerprints results/pbmc_audit/fingerprints.parquet \\
        --claims C1QA,C1QB,C1QC,FCGR3A,GATA1,MPO,PF4,CD34,CD38 \\
        --min-expressing-cells 500 \\
        --output results/audit/
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("audit")


def _read(path):
    p = Path(path)
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_pickle(Path(str(p).replace(".parquet", ".pkl")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--fingerprints", required=True)
    ap.add_argument("--claims", required=True,
                    help="Comma-separated gene symbols to audit (the published claims)")
    ap.add_argument("--output", default="results/audit/")
    ap.add_argument("--stability-floor", type=float, default=0.60,
                    help="Reproducibility below this is flagged as fragile")
    ap.add_argument("--min-expressing-cells", type=int, default=500,
                    help="Genes expressed in fewer than this many cells are reported "
                         "as insufficient-coverage, NOT as pipeline-fragile. The AUC "
                         "marker test needs enough expressing cells in a cluster to be "
                         "stable; this is an absolute count, not a fraction, because "
                         "adding total cells does not change a fraction. Default 500.")
    ap.add_argument("--expression-fraction", default=None,
                    help="Path to gene_expression_fraction.json from the run. If "
                         "omitted, looked for next to the fingerprints file.")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    claims = [g.strip() for g in args.claims.split(",") if g.strip()]

    specs = _read(args.specs)
    fp = _read(args.fingerprints)

    # Load per-gene expression fractions for the coverage gate.
    ef_path = args.expression_fraction
    if ef_path is None:
        cand = Path(args.fingerprints).parent / "gene_expression_fraction.json"
        ef_path = str(cand) if cand.exists() else None
    expr_count = {}
    expr_frac = {}
    if ef_path and Path(ef_path).exists():
        raw = json.loads(Path(ef_path).read_text())
        if isinstance(raw, dict) and "expressing_cells" in raw:
            expr_count = raw["expressing_cells"]
            expr_frac = raw.get("expressing_fraction", {})
            n_cells_total = raw.get("n_cells_total")
            log.info(f"Loaded coverage for {len(expr_count)} genes "
                     f"({n_cells_total} cells) from {ef_path}")
        else:
            # Legacy flat {gene: fraction} format — count unavailable
            expr_frac = raw
            log.info(f"Loaded legacy expression fractions for {len(expr_frac)} genes "
                     f"from {ef_path} (absolute counts unavailable)")
    else:
        log.warning("No gene_expression_fraction.json found; coverage gate disabled. "
                    "All genes will be treated as adequately covered.")

    # Restrict to successfully executed pipelines
    if "status" in fp.columns:
        ok = (fp["status"] == "ok").values
        specs = specs.iloc[ok].reset_index(drop=True)
        fp = fp[ok].reset_index(drop=True)
    n_specs = len(fp)
    log.info(f"Auditing {len(claims)} published claims across {n_specs} pipelines")

    norm_levels = sorted(specs["normalization"].astype(str).unique())

    rows = []
    for gene in claims:
        col = f"is_marker__{gene}"
        if col not in fp.columns:
            log.warning(f"  {gene}: not in panel (was the run done with "
                        f"--force-include-genes?); skipping")
            rows.append({"gene": gene, "in_panel": False, "covered": False,
                         "expr_frac": np.nan, "overall_stability": np.nan})
            continue
        n_exp = expr_count.get(gene, None)
        ef = expr_frac.get(gene, np.nan)
        if expr_count:
            covered = (n_exp is not None) and (n_exp >= args.min_expressing_cells)
        elif expr_frac:
            # Legacy: no absolute count available; fall back to a 5% fraction
            covered = np.isfinite(ef) and ef >= 0.05
        else:
            covered = True  # gate disabled
        marker = fp[col].astype(float).values
        overall = float(marker.mean())
        row = {"gene": gene, "in_panel": True, "covered": bool(covered),
               "expr_cells": int(n_exp) if n_exp is not None else np.nan,
               "expr_frac": float(ef) if np.isfinite(ef) else np.nan,
               "overall_stability": overall,
               "fragile": bool(covered and overall < args.stability_floor)}
        for nl in norm_levels:
            mask = (specs["normalization"].astype(str).values == nl)
            row[f"norm__{nl}"] = float(marker[mask].mean()) if mask.sum() else np.nan
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_csv(out / "claim_audit.csv", index=False)

    # ----- Report -----
    log.info("=" * 70)
    log.info("PUBLISHED CLAIM AUDIT — reproducibility across defensible pipelines")
    log.info("=" * 70)
    present = audit[audit["in_panel"]].copy()
    norm_cols = [c for c in present.columns if c.startswith("norm__")]

    covered = present[present["covered"]].copy()
    uncovered = present[~present["covered"]].copy()

    log.info(f"Coverage gate: >= {args.min_expressing_cells} expressing cells required "
             f"(absolute count, for AUC stability)")
    log.info(f"  {len(covered)} genes adequately covered (audited for fragility)")
    log.info(f"  {len(uncovered)} genes below coverage floor (reported separately, "
             f"NOT counted as fragile)")
    log.info("-" * 70)
    log.info(f"{'gene':12s} {'cells':>7s} {'overall':>8s}   reproducibility by normalization")
    for _, r in covered.iterrows():
        by_norm = "  ".join(
            f"{nl.replace('norm__','')}={r[nl]:.2f}"
            for nl in norm_cols if pd.notna(r[nl])
        )
        flag = "  <-- FRAGILE" if r["fragile"] else ""
        nc = f"{int(r['expr_cells'])}" if pd.notna(r["expr_cells"]) else "NA"
        log.info(f"{r['gene']:12s} {nc:>7s} {r['overall_stability']:8.2f}   {by_norm}{flag}")

    n_fragile = int(covered["fragile"].sum())
    log.info("-" * 70)
    log.info(f"Among {len(covered)} adequately-covered claims, {n_fragile} fall below "
             f"the {args.stability_floor:.0%} reproducibility floor")

    if len(uncovered):
        log.info("-" * 70)
        log.info("INSUFFICIENT COVERAGE (low reproducibility here reflects cell-population "
                 "sampling, not pipeline fragility; not counted above):")
        for _, r in uncovered.sort_values("expr_cells").iterrows():
            nc = f"{int(r['expr_cells'])}" if pd.notna(r["expr_cells"]) else "NA"
            log.info(f"  {r['gene']:12s} expressed in {nc} cells, "
                     f"overall reproducibility {r['overall_stability']:.2f}")

    # Normalization-dependence, computed on covered genes only
    worst = pd.DataFrame()
    if len(covered):
        covered["norm_spread"] = covered[norm_cols].max(axis=1) - covered[norm_cols].min(axis=1)
        worst = covered.sort_values("norm_spread", ascending=False).head(3)
        log.info("-" * 70)
        log.info("Covered claims whose reproducibility most depends on normalization:")
        for _, r in worst.iterrows():
            log.info(f"  {r['gene']:12s} spread = {r['norm_spread']:.2f} "
                     f"(min {r[norm_cols].min():.2f}, max {r[norm_cols].max():.2f})")

    summary = {
        "min_expressing_cells": args.min_expressing_cells,
        "n_claims_in_panel": len(present),
        "n_covered": len(covered),
        "n_insufficient_coverage": len(uncovered),
        "insufficient_coverage_genes": uncovered["gene"].tolist(),
        "n_fragile_among_covered": n_fragile,
        "stability_floor": args.stability_floor,
        "median_overall_stability_covered": float(covered["overall_stability"].median()) if len(covered) else None,
        "max_normalization_spread_covered": float(covered["norm_spread"].max()) if len(covered) and "norm_spread" in covered else None,
        "most_normalization_dependent_covered": str(worst.iloc[0]["gene"]) if len(worst) else None,
        "n_pipelines": n_specs,
    }
    (out / "AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {out / 'claim_audit.csv'} and {out / 'AUDIT_SUMMARY.json'}")

    # Figure: covered genes only, since the others are not a fragility story
    if len(covered):
        plot_audit(covered, norm_cols, out / "figure_audit.png")
    log.info("DONE")


def plot_audit(df, norm_cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genes = df["gene"].tolist()
    norm_names = [c.replace("norm__", "") for c in norm_cols]
    mat = df[norm_cols].values  # genes x normalizations

    fig, ax = plt.subplots(figsize=(1.4 * len(norm_names) + 3, 0.5 * len(genes) + 2),
                           constrained_layout=True)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(norm_names)))
    ax.set_xticklabels(norm_names, rotation=45, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    for i in range(len(genes)):
        for j in range(len(norm_names)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    ax.set_title("Reproducibility of published marker claims\nby normalization (fraction of pipelines)")
    fig.colorbar(im, ax=ax, label="fraction of pipelines reproducing claim")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.info(f"  Saved {path}")


if __name__ == "__main__":
    main()
