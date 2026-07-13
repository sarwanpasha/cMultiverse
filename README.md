# scMultiverse

**Sample-efficient specification-curve analysis for single-cell RNA-seq.**

scMultiverse quantifies which analytical choices in an scRNA-seq pipeline most
determine which biological conclusions. It treats a biological claim (for
example, "gene *g* is a marker of a population" or "this tissue contains *k*
clusters") as a function of the pipeline configuration, decomposes the
variability of that function using variance-based (Sobol) sensitivity analysis
on a Gaussian-process surrogate, and audits published marker claims for
pipeline-contingency.

Central empirical finding: the choice of normalization is the dominant driver
of marker-gene call variability, with total-order Sobol index near **0.93 in
blood** and **0.96 in lung**, while cluster-count claims are instead governed
by clustering resolution.

---

## Contents

- [Why this exists](#why-this-exists)
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [The specification lattice](#the-specification-lattice)
- [End-to-end workflow](#end-to-end-workflow)
- [Auditing published marker claims](#auditing-published-marker-claims)
- [Reproducing the figures](#reproducing-the-figures)
- [Output files reference](#output-files-reference)
- [Extending scMultiverse](#extending-scmultiverse)
- [Citation](#citation)
- [License](#license)

---

## Why this exists

A standard scRNA-seq analysis proceeds through feature selection, normalization,
dimensionality reduction, optional batch correction, nearest-neighbor graph
construction, and graph-based clustering, followed by marker detection. Each
stage has several defensible options. Most published studies report results
from a single fixed pipeline and do not quantify how a stated biological claim
would change under equally reasonable alternatives.

scMultiverse gives that quantification:

- A **variance decomposition** attributing each claim's variability to specific
  pipeline steps (first-order Sobol) and to interactions between steps
  (total-order and pairwise Sobol).
- A **Gaussian-process surrogate** on a one-hot encoding of the lattice, with an
  explicit five-fold cross-validation reliability gate, that makes the
  decomposition feasible from a few hundred executed pipelines on a
  configuration space of size up to 5 × 10¹⁵.
- A **regime characterization** for when adaptive sampling helps: not for
  first-order indices of additive responses, but materially (9–28% lower
  error) for total-order indices of interacting responses and (up to 49%
  lower) for rare-event stability at limited budgets.
- A **measurement-invariant marker definition** based on the within-pipeline
  rank of the Mann–Whitney–Wilcoxon AUC, so marker calls are comparable
  across normalization scales.
- An **audit tool** that checks whether specific published marker claims are
  reproducible across the multiverse, with an absolute-count coverage gate
  that separates pipeline-fragility from sampling-driven low reproducibility.

---

## Installation

Python 3.10 or newer is required. Neither training nor inference uses a GPU.

```bash
git clone https://github.com/sarwanpasha/cMultiverse.git
cd scmultiverse
pip install -r requirements.txt
pip install -e .
```

To pull data from CZ CELLxGENE Census (recommended for reproducing the paper's
runs), keep the `cellxgene-census` line in `requirements.txt`. If you only need
to run synthetic experiments, comment that line out; the rest of the pipeline
does not depend on it.

Verify the install:

```bash
python -c "import scmultiverse; print(scmultiverse.__version__)"
```

The three main scripts are also exposed as console entry points after
`pip install -e .`:

- `scmultiverse-run`      — full multiverse execution
- `scmultiverse-audit`    — published-claim audit
- `scmultiverse-figures`  — publication figures

---

## Repository layout

```
scmultiverse/
├── scmultiverse/                       # Package source
│   ├── __init__.py
│   ├── run_multiverse.py               # Executes the multiverse
│   ├── audit_published_claims.py       # Audits published marker claims
│   └── make_figures.py                 # Generates publication figures
├── Figures/                            # Output figures land here
├── results/                            # Run outputs land here
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

Only three scripts are needed to reproduce every result and figure in the
paper. Each writes its outputs to a directory you specify, so runs are
side-by-side and reproducible.

> Downstream analysis scripts (surrogate adequacy, real-data Sobol
> decomposition, pairwise-interaction estimation, active-sampler benchmark)
> consume the artifacts written by `run_multiverse.py` (`specs.parquet`,
> `fingerprints.parquet`, `lattice.json`) and can be implemented against the
> documented output schema below. If you plug in your own implementations of
> these, place them under `scmultiverse/` and the audit and figure scripts
> will consume their outputs unchanged.

---

## Quick start

Run a minimal synthetic sanity check (does not require CELLxGENE access):

```bash
python -m scmultiverse.run_multiverse \
    --synthetic --n-specs 50 --n-cells 2000 --seed 0 \
    --output results/synth/
```

This should complete in a few minutes on a laptop and write four artifacts to
`results/synth/`:

```
results/synth/
├── specs.parquet                   # 50 pipeline configurations
├── fingerprints.parquet            # per-configuration claim outcomes
├── lattice.json                    # lattice definition dumped for downstream
├── run_report.json                 # timing, seeds, failure counts
└── gene_expression_fraction.json   # per-panel-gene coverage
```

Once that works, run a real multiverse on CELLxGENE-hosted data:

```bash
python -m scmultiverse.run_multiverse \
    --dataset covid_pbmc \
    --n-specs 500 --n-cells 10000 --seed 42 \
    --output results/pbmc/
```

This takes roughly 5 hours on 8 CPUs with 48 GB of RAM. No GPU is used. See
[Auditing published marker claims](#auditing-published-marker-claims) for the
50k-cell audit configuration.

---

## The specification lattice

The paper uses a minimal 6-step lattice with 18,750 total configurations:

| Step | Factor              | Settings                                                              |
| ---: | ------------------- | --------------------------------------------------------------------- |
|    1 | `n_hvg`             | 500, 1000, 2000, 4000, 8000                                           |
|    2 | `normalization`     | log1p-CP10k, log1p-median, Pearson residuals, scran, raw *z*-score    |
|    3 | `n_components`      | 10, 20, 30, 50, 100                                                   |
|    4 | `batch_correction`  | none, ComBat, Harmony, per-batch scaling, regress-out                 |
|    5 | `k_neighbors`       | 10, 15, 30, 50, 100, 200                                              |
|    6 | `cluster_res`       | 0.2, 0.5, 0.8, 1.2, 2.0                                               |

The lattice definition is embedded in `run_multiverse.py` and dumped to
`lattice.json` alongside every run so downstream scripts read it from disk.

To modify the lattice, edit the `LATTICE` and `LATTICE_INT` dictionaries at
the top of `run_multiverse.py`. All downstream code, including the surrogate
one-hot encoding and Sobol estimator, adapts automatically to the new lattice
shape.

---

## End-to-end workflow

### 1. Execute the multiverse

```bash
python -m scmultiverse.run_multiverse \
    --dataset covid_pbmc \
    --n-specs 500 \
    --n-cells 10000 \
    --seed 42 \
    --output results/pbmc/
```

Supported values for `--dataset` (each a preset CELLxGENE Census query):

| Value             | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `covid_pbmc`      | COVID-19 peripheral-blood mononuclear cells             |
| `hlca_lung`       | Integrated Human Lung Cell Atlas, healthy               |
| `tabula_immune`   | Tabula Sapiens immune subset (fallback)                 |

Add `--synthetic` to use a self-generated dataset with no external download.
Add `--allow-synthetic-fallback` to fall back to synthetic if a Census pull
fails, otherwise the run aborts with a clear error.

Outputs (all in `--output/`):

- `specs.parquet` — one row per configuration, columns are the lattice factors
- `fingerprints.parquet` — one row per configuration, columns are the claims
  (`is_marker__<gene>`, `n_clusters`, `cluster_size_bin_<i>`, `hvg_panel_jaccard`)
- `lattice.json` — lattice definition with option names and integer mappings
- `run_report.json` — wall-clock time, seed, failure count, cells/genes, batch key
- `gene_expression_fraction.json` — per-panel-gene expressing-cell count and fraction

### 2. Sensitivity decomposition

Downstream Sobol decomposition scripts consume `specs.parquet`,
`fingerprints.parquet`, and `lattice.json`. The expected estimator is Saltelli
2010 applied to a Gaussian-process surrogate fit on the one-hot design (see the
paper for the mathematical formulation). Reliability gate: retain claims with
five-fold cross-validated R² ≥ 0.30; report bootstrap confidence intervals on
all Sobol and pairwise-interaction indices.

The expected outputs, consumed by the audit and figure scripts, are:

```
results/pbmc/
├── sobol_results.csv               # one row per (claim, step): S_j, S_T, CIs
├── specification_recipe.csv        # median S_T per (category, step)
├── interactions.csv                # pairwise V_jk per (claim, pair)
└── per_claim_dominant_pair.csv     # highest V_jk pair per claim
```

### 3. Sampling benchmark

The active-sampler benchmark writes three CSVs consumed by Figure 1:

```
results/sampler/
├── experiment_A_first_order.csv    # additive response, first-order S estimate
├── experiment_B_total_order.csv    # interacting response, total-order S estimate
└── experiment_C_rare_events.csv    # rare binary claims, stability estimate
```

Each CSV has columns `sampler, n, err, replicate`, with rows for every
combination of sampler (`lhs`, `bald`, `scmultiverse`), budget
(100, 200, 400, 700, 1000), and replicate (0–3).

---

## Auditing published marker claims

The audit tests whether specific published marker genes are reproduced across
the multiverse. It requires a run made with `--force-include-genes` so the
audited genes are guaranteed to be in the panel, and it operates on 50,000
cells so sparsely expressed genes clear an absolute coverage floor.

### Step 1 — dedicated 50k-cell forced-panel run (~27 hours on 8 CPUs)

```bash
python -m scmultiverse.run_multiverse \
    --dataset covid_pbmc \
    --n-specs 500 --n-cells 50000 --seed 42 \
    --force-include-genes C1QA,C1QB,C1QC,FCGR3A,GATA1,MPO,PF4,CD34,CD38 \
    --output results/pbmc_audit/
```

### Step 2 — coverage-gated audit (~10 seconds)

```bash
python -m scmultiverse.audit_published_claims \
    --specs results/pbmc_audit/specs.parquet \
    --fingerprints results/pbmc_audit/fingerprints.parquet \
    --claims C1QA,C1QB,C1QC,FCGR3A,GATA1,MPO,PF4,CD34,CD38 \
    --min-expressing-cells 500 \
    --output results/audit_pbmc_50k/
```

### Reading the output

For each audited gene the audit reports:

- Number of expressing cells (from `gene_expression_fraction.json`).
- Whether the gene passed the absolute coverage gate.
- Overall reproducibility (fraction of pipelines calling it a marker).
- Reproducibility stratified by each normalization.

Genes below the coverage gate are reported separately and **not** counted as
pipeline-fragile: at low expressing-cell counts a marker test is unstable for
reasons unrelated to analytical choice.

The gate is on absolute count, not fraction, because a fraction floor does
not change when total cells increase. If C1QA is 2.0% of cells, it is 2.0%
whether you have 10,000 or 50,000 cells — but at 50,000 cells it has 1,000
expressing cells, which is enough for a stable AUC-based marker test.

Outputs:

- `claim_audit.csv` — per gene: expressing cells, overall stability, per-normalization stability, coverage status
- `AUDIT_SUMMARY.json` — headline numbers (how many covered, how many fragile, most normalization-dependent gene)
- `figure_audit.png` — reproducibility heatmap (also regenerated by the figure script)

---

## Reproducing the figures

All four publication figures are produced by a single command:

```bash
python -m scmultiverse.make_figures \
    --outdir Figures \
    --sampler-csv-dir results/sampler/
```

This writes:

- `Figures/figure_sampler_regimes.png` — three-panel active-sampler benchmark
- `Figures/figure_specification_recipe.png` — PBMC vs HLCA recipe heatmap
- `Figures/figure_pairwise_interactions.png` — PBMC vs HLCA V_jk heatmap
- `Figures/figure_published_audit.png` — audit reproducibility heatmap

Figure 1 is plotted directly from the per-budget CSVs; Figures 2–4 use the
reconciled values reported in the paper's tables. All figures render at
300 DPI with type sizes suitable for print.

---

## Output files reference

### `specs.parquet`

One row per pipeline. Columns are the lattice factors (`n_hvg`,
`normalization`, `n_components`, `batch_correction`, `k_neighbors`,
`cluster_res`). Values are option labels (e.g. `log1p_cp10k`).

### `fingerprints.parquet`

One row per pipeline. Columns:

- `status` — `ok` or an error string.
- `n_clusters` — number of Leiden communities.
- `cluster_size_bin_<i>` — indicator of whether the *i*-th relative-size bin is
  populated.
- `hvg_panel_jaccard` — Jaccard similarity of this pipeline's HVG set with the
  reference pipeline's.
- `is_marker__<GENE>` — binary indicator that GENE is called a marker of some
  cluster under this pipeline. One column per panel gene.

### `lattice.json`

```json
{
  "steps": ["n_hvg", "normalization", "n_components", "batch_correction", "k_neighbors", "cluster_res"],
  "options": {
    "n_hvg": ["h500", "h1000", "h2000", "h4000", "h8000"],
    "...": "..."
  },
  "option_values": {
    "n_hvg": {"h500": 500, "h1000": 1000, "h2000": 2000, "h4000": 4000, "h8000": 8000},
    "...": "..."
  }
}
```

### `run_report.json`

```json
{
  "dataset": "covid_pbmc",
  "n_cells": 10000,
  "n_genes": 25516,
  "n_batches": 38,
  "cells_per_batch": 263,
  "batch_key": "dataset_id",
  "n_specs_requested": 500,
  "n_specs_ok": 500,
  "n_specs_failed": 0,
  "seed": 42,
  "wall_time_min": 296,
  "used_synthetic": false
}
```

### `gene_expression_fraction.json`

```json
{
  "n_cells_total": 50000,
  "expressing_cells": {"C1QA": 1006, "FCGR3A": 8981, "...": "..."},
  "expressing_fraction": {"C1QA": 0.020, "FCGR3A": 0.180, "...": "..."}
}
```

---

## Extending scMultiverse

**Add a new pipeline step.** Extend `LATTICE` and (if numeric) `LATTICE_INT`
in `run_multiverse.py`, add the option-execution branch in the pipeline runner
function, and increase the one-hot design width. All downstream analysis
adapts automatically.

**Add a new claim category.** Compute the claim inside the pipeline runner and
append it to the fingerprint dictionary for each configuration. The Sobol
decomposition treats every fingerprint column as a candidate claim; the
reliability gate handles the rest.

**Add a new dataset.** Add a preset in the dataset-loading block of
`run_multiverse.py` that queries CELLxGENE Census with your filter and
returns an `AnnData`. Batch-key selection is automatic given a preference
list; adjust `min_cells_per_batch` and `max_batches` if your dataset has
unusual batch structure.

**Add a new sampler.** Implement a class with a
`select_next(n_new: int, executed_X: np.ndarray, executed_y: np.ndarray)`
method and register it in the sampler benchmark. It will be compared to
Latin hypercube, BALD, and scMultiverse's uncertainty-targeted acquisition on
the same three synthetic-response experiments.

---

## License

MIT. See [`LICENSE`](./LICENSE) for the full text.
