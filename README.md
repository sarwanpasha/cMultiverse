# scMultiverse

Exhaustive pipeline multiverses for single-cell RNA-seq: run every combination of a set of analysis choices, record the conclusions each pipeline reaches, decompose how each choice drives those conclusions, and certify whether a given conclusion is robust.

## Requirements

Python 3.10 or newer on Linux. Everything runs on CPUs. The batch scripts in `jobs/` target a SLURM cluster; each step can also be run directly with `python`.

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
# optional, only for the scVI batch-correction option
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install scvi-tools
```

Set the working directory (inputs go to `data/`, outputs to `results/`):

```bash
export SCMULTIVERSE_ROOT=$PWD
```

## Pipeline lattice

Each pipeline is one choice per step (defined in `src/common.py`):

| Step | Options |
|---|---|
| Highly variable genes | 500, 1000, 2000, 4000, 8000 |
| Normalization | log1p-CP10k, log1p-median, acosh VST, log1p with poscounts size factors, analytic Pearson residuals |
| Principal components | 10, 20, 30, 50, 100 |
| Batch / covariate correction | none, ComBat, Harmony, per-batch scaling, covariate regression (optional: scVI) |
| Nearest neighbors | 10, 15, 30, 50, 100, 200 |
| Leiden resolution | 0.2, 0.5, 0.8, 1.2, 2.0 |

This gives 18,750 pipelines per dataset, each clustered with two Leiden seeds.

## Workflow

| Step | Script | What it does |
|---|---|---|
| 1. Data | `prep_census.py`, `prep_cite.py`, `prep_hao.py`, `simulate.py` | Pull one CELLxGENE dataset by dataset id with donor-balanced subsampling, load CITE-seq data with protein measurements, or simulate data with planted markers |
| 2. Values | `compute_values.py` | Per-normalization values and within-gene ranks for a gene panel |
| 3. Lattice | `run_lattice.py` | All pipelines sharing one (HVG, normalization) prefix; steps are cached and (PCs, k) units run in parallel |
| 4. Claims | `claims.py` | Cluster count, ARI/NMI, population recovery and matched-population marker AUCs under every value normalization |
| 5. Decomposition | `anova.py` | Exact first-order, total-order and pairwise Sobol indices on the full factorial, direct/mediated split |
| 6. Theory checks | `theory_check.py`, `theory_general.py` | Zero-tie and pair-type decompositions of normalization effects on rank-based marker statistics |
| 7. Ground truth | `ground_truth.py`, `stat_bench.py` | Accuracy against planted, protein-confirmed or curated markers |
| 8. Designs | `design_bench.py` | Cost-aware sampling designs and estimators against the exhaustive indices |
| 9. Fragility | `claims_universe.py`, `fragility_model.py` | Predict a claim's multiverse support from one analysis |
| 10. Disease claims | `cond_claims.py`, `cond_anova.py`, `boot_ref.py`, `cond_fragility.py` | Case-control composition and pseudobulk differential expression claims, donor bootstrap, prediction |
| 11. Certification | `cert_bench.py`, `summarize_cert.py` | Sequential, anytime-valid certification of claim robustness |

`jobs/run_dataset.sh` chains steps 1 to 5 for one dataset:

```bash
bash jobs/run_dataset.sh pbmc_s0 prep_census.py --name pbmc_s0 --which pbmc --seed 0
```

Disease claims for a case-control dataset:

```bash
sbatch --array=0-24 jobs/cond.sbatch <name>
sbatch jobs/condanova.sbatch <name>
sbatch jobs/boot.sbatch <name>
```

## Outputs

All outputs are written under `results/<dataset>/`: `lattice/` (partitions), `claims/` (claims per pipeline), `anova/` (Sobol indices), `theory/`, `ground_truth/`, `fragility/`, `cond*/` and `results/certification/`.

## License

MIT
