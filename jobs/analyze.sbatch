#!/bin/bash
#SBATCH --job-name=multiverse_analyze
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# usage: sbatch jobs/analyze.sbatch <dataset_name>
set -eo pipefail
cd "${SCMULTIVERSE_ROOT:-$PWD}"
source env/bin/activate
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python src/theory_check.py --name $1
python src/anova.py --name $1
python src/ground_truth.py --name $1
python src/design_bench.py --name $1 --reps 10 --fracs 0.01,0.02,0.05,0.1,0.2,0.4
echo ANALYZE_OK
