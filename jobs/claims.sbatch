#!/bin/bash
#SBATCH --job-name=multiverse_claims
#SBATCH --cpus-per-task=4
#SBATCH --mem=14000
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# usage: sbatch --array=0-24%20 jobs/claims.sbatch <dataset_name> [lattice_dir]
set -eo pipefail
cd "${SCMULTIVERSE_ROOT:-$PWD}"
source env/bin/activate
T=$SLURM_CPUS_PER_TASK; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
python src/claims.py --name $1 --task $SLURM_ARRAY_TASK_ID --lattice_dir ${2:-lattice}
