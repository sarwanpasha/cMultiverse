#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -eo pipefail
cd "${SCMULTIVERSE_ROOT:-$PWD}"
source env/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python src/cond_claims.py --name $1 --task $SLURM_ARRAY_TASK_ID "${@:2}"
