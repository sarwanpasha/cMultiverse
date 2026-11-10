#!/bin/bash
#SBATCH --job-name=multiverse_lattice
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# usage: sbatch --array=0-24%20 jobs/lattice.sbatch <dataset_name> [--quick]
set -eo pipefail
cd "${SCMULTIVERSE_ROOT:-$PWD}"
source env/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python src/run_lattice.py --name $1 --task $SLURM_ARRAY_TASK_ID --jobs $SLURM_CPUS_PER_TASK $2
