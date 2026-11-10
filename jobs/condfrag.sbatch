#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SCMULTIVERSE_ROOT:-$PWD}"
source env/bin/activate
export OMP_NUM_THREADS=8
python src/cond_fragility.py --names pbmc_covid,pbmc_covid40 "$@"
