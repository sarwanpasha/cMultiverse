#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SCMULTIVERSE_ROOT:-$PWD}"
source env/bin/activate
python src/anova.py --name $1 --extra_batch $2
