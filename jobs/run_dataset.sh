#!/bin/bash
# usage: bash jobs/run_dataset.sh <name> <prep_script.py> <prep args...>
cd "${SCMULTIVERSE_ROOT:-$PWD}"
N=$1; shift
p=$(sbatch --parsable --job-name=multiverse_prep_$N jobs/prep_one.sbatch "$@")
v=$(sbatch --parsable --dependency=afterok:$p --job-name=multiverse_values_$N jobs/values.sbatch $N)
l=$(sbatch --parsable --dependency=afterok:$p --array=0-24%20 --job-name=multiverse_lat_$N jobs/lattice.sbatch $N)
c=$(sbatch --parsable --dependency=afterok:$l:$v --array=0-24%20 --job-name=multiverse_cl_$N jobs/claims.sbatch $N)
a=$(sbatch --parsable --dependency=afterok:$c --job-name=multiverse_an_$N jobs/analyze.sbatch $N)
echo "CHAIN $N prep $p values $v lattice $l claims $c analyze $a"
