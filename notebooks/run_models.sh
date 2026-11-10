#!/bin/bash
# Runs the DELPHI models of Fig. 7 (Mauerhofer+25) on the 600 merger trees.
# Usage (from delphi25_valentin):   ./run_models.sh eIMF fiducial eSFE
# Each model is run with the 5 random seeds below (the paper shows the mean of 5 runs).
# The parameter templates are fig7/params/<model>.dat. Outputs go to Results_600/<model>_seed<seed>/.
# The per-halo debug prints of delphi.f90 are removed with grep; the rest of the run output goes to run.log.

cd "$(dirname "$0")"        # work from delphi25_valentin, wherever the script is started from

seeds=(32889 12345 24680 13579 98765)
ncpu=8

for model in "$@"; do
  for seed in "${seeds[@]}"; do
    outdir=Results_600/${model}_seed${seed}
    mkdir -p "$outdir"
    sed -e "s|OUTDIR|$outdir|" -e "s|SEED|$seed|" fig7/params/${model}.dat > "$outdir/params.dat"
    echo "$(date '+%H:%M:%S')  running $model, seed $seed -> $outdir"
    mpirun -np $ncpu ./delphi "$outdir/params.dat" | grep -v -e "redshift int" -e "log halo mass" -e "dust mass" > "$outdir/run.log"
  done
done
