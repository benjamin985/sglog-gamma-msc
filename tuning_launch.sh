#!/bin/bash
cd "$(dirname "$0")"
N=360
P="${1:-80}"
mkdir -p tuning
if [ -f tuning/LAUNCHING ]; then echo "already launching"; exit 0; fi
touch tuning/LAUNCHING
trap 'rm -f tuning/LAUNCHING' EXIT
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Sapphire Rapids is misdetected as Cooperlake by OpenBLAS 0.3.20; the
# Cooperlake dense eig/svd kernel is numerically broken (n>=200). Force a
# correct AVX-512 kernel.
export OPENBLAS_CORETYPE=SKYLAKEX
echo "Launching $N tuning candidates with parallelism $P (single-threaded BLAS)"
seq 1 "$N" | xargs -P "$P" -I{} env SGLOG_TUNING_IDX={} octave-cli --no-gui --quiet --eval "tuning_run" 2>> tuning/all_errors.log
echo "All tuning candidates finished."
