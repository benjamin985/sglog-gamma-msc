#!/usr/bin/env bash
# Launch the 20 locked dataset--variant jobs (5 datasets x 4 variants) with the
# eigengap-selected configuration, one Octave process per job. Threading is
# capped so 20 concurrent jobs (x 10 threads) stay within the 208-core machine.
set -euo pipefail
cd "$(dirname "$0")"

RUN_TAG="${1:-final_20260814}"
DATASETS=(uci bbc yale orl scene)
VARIANTS=(full no_gradient no_l2log tnn)

# Threads per job: 20 jobs x 10 = 200 < 208 cores.
export OPENBLAS_NUM_THREADS=10
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
# Sapphire Rapids misdetected as Cooperlake; Cooperlake eig/svd is broken.
export OPENBLAS_CORETYPE=SKYLAKEX

mkdir -p "results/${RUN_TAG}/logs"

for ds in "${DATASETS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    SGLOG_DATASETS="$ds" \
    SGLOG_VARIANTS="$v" \
    SGLOG_RUNS=20 \
    SGLOG_RUN_TAG="$RUN_TAG" \
    octave --quiet --eval "run_revision_experiments" \
      > "results/${RUN_TAG}/logs/${ds}_${v}.log" 2>&1 &
  done
done

echo "Launched 20 jobs into results/${RUN_TAG}. Waiting..."
wait
echo "All 20 final jobs finished."
