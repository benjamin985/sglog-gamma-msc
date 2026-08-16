#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p "results/${SGLOG_RUN_TAG:?SGLOG_RUN_TAG is required}/logs"

: "${OPENBLAS_NUM_THREADS:=32}"
: "${OMP_NUM_THREADS:=32}"
: "${SGLOG_DATASETS:?SGLOG_DATASETS is required}"
: "${SGLOG_VARIANTS:?SGLOG_VARIANTS is required}"
: "${SGLOG_RUNS:=20}"
: "${SGLOG_JOB_NAME:=${SGLOG_DATASETS}_${SGLOG_VARIANTS//,/_}}"

export OPENBLAS_NUM_THREADS OMP_NUM_THREADS SGLOG_DATASETS SGLOG_VARIANTS
export SGLOG_RUNS SGLOG_RUN_TAG
# Sapphire Rapids misdetected as Cooperlake; Cooperlake eig/svd is broken.
export OPENBLAS_CORETYPE=SKYLAKEX

octave --quiet --eval "run_revision_experiments" 2>&1 \
  | tee "results/${SGLOG_RUN_TAG}/logs/${SGLOG_JOB_NAME}.log"
