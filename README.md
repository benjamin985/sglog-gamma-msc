# Server rerun package for mathematics-4503790

Before running, read `DATASET_PROVENANCE.md` and verify that the five local
dataset hashes match. Same-named benchmark packages with different view subsets
must not be substituted silently.

This package reruns the revised SGLog-gamma*-MSC implementation without using
class labels during representation learning or graph construction. Labels are
passed only to post-construction leakage diagnostics and final evaluation; they
are never passed to the graph constructor or optimization solver.

## Repository layout

- `code/` — the solver, proximal operators, and label-free graph constructor
  (the reportable implementation).
- `run_revision_experiments.m` — the reportable 20-job runner.
- `TUNING_PROTOCOL.md`, `EXPERIMENT_PROTOCOL.md`, `DATASET_PROVENANCE.md` — the
  pre-registered label-free selection, evaluation/audit, and data-provenance
  protocols.
- `results/final_20260815_retuned_v2/` — the machine-readable outputs behind the
  reported tables and figures, plus `results/audit_report.md`.
- `diag_*.m`, `*_diagnostic.m`, `test_*.m`, `benchmark_*.m`, `timing_probe.m`,
  `smoke_test.m` — development diagnostics and verification tests. They do not
  produce reportable numbers; reportable outputs come only from
  `run_revision_experiments.m` reading `code/`.

## Required runtime

- MATLAB R2021b or later with the Statistics and Machine Learning Toolbox, or
  GNU Octave 6.4+ with the `control` and `statistics` packages. The exact
  runtime is recorded and must be reported truthfully in the manuscript.

## Main command

Copy the complete `server_package` directory to the new server. From MATLAB,
replace the example path below with the actual location on that server:

```matlab
cd('/path/to/server_package')
run_revision_experiments
```

The runner resumes by default: a dataset/variant job with `COMPLETED.ok` is
skipped and retained in `all_summaries.csv`. Run `smoke_test` before starting
the full jobs. Set `SGLOG_RESUME=0` only for a deliberate clean rerun.

On a Linux server, `run_server_job.sh` is the preferred wrapper. It enables
`pipefail`, so an Octave failure cannot be hidden by the `tee` process used to
write the log.

For the controlled BBCSport parameter-sensitivity grid requested by Reviewer
2, run `run_bbcsport_sensitivity`. It writes raw common-grid values to
`results/bbcsport_sensitivity/bbcsport_sensitivity_grid.csv` and checkpoints
every grid point.

Optional environment variables:

- `SGLOG_DATASETS`: comma-separated subset of `uci,bbc,yale,orl,scene`
- `SGLOG_VARIANTS`: comma-separated subset of `full,no_gradient,no_l2log,tnn`
- `SGLOG_RUNS`: number of recorded k-means++ initializations of the fixed
  spectral embedding (default: 20)
- `SGLOG_RUN_TAG`: optional output subdirectory (for example, `pilot`)
- `SGLOG_RESUME`: `1` by default; set `0` to rerun completed jobs
- `SGLOG_SMOKE_SAMPLES`: stratified sample count for a non-reportable smoke run
- `SGLOG_MAX_ITER`: optional iteration override for smoke testing only

Example:

```bash
SGLOG_RUN_TAG=final SGLOG_DATASETS=yale,orl \
SGLOG_VARIANTS=full,no_gradient SGLOG_RUNS=20 ./run_server_job.sh
```

## Outputs

Each dataset/variant directory under `results/` contains:

- `metrics_all_runs.csv`: per-run NMI, ARI, ACC, recall, precision, F-score,
  predicted cluster count, minimum cluster size, validity flag, and seed.
- `summary.csv`: mean and sample standard deviation for all six metrics, plus
  iteration count, representation-learning wall time, final residual/change,
  and stopping reason.
- `diagnostics.mat`: final/normalized affinity matrices, spectral embedding,
  eigenvalues, labels, cluster sizes, per-run confusion matrices, metric-
  recomputation audit, objective and residual histories, iteration count,
  representation-learning wall time, ground truth used only for evaluation,
  and the complete configuration.
- `source_hashes.csv` and `provenance.json`: SHA-256 hashes of the exact
  numerical source and dataset used, parameters, recorded seeds, runtime version,
  computer identifier, and generation timestamp. These files distinguish the
  revised rerun from the legacy scripts that produced the disputed submission
  values.

For a miniature end-to-end Octave check:

```bash
SGLOG_DATASETS=yale SGLOG_VARIANTS=full SGLOG_RUNS=3 \
SGLOG_SMOKE_SAMPLES=30 SGLOG_MAX_ITER=2 SGLOG_RUN_TAG=smoke \
octave --quiet --eval "run_revision_experiments"
```

Never report a `*_smoke` result in the manuscript.

`results/all_summaries.csv` combines all completed jobs.

## Material algorithm corrections

1. The scalar gamma* proximal update evaluates the objective at the boundary
   point zero and every positive real stationary root, then selects the global
   minimizer.
2. The l2,log proximal update acts jointly on each sample column after stacking
   all views, matching the manuscript definition.
3. The sparse-gradient operator is a weighted incidence matrix constructed
   once from normalized feature vectors. It never accesses ground-truth labels.
4. The three ADMM penalties retain separate symbols because they scale distinct
   constraints, but use a common initialization and continuation schedule.
5. Reported deviations are sample standard deviations over recorded k-means++
   initializations of one fixed spectral embedding, never zero-filled
   placeholders or solver-level uncertainty estimates.
6. A job fails rather than being summarized if it has a nonfinite metric,
   an empty/missing cluster, a degenerate affinity, or a metric table that
   cannot be regenerated from the saved assignments.
7. For $N\ge1000$, the two symmetric positive-semidefinite Sylvester
   coefficient matrices are diagonalized once and their eigenspaces are
   reused. This is an exact reformulation of the same matrix equation, not an
   iterative approximation. `test_cached_sylvester.m` verifies agreement with
   the reference solver in MATLAB and Octave; the two cache/solve files are
   included in each job's source-hash register.

## Figures from completed outputs

After all five datasets have completed, generate the reviewer-audit ablation
and BBCSport sensitivity figures without retraining:

```matlab
plot_revision_results(fullfile(pwd,'results','final_20260815_retuned_v2'), ...
    fullfile(pwd,'revision_figures'), false)
```

With the third argument set to `false`, the script refuses to make a final
ablation figure if any dataset/variant lacks both `summary.csv` and
`COMPLETED.ok`. It reads saved outputs only and fixes the displayed variant
order as Full, No-gradient, No-l2log, and TNN. For each completed full-model
job it also plots the ground-truth-sorted affinity, a confusion matrix from the
run whose NMI is closest to the 20-run mean (not the best run), and cluster
sizes for all 20 seeds.
