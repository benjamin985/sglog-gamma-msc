# Locked experiment protocol

The workflow is restartable at the dataset/variant level. A job is complete
only after its metrics, summary, compact diagnostics, provenance, and
`COMPLETED.ok` marker have all been written. Dense `Z`, `E`, and `G` blocks are
not duplicated in every diagnostic file; the final affinity, normalized
affinity, embedding, assignments, confusion matrices, cluster sizes, residual
histories, and configuration are retained for the review audit.

## Research question

Does each proposed regularizer improve clustering under a label-free graph and
an identical evaluation protocol, and are the reported gains stable over
independent clustering initializations?

## Inputs and outputs

- Inputs: the five supplied multi-view `.mat` datasets; normalized feature
  columns; mutual k-nearest-neighbor graphs constructed from features only.
- Outputs: all per-seed metrics, mean and sample standard deviation, solver
  residual histories, cluster assignments and sizes, affinity matrices,
  normalized affinity, spectral embedding, eigenvalues, confusion matrices,
  parameters, iteration counts, wall-clock time, and graph diagnostics.
- Provenance: each dataset/variant job records SHA-256 hashes of the exact code
  and dataset, the full parameter structure, all seeds, numerical-runtime version, computer
  identifier, and timestamp. Results without this provenance record are not
  eligible for the revised manuscript.

## Fixed evaluation

- Metrics: NMI, ARI, ACC, recall, precision, and F-score.
- Final seeds: `20260814` through `20260833`, one k-means++ start per seed.
- No failed or unfavorable seed is removed. A failed job is reported as failed
  and investigated rather than silently replaced.
- The spectral eigensolver initialization is fixed at seed `20260813`; reported
  variability comes from the 20 explicitly listed k-means initializations.
- Each completed job must reproduce its metric table from the saved assignments
  with maximum absolute difference at most `1e-12`. Any empty/missing cluster,
  nonfinite/out-of-range metric, or degenerate affinity makes the job fail.

## Variants

- `full`: sparse gradient + joint l2,log + tensor gamma*.
- `no_gradient`: lambda2 is zero.
- `no_l2log`: lambda1 is zero.
- `tnn`: gamma* is replaced by the standard tensor nuclear norm.

## Tuning rule

The original parameter triplets are treated as prespecified candidates. If a
solver failure or clearly degenerate affinity occurs, a coarse grid may be run
and logged in a separate `tuning` directory, but it may be used only to diagnose
label-free feasibility (residual reduction, finite iterates, and a
nondegenerate affinity matrix). Ground-truth clustering metrics are not used to
select the main configuration. If a locked value must be changed, the reason is
recorded before inspection of the final scores and all four variants are rerun
under that configuration in a clean `final` directory with the fixed 20 seeds.
All tuning artifacts are retained.

## Pilot and stop conditions

The BBCSport/full job is the pilot. Full experiments begin only if it runs end
to end, saves every required artifact, returns exactly five nonempty clusters
for every evaluation seed, and has finite objectives and residuals. Work stops
for redesign if data are missing, the metric implementation fails a sanity
check, or the representation is degenerate or does not approach feasibility.

For every final job, the maximum of the reconstruction, graph-gradient split,
and tensor-consensus primal residuals must be below `1e-5`. The maximum relative
successive difference of the view-wise `Z` blocks must independently be below
`1e-5`. A job otherwise stops at 200 iterations and records
`maximum_iterations` rather than being labelled converged.
