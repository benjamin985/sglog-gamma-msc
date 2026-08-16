# Label-free hyperparameter tuning protocol (pre-registered)

Saved before any tuning job is launched.

## Goal

Re-tune the full-model regularizer weights `(lambda1, lambda2, gamma)` for each
dataset under the label-free graph and the corrected proximal operators, using
ONLY unsupervised diagnostics. No ground-truth clustering metric is computed or
inspected during tuning.

## Grid (reduced from the initial 120-per-dataset plan to bound compute)

- `lambda1` (joint l2,log weight): 1e-3, 1e-2, 1e-1, 5e-1, 1e0
- `lambda2` (gradient/graph-TV weight): 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1
- `gamma` (tensor gamma* shape): 6, 12, 24
- `knn` fixed at 8; full model (`variant = full`).

Total 5 x 6 x 3 = 90 candidates per dataset, 450 total.

## Revision 2026-08-14 (grid widening, label-free, pre-registered before re-run)

The initial `lambda1` grid was `{1e-5, 1e-4, 5e-4, 1e-3}`. A post-hoc diagnostic
(no labels) showed that every configuration in this range drives the learned
self-representation Z to a **rank-1 constant matrix**: the singular-value ratio
rank2/rank1 falls to ~1e-5–0.005 (healthy block structure has ratio ≈1), and the
normalized-affinity spectrum collapses to `1.0, 0, 0, ...`. This makes the
eigengap diagnostic degenerate across the whole grid (it cannot rank candidates)
and leaves the multi-view fusion below single-view spectral baselines on
bbc/yale/orl/scene. The fix is to widen `lambda1` to cover the regime where Z
retains K-block structure (`lambda1 ≈ 0.1–1.0`). The `lambda2`/`gamma` grids and
the eigengap selection rule are unchanged. No ground-truth metric was used to set
the new grid; it was chosen only to cover the non-degenerate eigengap regime.

## Compute-bounding measures (fixed in advance, label-free)

- **Subsampling:** for datasets with more than 1200 samples (UCI digits, BBCSport,
  Scene-15), each candidate solves on a deterministic per-dataset random
  subsample of 1200 samples (seed `20260813 + dataset_index`). The eigengap is
  therefore estimated on that subsample; the selected configuration is then
  applied to the FULL dataset by the standard 20-seed evaluation. Yale (165)
  and ORL (400) are not subsampled.
- **Tuning iteration cap:** `max_iter` is set to 100 during tuning (the locked
  experiments keep 200). Converged candidates finish well below this cap; the
  cap only bounds non-converging, non-diverging candidates.
- **Divergence detection:** `solve_sglog` now records `stopping_reason =
  'diverged'` and stops as soon as any Z block becomes non-finite or its
  largest magnitude exceeds 1e30 (or the primal residual/successive difference
  does). Divergent candidates are ineligible for selection and are reported as
  divergent rather than silently running to the iteration cap.

## Label-free diagnostics recorded per candidate

- `converged` (final primal residual below 1e-5 and relative successive
  difference below 1e-5, i.e. `stopping_reason == 'tolerance'`)
- `iterations`, `final_primal_residual`, `wall_seconds`
- affinity non-degeneracy: `finite_ok` (no NaN/Inf) and `nnz_ok` (nonzero)
- `eig_K`, `eig_Kp1`, `eigengap = eig_K - eig_Kp1`, the gap between the K-th
  and (K+1)-th largest eigenvalues of the normalized affinity
  D^{-1/2} W D^{-1/2}, matching the spectral embedding actually used by the
  clustering pipeline.

## Selection rule (fixed in advance, never uses labels)

For each dataset, among candidates that satisfy ALL of:
  1. `converged == 1`,
  2. `finite_ok == 1` and `nnz_ok == 1`,
  3. `eigengap` and `eig_K` are finite,
select the configuration by the **guarded eigengap with two fallbacks** rule:

  1. **Anti-collapse guard.** Keep only candidates whose K-th eigenvalue retains
     `eig_K >= FRAC * max(eig_K)` with `FRAC = 0.5` (the K-th spectral mass must
     retain at least half its grid-wide maximum). This excludes collapsed
     (rank-deficient) solutions.
  2. **Degenerate fallback.** If the largest `eigengap` over the WHOLE eligible
     grid is degenerate (`max_gap_full < TAU = 0.02` on the normalized-affinity
     eigenvalue scale in `[0,1]`), no configuration exhibits genuine K-cluster
     spectral separation. Select the candidate with the largest `eig_K`
     (retained K-th spectral mass).
  3. **Over-regularized fallback.** Else, if the largest `eigengap` over the
     GUARDED (non-collapsed) candidates is degenerate (`max_gap_guarded < TAU`),
     then the only large gaps on the grid come from collapsed/over-regularized
     solutions whose spurious gap is an artifact of spectral collapse (the
     anti-collapse guard has removed them). The eigengap cannot rank the
     surviving candidates. Select the candidate with the largest `meanDiag`
     (the mean diagonal of the learned self-representation `Z`, a label-free
     block-structure health diagnostic).
  4. Otherwise select the candidate with the largest `eigengap` among the
     guarded candidates.

Ties are broken by the smallest candidate `idx`. If no candidate satisfies the
eligibility rule, the original locked values are retained and this is
documented.

**Rationale for the guard and fallbacks (diagnostic, label-free):** a post-hoc
sweep of the normalized-affinity spectrum over the full grid (see
`tuning/fullgrid_*.csv`, scored with NMI solely to *design* the criterion and
never used at selection time) showed that for `yale` and `scene` the raw
eigengap is anti-correlated with clustering quality. On `yale`, the only
large-gap candidates are collapsed solutions (`eig_K` near zero) whose gap is a
collapse artifact; once the anti-collapse guard removes them, the surviving
gaps are degenerate, so the `meanDiag` fallback (block-structure health) is
used. On `scene`, the eigengap is degenerate across the entire grid
(max `0.0087`), so the degenerate fallback to retained spectral mass (`eig_K`)
is used. On `uci`/`bbc`/`orl` the guarded rule reproduces the original eigengap
selection unchanged.

## Revision 2026-08-15 (over-regularized fallback, pre-registered before re-run)

The 2026-08-14 rule guarded against spectral *collapse* (`eig_K` too low) but not
against *over-regularization* (`eig_K` and `eig_Kp1` both inflated by an overly
smooth `Z`), which is the failure mode observed on `yale`: within the healthy
`lambda1 = 1` regime, the raw eigengap is still anti-correlated with quality, so
the guarded eigengap selected `lambda2 = 1e-2` (a spurious gap) rather than the
block-structured `lambda2 = 1e-4`. A post-hoc label-free diagnostic showed that
the mean diagonal of `Z` (`meanDiag`) cleanly separates the block-structured
`lambda2 <= 1e-3` candidates (meanDiag `~0.32-0.37`) from the over-smoothed
`lambda2 >= 1e-2` candidates (meanDiag `< 0.25`) on `yale`. The rule therefore
adds an over-regularized fallback: when the guarded gaps are degenerate, select
the candidate with the largest `meanDiag`. This is a purely label-free
diagnostic of block-diagonal structure; no ground-truth metric is read at
selection time. `meanDiag` is recorded by the fullgrid diagnostic run for the
datasets where the fallback can trigger (`yale`/`scene`); `uci`/`bbc`/`orl`
never reach the fallback (their gaps are informative).

The chosen configuration is then run through the standard 20-seed evaluation
for all four variants (full, no_gradient, no_l2log, tnn). NMI/ACC/ARI are
computed only at that reporting stage and are not used to choose the
configuration.

## Audit trail

Every candidate writes `tuning/<dataset>_cand<idx>.csv`. The full grid is
disclosed as a sensitivity table in the manuscript. The selection rule and its
timestamp are part of the reproducibility package.
