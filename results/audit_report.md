# Audit report: every reported number re-derived from saved run records

This report substantiates the reproducibility claim in the response-to-reviewers.
Every mean and sample standard deviation printed in the manuscript is re-derived
here from the saved per-run outputs and compared against the recorded summaries.

## Scope

- **20 jobs** = 5 datasets (UCI digits, BBCSport, Yale, ORL, Scene-15) × 4 variants
  (`full`, `no_gradient`, `no_l2log`, `tnn`).
- **400 per-run records** = 20 jobs × 20 recorded seeds `20260814`–`20260833`.
- Every reported mean and sample standard deviation (N−1) is recomputed from
  `metrics_all_runs.csv`, the same machine-readable records the manuscript tables
  were built from.

## Recomputation paths

Two independent paths cross-check the recorded summaries:

1. **MATLAB** — `code/summarize_runs.m` (`mean(x)`, `std(x,0)`), the exact routine
   cited in the response, driven by `audit_recompute.m`.
2. **Python 3** — an independent re-implementation (`statistics.mean`,
   `statistics.stdev`), as a cross-language verification.

## Result

| recomputation path | max \|Δ mean\| | max \|Δ std\| |
|---|---|---|
| MATLAB `summarize_runs.m` | **0** | **0** |
| Python 3 (independent) | 4.44×10⁻¹⁶ | 4.44×10⁻¹⁶ |

The direct MATLAB recompute reproduces the recorded summaries **bit-for-bit**
(maximum discrepancy 0), which is stronger than the `3.33×10⁻¹⁶` upper bound quoted
in the response (that figure came from an earlier text round-trip path; the direct
recompute reproduces the summaries exactly). The independent Python path agrees to
machine precision (4.44×10⁻¹⁶), ruling out any circularity from using the same
routine on both sides.

## Rounding-boundary note (2 of 240 values)

Two ACC means fall exactly on a 4th-decimal rounding boundary, so an independent
implementation can display a different last digit while still agreeing to ~10⁻¹⁶:

| dataset / variant | metric | recorded (full) | recorded (4 dp) | manuscript | Python mean (4 dp) |
|---|---|---|---|---|---|
| uci / tnn | acc | 0.95114999999999994 | 0.9511 | 0.9511 | 0.9512 |
| orl / no_gradient | acc | 0.93424999999999991 | 0.9342 | 0.9342 | 0.9343 |

The manuscript prints the correct rounding of the recorded summaries (0.9511 and
0.9342). The underlying MATLAB-vs-Python difference is ~4.44×10⁻¹⁶ (floating-point
accumulation order), not a data error.

## Table 1 — main results (Table 2–6, `full` variant, 6 metrics)

Each cell is `manuscript (4 dp)`; the recomputed value agrees to 0 (MATLAB) and
≤ 4.44×10⁻¹⁶ (Python). Full-precision values are in `audit_comparison.csv`.

| dataset | NMI | AR | ACC | Recall | Precision | F-score |
|---|---|---|---|---|---|---|
| UCI digits | 0.9837±0.0189 | 0.9646±0.0542 | 0.9667±0.0553 | 0.9864±0.0166 | 0.9529±0.0759 | 0.9684±0.0484 |
| BBCSport | 0.9634±0.0426 | 0.9539±0.0742 | 0.9645±0.0615 | 0.9722±0.0384 | 0.9593±0.0728 | 0.9653±0.0556 |
| Yale | 0.8206±0.0155 | 0.6024±0.0369 | 0.7558±0.0363 | 0.7053±0.0304 | 0.5700±0.0470 | 0.6293±0.0337 |
| ORL | 0.9892±0.0061 | 0.9499±0.0280 | 0.9525±0.0284 | 0.9832±0.0106 | 0.9217±0.0423 | 0.9511±0.0273 |
| Scene-15 | 0.8962±0.0211 | 0.8291±0.0567 | 0.8455±0.0598 | 0.8757±0.0347 | 0.8109±0.0696 | 0.8414±0.0523 |

## Table 2 — ablation (Table 8, NMI / ACC)

| dataset | full | no_gradient | no_l2log | tnn |
|---|---|---|---|---|
| UCI digits | 0.9837 / 0.9667 | 0.9841 / 0.9674 | 0.9819 / 0.9497 | 0.9818 / 0.9511 |
| BBCSport | 0.9634 / 0.9645 | 0.9591 / 0.9307 | 0.2191 / 0.4778 | 0.9279 / 0.8915 |
| Yale | 0.8206 / 0.7558 | 0.8256 / 0.7603 | 0.2276 / 0.1812 | 0.8209 / 0.7658 |
| ORL | 0.9892 / 0.9525 | 0.9861 / 0.9342 | 0.6332 / 0.3654 | 0.9871 / 0.9403 |
| Scene-15 | 0.8962 / 0.8455 | 0.9036 / 0.8525 | 0.8825 / 0.8376 | 0.9139 / 0.8628 |

## Reproducibility

- `audit_recompute.m` re-derives all 120 mean/std pairs (20 jobs × 6 metrics) via
  `code/summarize_runs.m` and writes `audit_comparison.csv` (dataset, variant,
  metric, recomputed vs recorded, Δ). Run `matlab -batch "audit_recompute"` to
  reproduce the maximum discrepancy of 0.
- Seeds, configuration, and per-run records live under each
  `results/final_20260815_retuned_v2/<dataset>/<variant>/` directory
  (`metrics_all_runs.csv`, `summary.csv`, `source_hashes.csv`, `provenance.json`).
