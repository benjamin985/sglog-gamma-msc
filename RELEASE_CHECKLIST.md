# Release checklist for the reproducibility package (Comment 2.7(d))

This file tells you exactly what to publish so the `[AUTHOR ACTION REQUIRED]`
item in `response/response_to_reviewers.md` (line 840) can be filled with a real
repository URL and DOI. The package under `server_package/` is already nearly
complete; the gaps are (1) an audit report, (2) a license, and (3) the public
hosting.

## What already exists (publish as-is)

- `code/` — the full solver, including the label-free graph constructor
  `construct_gradient_operator.m` (signature `(X, k)`, no label argument) and the
  two nonconvex proximal operators `prox_gamma_scalar.m`, `prox_l2log_columns.m`.
- `TUNING_PROTOCOL.md` — the pre-registered, label-free selection rule (grid,
  fallbacks, timestamps).
- `EXPERIMENT_PROTOCOL.md` — the 20-seed evaluation and audit protocol.
- `DATASET_PROVENANCE.md` — SHA-256 hashes of every processed `.mat` matrix and
  the exact view order/dimensions.
- `results/final_20260815_retuned_v2/` — `all_summaries.csv` plus one directory
  per dataset (66 CSVs total), including `*_gradient_audit.csv` per dataset.
- `tuning/` — the per-candidate tuning CSVs and `analyze_selection.py`.
- `test_nonconvex_prox_global.m` — the 672-case dense-scan test cited in
  Comment 1.2.

## What to add before publishing

### 1. Audit report (substantiates "20-job / 400-run recompute")

Create `results/audit_report.md` (or `.csv`) with one row per reported mean/std
and three columns:

1. the value printed in the manuscript,
2. the value recomputed from the saved per-run assignments by
   `code/summarize_runs.m`,
3. the absolute discrepancy.

State the maximum discrepancy over all rows (the response already claims
`3.33e-16`; the report must reproduce that number from the saved records). Add a
line listing the 20 jobs (5 datasets × 4 variants = `full`, `no_gradient`,
`no_l2log`, `tnn`) and the 20 recorded seeds `20260814`–`20260833`.

### 2. License

- Code: MIT or BSD-3-Clause.
- Do NOT put a blanket license on the datasets (they are third-party).

### 3. Data availability statement (paste into the manuscript)

Replace the `[AUTHOR ACTION REQUIRED]` placeholder with:

> The source code, configuration files, recorded seeds, source/data SHA-256
> hashes, result-generation scripts, and machine-readable outputs for all
> reported tables and figures are available at `<REPOSITORY URL>`, with a
> permanent archived release at `<DOI>`. The UCI digits data are from the UCI
> Machine Learning Repository; BBCSport is derived from the BBC corpus of Greene
> and Cunningham (BBC copyright, acquisition/preparation instructions provided
> rather than a redistributed copy); Yale and ORL are from the official face
> database releases; Scene-15 is from the originating scene-recognition work,
> with the processed multi-view descriptor configuration following the cited
> benchmark. Per-dataset SHA-256 checksums are recorded in
> `DATASET_PROVENANCE.md`.

## Step-by-step

1. Delete non-essential logs and `tuning_old_20260814/` (keep `tuning/` CSVs).
2. Add `LICENSE` and `results/audit_report.md`.
3. `git init`, commit everything, push to a public GitHub repository.
4. Create a Zenodo (or OSF/figshare) archive linked to the repo → obtain a DOI.
5. Fill `<REPOSITORY URL>` and `<DOI>` in:
   - `response/response_to_reviewers.md` line 840, and
   - the manuscript Data and Code Availability Statement.

## BBC copyright caveat

Do not host `bbcsport_2view.mat` if redistribution is uncertain. Provide the
preparation script + checksum instead; the statement above already says this.
