#!/usr/bin/env python3
"""Label-free hyperparameter selection from tuning CSVs (pre-registered rule v3).

For each dataset, among candidates with converged==1, finite_ok==1, nnz_ok==1,
and finite eig_K/eig_Kp1/eigengap, choose the configuration by the rule below.
No ground-truth metric is used at selection time.

Selection rule (fixed in advance, never uses labels):

  1. Guard against spectral collapse: keep only candidates whose K-th
     eigenvalue retains eig_K >= FRAC * max(eig_K) over the grid (FRAC = 0.5).
  2. If the largest eigengap over the WHOLE eligible grid is degenerate
     (max_gap_full < TAU), the eigengap carries no genuine K-cluster spectral
     separation anywhere on the grid. Select the candidate with the largest
     eig_K (retained K-th spectral mass).
  3. Else, if the largest eigengap among the GUARDED (non-collapsed) candidates
     is degenerate (max_gap_guarded < TAU), then the only large gaps on the grid
     come from collapsed/over-regularized solutions (a documented failure mode
     in which over-regularization inflates the gap while collapsing the
     spectrum). Select the candidate with the largest meanDiag (the mean diagonal
     of the learned self-representation Z, a label-free block-structure health
     diagnostic).
  4. Otherwise select the candidate with the largest eigengap among the guarded
     candidates.

Ties are broken by the smallest candidate idx.
Writes tuning/selected_configs.csv and prints an eligibility summary.
"""
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TUNING = os.path.join(ROOT, 'tuning')

DATASETS = ['uci', 'bbc', 'yale', 'orl', 'scene']

# Degenerate eigengap threshold: below this (on the normalized-affinity
# eigenvalue scale in [0,1]) the eigengap cannot rank candidates.
TAU = 0.02
# Anti-collapse guard: a candidate's eig_K must retain at least this fraction
# of the grid-wide maximum eig_K.
FRAC = 0.5


def _f(x):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return None
    if v != v or abs(v) == float('inf'):
        return None
    return v


def _load_cands():
    """Return list of candidate dicts from *_cand*.csv (label-free fields)."""
    rows = []
    for f in glob.glob(os.path.join(TUNING, '*_cand*.csv')):
        with open(f, newline='') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                r['_file'] = os.path.basename(f)
                rows.append(r)
    return rows


def _load_meandiag():
    """Return {(dataset, lambda1, lambda2, gamma): meanDiag} from fullgrid CSVs.

    meanDiag is a label-free diagnostic recorded by the fullgrid diagnostic run
    (diag_fullgrid_yale_scene.m). It is read here for the over-regularized
    fallback only; the NMI column present in those files is never read.
    """
    out = {}
    for f in glob.glob(os.path.join(TUNING, 'fullgrid_*.csv')):
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                if 'meanDiag' not in r:
                    continue
                ds = r['dataset'].strip().strip('"')
                key = (ds, _f(r['lambda1']), _f(r['lambda2']), _f(r['gamma']))
                md = _f(r.get('meanDiag'))
                if md is not None and all(k is not None for k in key[1:]):
                    out[key] = md
    return out


def main():
    rows = _load_cands()
    mean_diag = _load_meandiag()
    if not rows:
        print('No tuning CSVs found.', file=sys.stderr)
        sys.exit(1)

    selected = []
    for ds in DATASETS:
        cands = [r for r in rows if r['dataset'].strip().strip('"') == ds]
        eligible = []
        for r in cands:
            conv = int(float(r['converged']))
            fin = int(float(r['finite_ok']))
            nnz = int(float(r['nnz_ok']))
            gap = _f(r['eigengap'])
            ek = _f(r['eig_K'])
            if conv == 1 and fin == 1 and nnz == 1 and gap is not None and ek is not None:
                eligible.append(r)
        n_total = len(cands)
        n_conv = sum(1 for r in cands if int(float(r['converged'])) == 1)
        n_finite = sum(1 for r in cands if int(float(r['finite_ok'])) == 1)
        n_nnz = sum(1 for r in cands if int(float(r['nnz_ok'])) == 1)
        print(f'[{ds}] total={n_total} converged={n_conv} finite={n_finite} '
              f'nnz={n_nnz} eligible={len(eligible)}')

        best = None
        rule = 'guarded_eigengap'
        if eligible:
            max_ek = max(_f(r['eig_K']) for r in eligible)
            guard = FRAC * max_ek
            guarded = [r for r in eligible if _f(r['eig_K']) >= guard]
            max_gap_full = max(_f(r['eigengap']) for r in eligible)
            max_gap_guarded = max(_f(r['eigengap']) for r in guarded)
            if max_gap_full < TAU:
                rule = 'degenerate_fallback_max_eig_K'
                best = max(eligible, key=lambda r: _f(r['eig_K']))
            elif max_gap_guarded < TAU:
                rule = 'overregularized_fallback_max_meanDiag'
                # meanDiag is needed only here (yale). Fall back to max eig_K if
                # a candidate's meanDiag is unavailable for any reason.
                def md(r):
                    key = (ds, _f(r['lambda1']), _f(r['lambda2']), _f(r['gamma']))
                    v = mean_diag.get(key)
                    return v if v is not None else -float('inf')
                best = max(guarded, key=md)
                if md(best) == -float('inf'):
                    print('   WARNING: meanDiag unavailable; using max eig_K fallback.')
                    best = max(guarded, key=lambda r: _f(r['eig_K']))
                    rule = 'overregularized_fallback_max_eig_K'
            else:
                best = max(guarded, key=lambda r: _f(r['eigengap']))
            print(f'   max_gap_full={max_gap_full:.6g} max_gap_guarded='
                  f'{max_gap_guarded:.6g} max_eig_K={max_ek:.6g} -> rule={rule}')

        if best is not None:
            r = best
            sel = {
                'dataset': ds,
                'lambda1': r['lambda1'],
                'lambda2': r['lambda2'],
                'gamma': r['gamma'],
                'eigengap': _f(r['eigengap']),
                'eig_K': _f(r['eig_K']),
                'eig_Kp1': _f(r['eig_Kp1']),
                'idx': int(float(r['idx'])),
                'n_eligible': len(eligible),
                'rule': rule,
            }
            selected.append(sel)
            print(f"   -> SELECTED idx={sel['idx']} l1={r['lambda1']} "
                  f"l2={r['lambda2']} g={r['gamma']} eig_K={sel['eig_K']:.6g} "
                  f"eigengap={sel['eigengap']:.6g} [{rule}]")
        else:
            print('   -> NO eligible candidate (locked values retained)')
            selected.append({'dataset': ds, 'lambda1': '', 'lambda2': '',
                             'gamma': '', 'eigengap': '', 'eig_K': '',
                             'eig_Kp1': '', 'idx': '', 'n_eligible': 0,
                             'rule': 'none'})

    out = os.path.join(TUNING, 'selected_configs.csv')
    fields = ['dataset', 'lambda1', 'lambda2', 'gamma', 'eigengap',
              'eig_K', 'eig_Kp1', 'idx', 'n_eligible', 'rule']
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in selected:
            w.writerow(s)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
