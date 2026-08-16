#!/usr/bin/env python3
"""End-to-end validation of candidate label-free selection criteria.

For yale and scene (the two datasets where the eigengap rule is broken), NMI is
available from fullgrid_*.csv (scored ONLY for criterion design, never used to
select). For uci/bbc/orl the current eigengap selections are already good; we
check whether a criterion PRESERVES them (same lambda1/lambda2/gamma).

The candidate criterion is expressed as a pure function of the label-free
per-candidate fields {eig_K, eig_Kp1, eigengap} plus grid-level statistics.
No NMI is used inside the criterion.
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['uci', 'bbc', 'yale', 'orl', 'scene']

# Current (already-good) selections for uci/bbc/orl — these are the eigengap
# selections, with NMI validated by the final 20-seed eval.
CURRENT_GOOD = {
    'uci':  (0.01, 0.001, 12, 0.984),
    'bbc':  (0.01, 0.0001, 6, 0.963),
    'orl':  (0.1, 0.001, 24, 0.989),
}


def f(x):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return None
    if v != v or abs(v) == float('inf'):
        return None
    return v


def load_full():
    rows = []
    for fp in glob.glob(os.path.join(ROOT, '*_cand*.csv')):
        with open(fp, newline='') as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def load_fullgrid_nmi():
    out = {}
    for fp in glob.glob(os.path.join(ROOT, 'fullgrid_*.csv')):
        ds = os.path.basename(fp).replace('fullgrid_', '').replace('.csv', '')
        with open(fp, newline='') as fh:
            for r in csv.DictReader(fh):
                out[(ds, f(r['lambda1']), f(r['lambda2']), f(r['gamma']))] = f(r['nmi'])
    return out


def eligible(rows, ds):
    out = []
    for r in rows:
        if r['dataset'].strip().strip('"') != ds:
            continue
        if int(float(r['converged'])) != 1:
            continue
        if int(float(r['finite_ok'])) != 1 or int(float(r['nnz_ok'])) != 1:
            continue
        ek, ekp, gap = f(r['eig_K']), f(r['eig_Kp1']), f(r['eigengap'])
        if ek is None or ekp is None or gap is None:
            continue
        out.append({'l1': f(r['lambda1']), 'l2': f(r['lambda2']),
                    'g': f(r['gamma']), 'eig_K': ek, 'eig_Kp1': ekp, 'gap': gap})
    return out


def select(cands, rule):
    """rule: (score_fn, guard_fn) or a callable taking list -> selected dict."""
    if callable(rule):
        return rule(cands)
    score_fn, guard_fn = rule
    g = [x for x in cands if guard_fn(x)]
    if not g:
        return None
    return max(g, key=score_fn)


def make_criteria():
    crits = {}
    # (a) current: max gap.
    crits['max_gap'] = (lambda x: x['gap'], lambda x: True)
    # (b) max eig_K.
    crits['max_eig_K'] = (lambda x: x['eig_K'], lambda x: True)
    # (c) max gap, guard eig_K >= frac*max_eig_K.
    # (d) max gap, guard eig_Kp1 >= frac*max_eig_Kp1.
    # (e) max eig_K, guard gap > 0 (reject identity with no gap).
    return crits


def grid_guards(cands, kind, fracs=(0.5, 0.33)):
    """Return dict of {label: guard_fn} using grid-statistic references."""
    vals = [x[kind] for x in cands]
    refs = {'max': max(vals),
            'p90': sorted(vals)[int(0.9 * (len(vals) - 1))],
            'med': sorted(vals)[len(vals) // 2]}
    guards = {}
    for rn, ref in refs.items():
        for frac in fracs:
            th = frac * ref
            guards[f'{kind}>={frac}*{rn}({th:.3f})'] = (lambda x, th=th: x[kind] >= th)
    return guards


def main():
    rows = load_full()
    nmi = load_fullgrid_nmi()
    print(f'fullgrid NMI cells loaded: {len(nmi)}')

    # Build candidate pools.
    pools = {ds: eligible(rows, ds) for ds in DATASETS}

    for ds in DATASETS:
        cands = pools[ds]
        print(f'\n===== {ds} (n={len(cands)}) =====')
        # Report top-N by NMI for yale/scene (ground truth reference, for design only).
        if ds in ('yale', 'scene'):
            scored = []
            for x in cands:
                n = nmi.get((ds, x['l1'], x['l2'], x['g']))
                if n is not None:
                    scored.append((n, x))
            scored.sort(key=lambda t: -t[0])
            print('  top-8 by NMI (reference):')
            for n, x in scored[:8]:
                print(f'    NMI={n:.4f} l1={x["l1"]:.4g} l2={x["l2"]:.4g} g={x["g"]:g} '
                      f'eig_K={x["eig_K"]:.4f} eig_Kp1={x["eig_Kp1"]:.4f} gap={x["gap"]:.4f}')

        # Evaluate criteria.
        print('  criteria -> selection -> NMI:')
        # max gap
        sel = select(cands, (lambda x: x['gap'], lambda x: True))
        _report(ds, sel, nmi, 'max_gap (current)')
        # max eig_K
        sel = select(cands, (lambda x: x['eig_K'], lambda x: True))
        _report(ds, sel, nmi, 'max_eig_K')
        # eig_K guards
        for label, guard in grid_guards(cands, 'eig_K').items():
            sel = select(cands, (lambda x: x['gap'], guard))
            _report(ds, sel, nmi, f'gap | {label}')
        # eig_Kp1 guards
        for label, guard in grid_guards(cands, 'eig_Kp1').items():
            sel = select(cands, (lambda x: x['gap'], guard))
            _report(ds, sel, nmi, f'gap | {label}')
        # max eig_K subject to gap > 0
        sel = select(cands, (lambda x: x['eig_K'], lambda x: x['gap'] > 0))
        _report(ds, sel, nmi, 'max_eig_K | gap>0')
        # max eig_K subject to gap > frac*max_gap
        max_gap = max(x['gap'] for x in cands)
        sel = select(cands, (lambda x: x['eig_K'], lambda x: x['gap'] > 0.3 * max_gap))
        _report(ds, sel, nmi, 'max_eig_K | gap>0.3*maxgap')


def _report(ds, sel, nmi, label):
    if sel is None:
        print(f'    {label}: NONE')
        return
    n = nmi.get((ds, sel['l1'], sel['l2'], sel['g']))
    n_s = f'{n:.4f}' if n is not None else 'n/a'
    if ds in CURRENT_GOOD:
        cur = CURRENT_GOOD[ds]
        same = (abs(sel['l1'] - cur[0]) < 1e-12 and abs(sel['l2'] - cur[1]) < 1e-12
                and abs(sel['g'] - cur[2]) < 1e-12)
        n_s += f' (cur_good={cur[3]:.3f}, preserved={same})'
    print(f'    {label}: l1={sel["l1"]:.4g} l2={sel["l2"]:.4g} g={sel["g"]:g} '
          f'eig_K={sel["eig_K"]:.4f} gap={sel["gap"]:.4f} -> NMI {n_s}')


if __name__ == '__main__':
    main()
