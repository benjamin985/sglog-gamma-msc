#!/usr/bin/env python3
"""Comprehensive label-free criterion sweep on the full 90-candidate grid.

Selection is done from *_cand*.csv (label-free fields eig_K/eig_Kp1/eigengap).
Scoring uses fullgrid_*.csv NMI for yale/scene and the known-good eigengap
configs for uci/bbc/orl (which are already validated good at full-data eval).
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['uci', 'bbc', 'yale', 'orl', 'scene']

CURRENT_GOOD = {
    'uci': (0.01, 0.001, 12, 0.984),
    'bbc': (0.01, 0.0001, 6, 0.963),
    'orl': (0.1, 0.001, 24, 0.989),
}


def f(x):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return None
    if v != v or abs(v) == float('inf'):
        return None
    return v


def load_cands():
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
        out.append({'l1': f(r['lambda1']), 'l2': f(r['lambda2']), 'g': f(r['gamma']),
                    'ek': ek, 'ekp': ekp, 'gap': gap})
    return out


def maxgap(c):
    return max(c, key=lambda x: x['gap'])


def maxek(c):
    return max(c, key=lambda x: x['ek'])


def maxekp(c):
    return max(c, key=lambda x: x['ekp'])


def guard_ek_then_gap(c, frac):
    m = max(x['ek'] for x in c)
    th = frac * m
    g = [x for x in c if x['ek'] >= th]
    return max(g, key=lambda x: x['gap']) if g else None


def guard_ekp_then_gap(c, frac):
    m = max(x['ekp'] for x in c)
    th = frac * m
    g = [x for x in c if x['ekp'] >= th]
    return max(g, key=lambda x: x['gap']) if g else None


def mingap_pos_ek(c):
    g = [x for x in c if x['ek'] > 0]
    return min(g, key=lambda x: x['gap']) if g else None


def fallback(c, tau):
    """max gap if informative; else max eig_K (degenerate-gap fallback)."""
    mg = max(x['gap'] for x in c)
    if mg < tau:
        return maxek(c)
    return maxgap(c)


def guard_fallback(c, tau, frac):
    mg = max(x['gap'] for x in c)
    if mg < tau:
        return maxek(c)
    return guard_ek_then_gap(c, frac)


RULES = {
    'max_gap (current)': maxgap,
    'max_eig_K': maxek,
    'max_eig_Kp1': maxekp,
    'gap | eig_K>=0.5max': lambda c: guard_ek_then_gap(c, 0.5),
    'gap | eig_K>=0.33max': lambda c: guard_ek_then_gap(c, 0.33),
    'gap | eig_Kp1>=0.5max': lambda c: guard_ekp_then_gap(c, 0.5),
    'min_gap | eig_K>0': mingap_pos_ek,
    'fallback tau=0.02': lambda c: fallback(c, 0.02),
    'guard+fallback tau=0.02 f=0.5': lambda c: guard_fallback(c, 0.02, 0.5),
    'guard+fallback tau=0.02 f=0.33': lambda c: guard_fallback(c, 0.02, 0.33),
    'guard+fallback tau=0.01 f=0.5': lambda c: guard_fallback(c, 0.01, 0.5),
}


def score(ds, sel, nmi):
    if sel is None:
        return None, None
    if ds in CURRENT_GOOD:
        cur = CURRENT_GOOD[ds]
        same = (abs(sel['l1'] - cur[0]) < 1e-9 and abs(sel['l2'] - cur[1]) < 1e-9
                and abs(sel['g'] - cur[2]) < 1e-9)
        return (cur[3] if same else None), same
    n = nmi.get((ds, sel['l1'], sel['l2'], sel['g']))
    return n, None


def main():
    rows = load_cands()
    nmi = load_fullgrid_nmi()
    print(f'fullgrid NMI cells: {len(nmi)}\n')
    pools = {ds: eligible(rows, ds) for ds in DATASETS}

    header = f'{"criterion":28s}' + ''.join(f'{ds:>10s}' for ds in DATASETS) + '  mean'
    print(header)
    for name, rule in RULES.items():
        vals = []
        line = f'{name:28s}'
        for ds in DATASETS:
            sel = rule(pools[ds])
            n, same = score(ds, sel, nmi)
            if n is None:
                line += f'{"--":>10s}'
                vals.append(float('nan'))
            else:
                vals.append(n)
                line += f'{n:>10.3f}'
        m = sum(x for x in vals if x == x) / sum(1 for x in vals if x == x)
        print(f'{line}  {m:.4f}')

    # Detailed per-rule selections for yale/scene.
    print('\n--- selections (yale/scene) ---')
    for name, rule in RULES.items():
        for ds in ('yale', 'scene'):
            sel = rule(pools[ds])
            if sel is None:
                print(f'  {name} [{ds}]: NONE')
                continue
            n = nmi.get((ds, sel['l1'], sel['l2'], sel['g']))
            n_s = f'{n:.4f}' if n is not None else 'n/a'
            print(f'  {name} [{ds}]: l1={sel["l1"]:.4g} l2={sel["l2"]:.4g} g={sel["g"]:g} '
                  f'ek={sel["ek"]:.4f} gap={sel["gap"]:.4f} -> NMI {n_s}')


if __name__ == '__main__':
    main()
