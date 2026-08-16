#!/usr/bin/env python3
"""Analyze candidate label-free selection criteria on the full 90-candidate grid.

Loads all *_cand*.csv (the pre-registered 90-candidate/dataset tuning output,
which records eig_K, eig_Kp1, eigengap per candidate but NO labels), and for
each dataset reports what configuration several candidate criteria would select.
This is label-free: no NMI is read from the tuning grid (NMI lives only in the
separate diag_landscape_*.csv files, used here solely to SCORE criteria).
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['uci', 'bbc', 'yale', 'orl', 'scene']


def load_full():
    rows = []
    for f in glob.glob(os.path.join(ROOT, '*_cand*.csv')):
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def load_diag():
    """Return {dataset: {key:(l1,l2): value}} from diag_landscape CSVs (has NMI)."""
    diag = {}
    for f in glob.glob(os.path.join(ROOT, 'diag_landscape_*.csv')):
        ds = os.path.basename(f).replace('diag_landscape_', '').replace('.csv', '')
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                key = (float(r['lambda1']), float(r['lambda2']))
                diag.setdefault(ds, {})[key] = r
    return diag


def f(x):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return None
    if v != v or abs(v) == float('inf'):
        return None
    return v


def eligible(rows, ds):
    out = []
    for r in rows:
        if r['dataset'].strip().strip('"') != ds:
            continue
        if int(float(r['converged'])) != 1:
            continue
        if int(float(r['finite_ok'])) != 1 or int(float(r['nnz_ok'])) != 1:
            continue
        ek, ekp = f(r['eig_K']), f(r['eig_Kp1'])
        gap = f(r['eigengap'])
        if ek is None or ekp is None or gap is None:
            continue
        out.append({
            'idx': int(float(r['idx'])),
            'l1': f(r['lambda1']), 'l2': f(r['lambda2']), 'g': f(r['gamma']),
            'eig_K': ek, 'eig_Kp1': ekp, 'gap': gap,
            'obj': f(r.get('objective')),
        })
    return out


def report(ds, cands, name, pick):
    """pick(cands) -> selected dict or None."""
    try:
        sel = pick(cands)
    except Exception as e:
        print(f'  {name}: ERROR {e}')
        return
    if sel is None:
        print(f'  {name}: NONE')
        return
    print(f'  {name}: l1={sel["l1"]:.4g} l2={sel["l2"]:.4g} g={sel["g"]:g} '
          f'eig_K={sel["eig_K"]:.4f} eig_Kp1={sel["eig_Kp1"]:.4f} '
          f'gap={sel["gap"]:.4f}')


def main():
    rows = load_full()
    diag = load_diag()

    for ds in DATASETS:
        c = eligible(rows, ds)
        c.sort(key=lambda x: -x['gap'])
        ek_all = [x['eig_K'] for x in c]
        max_ek = max(ek_all)
        p90_ek = sorted(ek_all)[int(0.9 * (len(ek_all) - 1))]
        med_ek = sorted(ek_all)[len(ek_all) // 2]
        print(f'\n=== {ds}: n_eligible={len(c)} eig_K range=[{min(ek_all):.4f}, {max_ek:.4f}] '
              f'p90={p90_ek:.4f} median={med_ek:.4f}')

        # Current rule: max gap.
        report(ds, c, 'max gap (current)', lambda s: max(s, key=lambda x: x['gap']))

        # Guard: eig_K >= frac * reference, then max gap.
        for ref_name, ref in [('max', max_ek), ('p90', p90_ek), ('median', med_ek)]:
            for frac in (0.5, 0.33):
                th = frac * ref
                def pick(s, th=th):
                    g = [x for x in s if x['eig_K'] >= th]
                    return max(g, key=lambda x: x['gap']) if g else None
                report(ds, c, f'gap | eig_K>={frac}*{ref_name}(={th:.3f})', pick)

        # Guard: eig_K > 0 only.
        report(ds, c, 'gap | eig_K>0',
               lambda s: max([x for x in s if x['eig_K'] > 0], key=lambda x: x['gap'])
               if any(x['eig_K'] > 0 for x in s) else None)

        # Max eig_K (no gap).
        report(ds, c, 'max eig_K', lambda s: max(s, key=lambda x: x['eig_K']))

        # Max eig_K among eig_K > 0.
        report(ds, c, 'max eig_K | eig_K>0',
               lambda s: max([x for x in s if x['eig_K'] > 0], key=lambda x: x['eig_K'])
               if any(x['eig_K'] > 0 for x in s) else None)

        # Guard on eig_Kp1 (K+1-th eigenvalue): reject over-regularization that
        # collapses eig_Kp1 while leaving a spurious gap. Then max gap.
        ekp_all = [x['eig_Kp1'] for x in c]
        max_ekp = max(ekp_all)
        p90_ekp = sorted(ekp_all)[int(0.9 * (len(ekp_all) - 1))]
        med_ekp = sorted(ekp_all)[len(ekp_all) // 2]
        print(f'  [eig_Kp1 range={min(ekp_all):.4f},{max_ekp:.4f} p90={p90_ekp:.4f} med={med_ekp:.4f}]')
        for ref_name, ref in [('max', max_ekp), ('p90', p90_ekp), ('median', med_ekp)]:
            for frac in (0.5, 0.33):
                th = frac * ref
                def pick(s, th=th):
                    g = [x for x in s if x['eig_Kp1'] >= th]
                    return max(g, key=lambda x: x['gap']) if g else None
                report(ds, c, f'gap | eig_Kp1>={frac}*{ref_name}(={th:.3f})', pick)
        # Absolute eig_Kp1 guards.
        for th in (0.15, 0.10, 0.08, 0.05):
            def pick(s, th=th):
                g = [x for x in s if x['eig_Kp1'] >= th]
                return max(g, key=lambda x: x['gap']) if g else None
            report(ds, c, f'gap | eig_Kp1>={th:.2f}(abs)', pick)


if __name__ == '__main__':
    main()
