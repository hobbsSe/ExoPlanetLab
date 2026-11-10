#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Run-time table of the runs in Results/.

Reads every run.log and groups the finished runs by solver, number of
companions fitted, RVs, evidence mode and the number of cores (1 for jobs of
Tools/run_batch.py). Per group: the number of runs, the run time (median and
range), the production steps of the tempered / emcee fit and the time per
1000 steps. Runs that overlap a busy machine are not separated out: the
ranges show them.

Usage
-----
    python3 Tools/runtime_summary.py                 # all runs
    python3 Tools/runtime_summary.py --since 20261002
    python3 Tools/runtime_summary.py --out Results/runtime_summary.txt
"""

import argparse
import glob
import os
import re

import numpy as np

# the main code, data and Results/ live one level up (this script is in Tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'Results')


def parse(d):
    """One run folder -> dict, or None if the run did not finish."""
    f = os.path.join(d, 'run.log')
    m = re.match(r'(\d+)plt_([a-z]+)_(\d{8})_\d{6}', os.path.basename(d))
    if not (m and os.path.exists(f)):
        return None
    log = open(f, errors='replace').read()
    dur = re.findall(r'^Duration:\s+(\d+):(\d+):(\d+)', log, re.M)
    if not dur:
        return None
    h, mi, s = (int(v) for v in dur[-1])
    pt = re.findall(r'Running PT-MCMC: \d+ walkers x (\d+) temperatures.*?'
                    r'(?:in up to (\d+) parallel blocks|(\d+) workers)', log)
    em = re.findall(r'Running MCMC: \d+ walkers.*?(\d+) workers', log)
    cores = (int(pt[-1][1] or pt[-1][2]) if pt else int(em[-1]) if em else 1)
    steps = re.findall(r'production = (\d+) steps', log)
    return dict(n=int(m.group(1)), solver=m.group(2), date=m.group(3),
                minutes=60 * h + mi + s / 60.0,
                rv='RV data used = True' in log,
                evidence='Log evidence (N' in log,
                temps=int(pt[-1][0]) if pt else 0,
                cores=1 if cores <= 1 else cores,
                # the nested samplers have no production steps (a 'production'
                # line in their logs is the emcee fit that places their box)
                steps=int(steps[-1]) if steps and m.group(2) == 'emcee' else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='', help='only runs from this date on (YYYYMMDD)')
    ap.add_argument('--results', default=RES, help='results directory (default Results/)')
    ap.add_argument('--out', default=None, help='also write the table here')
    args = ap.parse_args()

    runs = [r for r in (parse(d) for d in glob.glob(os.path.join(args.results, '*plt_*')))
            if r and r['date'] >= args.since]
    groups = {}
    for r in runs:
        key = (r['solver'], r['n'], r['rv'], r['evidence'], r['temps'], r['cores'])
        groups.setdefault(key, []).append(r)
    lines = [f"Run times of {len(runs)} finished runs in {args.results}"
             + (f" since {args.since}" if args.since else ''),
             f"{'solver':8s} {'N':>2s} {'RV':>3s} {'ln Z':>4s} {'temps':>5s} {'cores':>5s} "
             f"{'runs':>4s} {'median min':>10s} {'range min':>15s} {'median steps':>12s} "
             f"{'s / 1000 steps':>14s}"]
    for key in sorted(groups):
        g = groups[key]
        t = np.array([r['minutes'] for r in g])
        st = np.array([r['steps'] for r in g], dtype=float)
        ok = st > 0
        per = (f"{np.median(60 * t[ok] / st[ok] * 1000):14.0f}" if np.any(ok) else f"{'':14s}")
        lines.append(f"{key[0]:8s} {key[1]:2d} {'yes' if key[2] else 'no':>3s} "
                     f"{'yes' if key[3] else 'no':>4s} {key[4]:5d} {key[5]:5d} {len(g):4d} "
                     f"{np.median(t):10.1f} {t.min():7.1f}-{t.max():<7.1f} "
                     + (f"{np.median(st[ok]):12.0f}" if np.any(ok) else f"{'':12s}") + ' ' + per)
    lines.append("(run time is the whole run: screen, peels, joint fit, plots; s / 1000 steps is "
                 "that time over the joint fit's production steps)")
    print('\n'.join(lines))
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"\nWritten: {os.path.relpath(args.out)}")


if __name__ == '__main__':
    main()
