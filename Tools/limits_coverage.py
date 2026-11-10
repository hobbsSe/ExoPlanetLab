#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Coverage of the upper limits (limits.py) on injected companions.

Companions are injected into the geometry (times, scan angles, errors, design
matrix) of a quiet DR4 pre-release source, with Student-t (nu = 3) noise, and
the limit is computed at the true period. The coverage is the fraction of
cases in which the true a0 lies above the 68% and 95% limits.

    python3 Tools/limits_coverage.py                       # all three tests
    python3 Tools/limits_coverage.py --tests B --n 400 --workers 2

Tests:
    A  circular truths,  circular-only limit         (e_grid 0)
    B  circular truths,  marginalised limit          (e_grid 0, 0.3, 0.6; tool default)
    C  eccentric truths (e uniform in 0-0.6), marginalised limit

Truths: a0 uniform in 0 to --a0-max mas, P log-uniform in 30-1500 d,
isotropic orientation, periastron time uniform over one period.
Output: Results/limits_coverage/<test>.txt (one line per case) and a summary.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gaia_dr4_epoch   # noqa: E402
import limits           # noqa: E402

TESTS = {'A': dict(truth_e_max=0.0, e_grid=(0.0,)),
         'B': dict(truth_e_max=0.0, e_grid=(0.0, 0.3, 0.6)),
         'C': dict(truth_e_max=0.6, e_grid=(0.0, 0.3, 0.6))}

_G = {}


def _init(source_id):
    d = gaia_dr4_epoch.load_source_epoch_data(source_id)
    _G.update(X=np.asarray(d['proj_design'], float), sig=np.asarray(d['sigma_mas'], float),
              t=np.asarray(d['t_obs_day'], float), psi=np.asarray(d['scan_angle_rad'], float),
              lam=d.get('proj_prior_prec'))


def one_case(job):
    test, k, seed, a0_max, n_phase, n_orient = job
    cfg = TESTS[test]
    X, sig, t, psi, lam = _G['X'], _G['sig'], _G['t'], _G['psi'], _G['lam']
    rng = np.random.default_rng(seed)
    # truth (the same random draws for every test with the same case index k)
    a0 = rng.uniform(0.0, a0_max)
    P = float(np.exp(rng.uniform(np.log(30.0), np.log(1500.0))))
    T0 = t.min() + rng.uniform(0.0, P)
    e_u = rng.uniform(0.0, 1.0)
    u = limits.unit_orbits(1, rng)[0]
    noise = sig * rng.standard_t(3.0, size=len(t))
    e = cfg['truth_e_max'] * e_u
    d = limits.ti_columns(t, psi, P, e, T0) @ (a0 * u) + noise
    # limit at the true period
    nu, f = limits.noise_fit(d, X, sig, lam=lam, noise='student')
    ms, Cs = [], []
    for eg in cfg['e_grid']:
        for ph in (np.arange(n_phase) / n_phase if eg > 0 else (0.0,)):
            m, C = limits.ti_fit(d, X, sig, limits.ti_columns(t, psi, P, eg, t.min() + ph * P),
                                 nu, f, lam=lam)
            ms.append(m)
            Cs.append(C)
    units = limits.unit_orbits(n_orient, np.random.default_rng(10_000 + k))
    lim95 = limits.limit_at_period(ms, Cs, units, level=0.95)['a0_limit']
    lim68 = limits.limit_at_period(ms, Cs, units, level=0.68)['a0_limit']
    return test, k, a0, P, e, lim68, lim95


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='4181040337841125632', help='geometry (G = 9.0 star)')
    ap.add_argument('--tests', default='ABC')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--a0-max', type=float, default=0.12)
    ap.add_argument('--n-phase', type=int, default=8)
    ap.add_argument('--n-orient', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default=os.path.join(ROOT, 'Results', 'limits_coverage'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(T, k, 1000 + k, args.a0_max, args.n_phase, args.n_orient)
            for T in args.tests for k in range(args.n)]
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        res = pool.map(one_case, jobs, chunksize=4)
    lines = [f'Coverage of the upper limits, geometry {args.source}, Student-t nu = 3 noise, '
             f'{args.n} cases per test, a0 uniform in 0-{args.a0_max:g} mas, P 30-1500 d']
    for T in args.tests:
        r = np.array([x[2:] for x in res if x[0] == T])
        a0, lim68, lim95 = r[:, 0], r[:, 3], r[:, 4]
        with open(os.path.join(args.out, f'{T}.txt'), 'w') as fh:
            fh.write('# a0_true P e lim68 lim95\n')
            np.savetxt(fh, r, fmt='%.6g')
        lines.append(f"  {T}: truths e <= {TESTS[T]['truth_e_max']:g}, limit over e = "
                     f"{TESTS[T]['e_grid']}:  above 95% limit {100 * np.mean(a0 > lim95):.1f}%, "
                     f"above 68% limit {100 * np.mean(a0 > lim68):.1f}%")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main()
