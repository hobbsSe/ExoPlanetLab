#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Scaling of the run time with the number of measurements and of companions
.

Two parts.

1. Controlled timing of the likelihood, the cost of a sampler step: one step of
   the tempered fit evaluates the likelihood once for each walker at each
   temperature (64 x 16 = 1024 orbit sets by default), in one vectorised call.
   That call is timed here, on one core, for N_obs = 200 ... 3200 measurements
   (the observing geometry of a pre-release source, rows resampled) and 1-3
   orbits, for the Student-t likelihood (single-star terms refitted by IRLS for
   every orbit set) and for the Gaussian one (projected residuals). The
   exponent of the scaling with N_obs is fitted.

2. Measured wall time per step of the evidence runs in Results/ whose tags
   start with the given prefixes (default MP MPS MPT MPB BS: the two-companion
   tests, all on the 637-point geometry, one core each), by number of orbits,
   and the steps they needed.

    python3 Tools/scaling_benchmark.py
    python3 Tools/scaling_benchmark.py --source-id 1457486023639239296 --repeat 5

Writes Results/scaling_benchmark.txt.
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config   # noqa: E402

config.limit_blas_threads(1)

import gaia_dr4_epoch   # noqa: E402
import orbit            # noqa: E402
import robust           # noqa: E402


def _orbit_sets(rng, k, n_orb, t):
    """(k, n_orb, 7) plausible orbits: P 50-2000 d, e 0-0.8, TI of ~0.1 mas."""
    P = np.exp(rng.uniform(np.log(50.0), np.log(2000.0), (k, n_orb)))
    e = rng.uniform(0.0, 0.8, (k, n_orb))
    T0 = t.min() + rng.uniform(0.0, 1.0, (k, n_orb)) * P
    ti = rng.normal(0.0, 0.1, (k, n_orb, 4))
    return np.concatenate([P[..., None], e[..., None], T0[..., None], ti], axis=2)


def _time(fn, repeat):
    fn()                                    # warm-up
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def likelihood_timing(source_id, n_obs_list, n_orb_list, k, repeat, seed=0):
    d = gaia_dr4_epoch.load_source_epoch_data(source_id)
    rng = np.random.default_rng(seed)
    n0 = len(d['t_obs_day'])
    rows = []
    for n_obs in n_obs_list:
        idx = np.sort(rng.integers(0, n0, n_obs))
        t = np.asarray(d['t_obs_day'], float)[idx] + rng.normal(0.0, 0.01, n_obs)
        ang = np.asarray(d['scan_angle_rad'], float)[idx]
        sig = np.asarray(d['sigma_mas'], float)[idx]
        X = np.asarray(d['proj_design'], float)[idx]
        lam = d.get('proj_prior_prec')
        y = sig * rng.standard_t(3, n_obs)
        rob = robust.StudentTSingleStar(y, X, sig, lam=lam)
        # Gaussian model: projection (I - H) with H = X (X'WX + L)^-1 X'W
        W = 1.0 / sig ** 2
        A = X.T @ (X * W[:, None]) + (np.diag(lam) if lam is not None else 0.0)
        G = np.linalg.solve(A, (X * W[:, None]).T)          # (p, N)
        nu, f = np.full(k, 3.0), np.full(k, 1.0)
        for n_orb in n_orb_list:
            par = _orbit_sets(rng, k, n_orb, t)

            def model():
                m = np.zeros((k, n_obs))
                for j in range(n_orb):
                    m += orbit.orbital_model_AL_vec(t, par[:, j, :], ang)
                return m

            def student():
                return rob.loglike(model(), nu, f)

            def gauss():
                r = y[None, :] - model()
                r = r - (r @ G.T) @ X.T
                return -0.5 * np.sum((r / sig) ** 2, axis=1)

            rows.append((n_obs, n_orb, _time(model, repeat), _time(student, repeat),
                         _time(gauss, repeat)))
    return rows, n0


def measured_runs(prefixes, res):
    """{n_orbits: [(seconds per step, production steps, minutes)]} from run logs."""
    out = {}
    for d in glob.glob(os.path.join(res, '*plt_emcee_*')):
        m = re.match(r'(\d+)plt_emcee_\d{8}_\d{6}-(.+)$', os.path.basename(d))
        if not m or not any(m.group(2).startswith(p + '-') for p in prefixes):
            continue
        f = os.path.join(d, 'run.log')
        if not os.path.exists(f):
            continue
        log = open(f, errors='replace').read()
        st = re.findall(r'joint fit[^\n]*?\s(\d+):(\d+):(\d+(?:\.\d+)?)\s+[\d.]+%', log)
        steps = re.findall(r'production = (\d+) steps', log)
        burn = re.findall(r'nsteps_burn\s*=\s*(\d+)|burn-in[^\n]*?(\d+) steps', log)
        if not (st and steps and 'Log evidence (N' in log):
            continue
        h, mi, s = st[-1]
        sec = 3600 * int(h) + 60 * int(mi) + float(s)
        n_prod = int(steps[-1])
        n_burn = next((int(a or b) for a, b in burn), 5000)
        out.setdefault(int(m.group(1)), []).append((sec / (n_prod + n_burn), n_prod, sec / 60.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-id', type=int, default=4181040337841125632)
    ap.add_argument('--n-obs', default='200,400,800,1600,3200')
    ap.add_argument('--n-orbits', default='1,2,3')
    ap.add_argument('--k', type=int, default=1024, help='orbit sets per call (walkers x temperatures)')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--prefixes', default='MP,MPS,MPT,MPB,BS')
    ap.add_argument('--results', default=os.path.join(ROOT, 'Results'))
    args = ap.parse_args()

    n_obs_list = [int(v) for v in args.n_obs.split(',')]
    n_orb_list = [int(v) for v in args.n_orbits.split(',')]
    rows, n0 = likelihood_timing(args.source_id, n_obs_list, n_orb_list, args.k, args.repeat)

    L = [f"Scaling benchmark (Tools/scaling_benchmark.py), one core, BLAS limited to one thread.",
         f"Geometry of source {args.source_id} ({n0} measurements), rows resampled.",
         f"Time of one vectorised likelihood call for {args.k} orbit sets "
         f"(= one step of a tempered fit with {args.k} walker-temperatures), best of {args.repeat}:",
         "",
         f"  {'N_obs':>6s} {'orbits':>6s} {'orbit model':>12s} {'Student-t':>11s} {'Gaussian':>10s}"
         f" {'per evaluation (Student-t)':>28s}",
         f"  {'':>6s} {'':>6s} {'[ms]':>12s} {'[ms]':>11s} {'[ms]':>10s} {'[microseconds]':>28s}"]
    for n_obs, n_orb, tm, ts, tg in rows:
        L.append(f"  {n_obs:6d} {n_orb:6d} {1e3 * tm:12.1f} {1e3 * ts:11.1f} {1e3 * tg:10.1f}"
                 f" {1e6 * ts / args.k:28.1f}")
    L.append("")
    for n_orb in n_orb_list:
        sub = [(r[0], r[3], r[4]) for r in rows if r[1] == n_orb]
        if len(sub) > 2:
            x = np.log([s[0] for s in sub])
            a_s = np.polyfit(x, np.log([s[1] for s in sub]), 1)[0]
            a_g = np.polyfit(x, np.log([s[2] for s in sub]), 1)[0]
            L.append(f"  {n_orb} orbit(s): time ~ N_obs^{a_s:.2f} (Student-t), N_obs^{a_g:.2f} (Gaussian)")
    one = {r[0]: r for r in rows if r[1] == n_orb_list[0]}
    for n_obs in n_obs_list:
        extra = [f"{r[1]} orbits x{r[3] / one[n_obs][3]:.2f}" for r in rows
                 if r[0] == n_obs and r[1] != n_orb_list[0]]
        if extra:
            L.append(f"  N_obs = {n_obs}: Student-t, relative to one orbit: " + ', '.join(extra))

    runs = measured_runs(args.prefixes.split(','), args.results)
    L += ["", f"Measured evidence runs (tags {args.prefixes}; one core each, up to nine side by side):",
          f"  {'orbits':>6s} {'runs':>5s} {'s per step (median, range)':>30s} {'production steps (median, range)':>36s}"
          f" {'joint fit [min] (median, range)':>34s}"]
    for n in sorted(runs):
        a = np.array(runs[n])
        L.append(f"  {n:6d} {len(a):5d} "
                 f"{np.median(a[:, 0]):12.3f} ({a[:, 0].min():.3f}-{a[:, 0].max():.3f})".ljust(46)
                 + f"{int(np.median(a[:, 1])):14d} ({int(a[:, 1].min())}-{int(a[:, 1].max())})".ljust(38)
                 + f"{np.median(a[:, 2]):12.0f} ({a[:, 2].min():.0f}-{a[:, 2].max():.0f})")
    text = '\n'.join(L)
    print(text)
    out = os.path.join(args.results, 'scaling_benchmark.txt')
    with open(out, 'w') as fh:
        fh.write(text + '\n')
    print(f"\nWritten: {os.path.relpath(out)}")


if __name__ == '__main__':
    main()
