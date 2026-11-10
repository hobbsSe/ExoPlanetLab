#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
False-positive rate of the signal screen on data without a companion.

The screen (detect.py) decides whether a run fits an orbit at all, so its
false-alarm rate on companion-free data is the false-positive rate of the
pipeline. This runs it -- with exactly the settings ExoPlanetLab.py uses
(P from 1 d to 2 x the 10-yr mission, N_MAX screens, bootstrap FAP threshold
1e-2, the upper-bound rule of BOUNDARY_FAP_MAX) -- on:

  white   pure noise in the observing geometry of the five quiet DR4
          pre-release sources used for the injection tests: data = (I-H) n,
          n ~ N(0, sigma_i) at the quoted errors. Expected false-positive rate:
          the FAP threshold, 1%, if the bootstrap FAP is calibrated.
  excess  the same plus white noise of EXCESS_FRAC x sigma_i that the quoted
          errors do not include (default 1.0, i.e. chi2/dof ~ 2, as real data
          show 1.4-3). The screen uses the quoted errors; its permutation
          bootstrap should be insensitive to the scale of white noise.
  real    the actual single-star residuals of the five quiet sources (one per
          source; they were chosen as quiet, so this is a consistency check).

Correlated noise (per transit, basic angle, attitude) is not simulated.

Each realisation is one line of a JSON-lines file; the script skips those
already present, so an interrupted run resumes where it stopped. One core per
worker process.

Usage:
    python3 Tools/false_positive_test.py --n 200               # 5 x 200 per condition
    python3 Tools/false_positive_test.py --n 2 --geometries 1 --workers 2   # smoke test
    python3 Tools/false_positive_test.py --summary             # summarise only
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp

# the main code, data and Results/ live one level up (this script is in Tools/)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Same quiet geometries as the injection campaigns (make_injection_jobs.py).
GEOMETRIES = [2309425390592896, 20694084440761600, 435469040545191680,
              1663617687609809280, 4181040337841125632]
CONDITIONS = ('white', 'excess', 'real', 'student')
# 'student': noise sigma_i t_nu with nu = STUDENT_NU, as the injection
# campaigns (INJECT_NOISE_NU). With --robust 1 the screen uses the Student-t
# score of the residual (detect.py, robust screen); with --inject-snr S an orbit
# (INJECT_ORBIT, total signal S after the single-star fit) is added, and a
# 'detection' is then a true positive: the screen's sensitivity.
STUDENT_NU = 3.0
INJECT_ORBIT = (500.0, 0.3, 0.4, 60.0, 30.0, 90.0)     # P, e, phase, inc, Omega, omega
P_MIN, P_MAX = 1.0, 2 * 10 * 365.25          # as ExoPlanetLab.py (MISSION_YR = 10)
OUT_DEFAULT = os.path.join(HERE, 'Results', 'false_positive', 'fp_screen.jsonl')

_GEO = {}


def _init_worker():
    from config import limit_blas_threads
    limit_blas_threads(1)


def _geometry(sid):
    if sid not in _GEO:
        import numpy as np
        import gaia_dr4_epoch
        import orbit
        g = gaia_dr4_epoch.load_source_epoch_data(sid)
        _GEO[sid] = dict(t=np.asarray(g['t_obs_day'], float),
                         ang=np.asarray(g['scan_angle_rad'], float),
                         sig=np.asarray(g['sigma_mas'], float),
                         real=np.asarray(g['along_scan_mas'], float),
                         proj=orbit.SingleStarProjector(g['proj_design'], g['proj_gain']))
    return _GEO[sid]


def run_one(task):
    """task = (condition, geometry index, realisation index, excess_frac)."""
    import numpy as np
    import detect
    from config import N_MAX
    cond, gi, k, excess, robust, snr = task
    sid = GEOMETRIES[gi]
    g = _geometry(sid)
    seed = 10_000_000 * CONDITIONS.index(cond) + 100_000 * gi + k
    if cond == 'real':
        data = g['real']
    else:
        rng = np.random.default_rng(seed)
        if cond == 'student':
            n = g['sig'] * rng.standard_t(STUDENT_NU, len(g['t']))
        else:
            n = g['sig'] * rng.standard_normal(len(g['t']))
        if cond == 'excess':
            n = n + excess * g['sig'] * rng.standard_normal(len(g['t']))
        if snr > 0:
            import orbit
            P, e, ph, inc, Om, om = INJECT_ORBIT
            unit = np.array([P, e, g['t'].min() + ph * P,
                             *orbit.ti_from_campbell(1.0, np.radians(om), np.radians(Om),
                                                     np.radians(inc))])
            s_unit = float(np.sqrt(np.sum((orbit.orbital_model_AL(
                g['t'], unit, g['ang'], projector=g['proj']) / g['sig']) ** 2)))
            unit[3:] *= snr / s_unit
            n = n + orbit.orbital_model_AL(g['t'], unit, g['ang'], projector=None)
        data = g['proj'].apply(n)
    t0 = time.time()
    r = detect.detect_signals(g['t'], g['ang'], data, g['sig'], p_min=P_MIN, p_max=P_MAX,
                              n_max_screen=N_MAX, n_pool=1, projector=g['proj'],
                              verbose=False, robust=bool(robust))
    return dict(condition=cond, geometry=sid, k=k, seed=seed, robust=int(robust),
                inject_snr=float(snr),
                excess_frac=excess if cond == 'excess' else 0.0,
                n_detected=int(r.n_detected),
                periods=[float(p[0]) for p in r.params],
                long_period=[bool(x) for x in getattr(r, 'long_period', [])],
                screens=[dict(screen=s['screen'], candidate_period=float(s['candidate_period']),
                              power=float(s['power']), fap=float(s['fap']),
                              n_bootstrap=int(s['n_bootstrap']), fap_method=s['fap_method'],
                              outcome=s['outcome']) for s in r.screens],
                n_dropped=len(r.dropped), seconds=round(time.time() - t0, 1))


def load(out):
    rows = []
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summary(rows):
    import numpy as np
    from scipy.stats import beta
    if not rows:
        print('No results yet.')
        return

    def ci(k, n):   # Clopper-Pearson 95%
        lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        return lo, hi

    groups = sorted({(CONDITIONS.index(r['condition']), r.get('robust', 0),
                      r.get('inject_snr', 0.0)) for r in rows})
    for ci_, rob, snr in groups:
        cond = CONDITIONS[ci_]
        R = [r for r in rows if r['condition'] == cond and r.get('robust', 0) == rob
             and r.get('inject_snr', 0.0) == snr]
        n = len(R)
        k = sum(r['n_detected'] > 0 for r in R)
        lo, hi = ci(k, n)
        extra = f", excess noise {R[0]['excess_frac']:g} x sigma" if cond == 'excess' else ''
        extra += ", ROBUST screen" if rob else ''
        extra += f", injected orbit S = {snr:g}" if snr > 0 else ''
        print(f"\n=== {cond}: {n} realisations{extra}; median {np.median([r['seconds'] for r in R]):.0f} s each")
        if snr > 0:
            good = sum(r['n_detected'] > 0 and abs(r['periods'][0] / INJECT_ORBIT[0] - 1) < 0.2
                       for r in R)
            print(f"Detections (screen found >= 1 signal): {k}/{n} = {100 * k / n:.1f}% "
                  f"[95%: {100 * lo:.1f}-{100 * hi:.1f}%]; first period within 20% of the "
                  f"injected one: {good}/{n}")
        else:
            print(f"False positives (screen found >= 1 signal): {k}/{n} = {100 * k / n:.2f}% "
                  f"[95%: {100 * lo:.2f}-{100 * hi:.2f}%]"
                  + ("  (nominal 1%)" if cond != 'real' else ''))
        for gi, sid in enumerate(GEOMETRIES):
            Rg = [r for r in R if r['geometry'] == sid]
            if Rg:
                print(f"   {sid:>20d}: {sum(r['n_detected'] > 0 for r in Rg):3d}/{len(Rg)}")
        first = [r['screens'][0] for r in R if r['screens']]
        outc = {}
        for s in first:
            outc[s['outcome']] = outc.get(s['outcome'], 0) + 1
        print("First-screen outcomes: " + ", ".join(f"{o} {c}" for o, c in sorted(outc.items())))
        faps = np.array([s['fap'] for s in first])
        if cond != 'real' and snr == 0:
            print("First-screen FAP <= x (calibrated: fraction = x): " + ", ".join(
                f"{x:g}: {np.mean(faps <= x):.3f}" for x in (0.01, 0.05, 0.1, 0.5)))
        nlong = sum(any(r['long_period']) for r in R)
        if nlong:
            print(f"Accepted long-period (upper-bound) detections: {nlong}")
        det = [r for r in R if r['n_detected'] > 0] if snr == 0 else []
        for r in det[:30]:
            fs = [s for s in r['screens'] if s['outcome'].startswith('detected')]
            print(f"   FP: geometry {r['geometry']} k={r['k']}: P = "
                  + ", ".join(f"{p:.1f}" for p in r['periods'])
                  + " d; FAP " + ", ".join(f"{s['fap']:.1e}" for s in fs)
                  + ("  [long-period]" if any(r['long_period']) else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=200,
                    help='noise realisations per geometry and condition (default 200)')
    ap.add_argument('--conditions', default='white,excess,real',
                    help='comma-separated subset of white,excess,real')
    ap.add_argument('--excess-frac', type=float, default=1.0,
                    help='extra white noise in units of the quoted errors (default 1.0)')
    ap.add_argument('--geometries', type=int, default=len(GEOMETRIES),
                    help='use the first N geometries (default all 5)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help='worker processes, one core each (default cores - 1)')
    ap.add_argument('--robust', type=int, default=0, choices=[0, 1],
                    help="1: the robust (Student-t score) screen (default 0)")
    ap.add_argument('--inject-snr', type=float, default=0.0,
                    help='add the test orbit with this total signal S (default 0: none)')
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--summary', action='store_true', help='only summarise existing results')
    args = ap.parse_args()

    rows = load(args.out)
    if args.summary:
        summary(rows)
        return
    done = {(r['condition'], GEOMETRIES.index(r['geometry']), r['k'])
            for r in rows if r.get('robust', 0) == args.robust
            and r.get('inject_snr', 0.0) == args.inject_snr}
    tasks = []
    for cond in args.conditions.split(','):
        cond = cond.strip()
        if cond not in CONDITIONS:
            sys.exit(f"unknown condition {cond!r}")
        for gi in range(args.geometries):
            for k in range(1 if cond == 'real' else args.n):
                if (cond, gi, k) not in done:
                    tasks.append((cond, gi, k, args.excess_frac, args.robust, args.inject_snr))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{len(tasks)} screens to run ({len(done)} already in {args.out}), "
          f"{args.workers} workers, one core each.", flush=True)
    t_start = time.time()
    if tasks:
        ctx = mp.get_context('fork')
        with ctx.Pool(args.workers, initializer=_init_worker) as pool, open(args.out, 'a') as fh:
            for i, r in enumerate(pool.imap_unordered(run_one, tasks), 1):
                fh.write(json.dumps(r) + '\n')
                fh.flush()
                if i % 10 == 0 or i == len(tasks):
                    el = time.time() - t_start
                    print(f"  {i}/{len(tasks)} done, {el / 60:.1f} min elapsed, "
                          f"~{el / i * (len(tasks) - i) / 60:.0f} min left", flush=True)
    summary(load(args.out))


if __name__ == '__main__':
    main()
