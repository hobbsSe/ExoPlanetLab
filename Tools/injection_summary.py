#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Collect an injection-recovery campaign (make_injection_jobs.py + run_batch.py).

For every Results/ directory whose name contains the job prefix (default
'-INJ-') and holds injection_truth.json and posterior_samples.npz, compare the
posterior with the injected orbit:

  q      quantile of the true value in the posterior (the fraction of samples
         below it); a calibrated posterior gives q uniform in [0, 1]
  in68 / in95   truth inside the central 68% / 95% interval
  z      the normal score of q, Phi^-1(1 - q), with Phi the standard normal
         cumulative distribution function; positive when the posterior lies
         above the truth, and N(0, 1) for a calibrated posterior of any shape
         (a (median - truth)/width score is meaningless for bimodal posteriors,
         e.g. a narrow peak plus a mode containing the truth)

for P, e, T0 (compared modulo P), A, B, F, G, and the derived a0 and
inclination. Omega and omega are not compared: astrometry alone leaves them
ambiguous by 180 degrees. Per run:

  detected    the detect.py screen found a significant signal
  precise     the posterior median period is within --p-tol (default 20%) of
              the truth -- a statement about precision only: a weak signal or a
              period beyond the baseline gives an honest but broad posterior
  wrong mode  the true period lies outside the central 99% of the posterior
              AND the median is off by more than --p-tol (a harmonic, an alias
              or a noise peak; a narrow posterior with the truth ~3 sigma out in
              a tail is the right mode)

Calibration is only meaningful when the injections are drawn from the prior.
They are not in P and e: the injections stop at P = 2 x baseline and e = 0.9,
while the prior extends to P = 7305 d (~4 x baseline) and e = 1. For periods
beyond the baseline, where long eccentric orbits also fit the observed arc,
the posterior then legitimately spreads over periods never injected, and the
truth sits low in it (first campaigns: mean quantile 0.35 at 1-2 x baseline,
0.52 after restriction). So all quantities are computed from the posterior
restricted to the injected ranges (--p-max-frac, --e-max: the same values as
make_injection_jobs.py), i.e. the posterior under a prior equal to the
injection distribution in P and e; q_P_full keeps the unrestricted quantile.

The calibration (coverage, bias) is computed over ALL converged runs with a
detection -- selecting on 'precise' would remove exactly the tail cases that
coverage must count. A table of the true period's posterior quantile against
P/baseline shows any systematic lean (uniform quantiles = calibrated).

Writes Results/injection_summary.csv (one row per run) and prints coverage,
bias and detection efficiency per total-S bin.

Usage:  python3 Tools/injection_summary.py [--prefix INJ] [--results Results]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import norm

TOOLS = os.path.dirname(os.path.abspath(__file__))
# the main code, data and Results/ live one level up (this script is in Tools/)
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, ROOT)
from generate_sources import campbell_elements   # noqa: E402

PARAMS = ['P', 'e', 'T0', 'A', 'B', 'F', 'G', 'a0', 'inc']
P_TOL = 0.2          # set from --p-tol in main()
S_BINS = [5, 10, 20, 50, 100.001]


def compare(samples, truth):
    """(q, z, in68, in95) of truth against a 1-D sample array; z = Phi^-1(1 - q)
    (Phi the standard normal CDF), positive when the posterior lies above the
    truth."""
    q = float(np.mean(samples < truth))
    z = -float(norm.ppf(np.clip(q, 1e-4, 1 - 1e-4)))
    return q, z, 0.16 <= q <= 0.84, 0.025 <= q <= 0.975


P_MAX_FRAC, E_MAX_INJ = 2.0, 0.9   # set from --p-max-frac / --e-max in main()


def one_run(d):
    tr = json.load(open(os.path.join(d, 'injection_truth.json')))
    z = np.load(os.path.join(d, 'posterior_samples.npz'))
    names = [str(n) for n in z['names']]
    S = z['samples']
    q_P_full = float(np.mean(S[:, names.index('P')] < tr['P']))
    keep = ((S[:, names.index('P')] <= P_MAX_FRAC * tr['baseline_d'])
            & (S[:, names.index('e')] <= E_MAX_INJ))
    frac_removed = 1.0 - float(np.mean(keep))
    if keep.sum() >= 200:        # restrict to the injected ranges (see above)
        S = S[keep]
    col = {n: S[:, names.index(n)] for n in ('P', 'e', 'T0', 'A', 'B', 'F', 'G')}
    a0, _, _, inc = campbell_elements(col['A'], col['B'], col['F'], col['G'])
    col['a0'], col['inc'] = a0, np.degrees(inc)
    truth = dict(P=tr['P'], e=tr['e'], T0=tr['T0'], A=tr['A'], B=tr['B'], F=tr['F'],
                 G=tr['G'], a0=tr['a0_mas'], inc=tr['inc_deg'])
    # T0 is defined modulo P: compare the offset from the true T0, wrapped per sample.
    col['T0'] = ((col['T0'] - truth['T0'] + 0.5 * col['P']) % col['P']) - 0.5 * col['P']
    truth['T0'] = 0.0
    log = open(os.path.join(d, 'run.log')).read()
    m = re.search(r'Fast screening detected N = (\d+)', log)
    row = dict(run=os.path.basename(d.rstrip('/')), S=tr['snr_total'],
               snr_epoch=tr['snr_per_epoch'], P_true=tr['P'], e_true=tr['e'],
               inc_true=tr['inc_deg'], a0_true=tr['a0_mas'],
               P_frac=tr['P'] / tr['baseline_d'], geometry=tr['geometry_source_id'],
               n_screen=int(m.group(1)) if m else -1,
               converged='Overall: CONVERGED' in log,
               # Truly frozen = acceptance exactly 0 (the blind-start bug); the
               # solver's 'FROZEN CHAIN' warning fires below 0.05, which slow but
               # moving weak-signal chains also reach.
               joint_acc=(float(re.findall(r'^Acceptance fraction: ([\d.]+)', log, re.M)[-1])
                          if re.findall(r'^Acceptance fraction: ([\d.]+)', log, re.M) else float('nan')),
               P_med=float(np.median(col['P'])), q_P_full=q_P_full,
               frac_outside_injected=frac_removed)
    row['detected'] = row['n_screen'] >= 1
    row['precise'] = row['detected'] and abs(row['P_med'] / tr['P'] - 1.0) < P_TOL
    for p in PARAMS:
        q, zz, i68, i95 = compare(col[p], truth[p])
        row.update({f'q_{p}': q, f'z_{p}': zz, f'in68_{p}': i68, f'in95_{p}': i95})
    if 's_rel' in names:         # simulated relative astrometry: s = a_rel / a0
        info = json.load(open(os.path.join(d, 'run_info.json')))
        s_true = float(info['settings'].get('EXOPLANETLAB_RELAST_SIM_SCALE', 50.0))
        q, zz, i68, i95 = compare(S[:, names.index('s_rel')], s_true)
        row.update(q_s_rel=q, z_s_rel=zz, in68_s_rel=i68, in95_s_rel=i95)
    row['wrong_mode'] = (not (0.005 <= row['q_P'] <= 0.995)) and not row['precise']
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='INJ',
                    help='job-name prefix(es), comma-separated to combine campaigns (INJ2,INJ3)')
    ap.add_argument('--p-max-frac', type=float, default=2.0,
                    help='upper end of the injected P range / baseline (default 2.0, '
                         'as make_injection_jobs.py)')
    ap.add_argument('--e-max', type=float, default=0.9,
                    help='upper end of the injected e range (default 0.9)')
    ap.add_argument('--p-tol', type=float, default=0.2,
                    help='fractional period tolerance for the right mode (default 0.2)')
    ap.add_argument('--rerun-file', default=None,
                    help='write the jobs of frozen runs (from --jobs) to this file')
    ap.add_argument('--jobs', default=os.path.join(TOOLS, 'jobs_injection.txt'))
    ap.add_argument('--results', default=os.path.join(ROOT, 'Results'))
    ap.add_argument('--out', default=None,
                    help='output CSV (default <results>/injection_summary.csv)')
    args = ap.parse_args()
    global P_TOL, P_MAX_FRAC, E_MAX_INJ
    P_TOL, P_MAX_FRAC, E_MAX_INJ = args.p_tol, args.p_max_frac, args.e_max

    prefixes = [x.strip() for x in args.prefix.split(',') if x.strip()]   # e.g. INJ2,INJ3
    dirs = sorted(d for px in prefixes
                  for d in glob.glob(os.path.join(args.results, f'*-{px}-*'))
                  if os.path.exists(os.path.join(d, 'injection_truth.json')))
    rows, missing = [], []
    for d in dirs:
        if not os.path.exists(os.path.join(d, 'posterior_samples.npz')):
            missing.append(os.path.basename(d))
            continue
        try:
            rows.append(one_run(d))
        except Exception as ex:
            missing.append(f"{os.path.basename(d)} ({ex!r})")
    if not rows:
        sys.exit(f"No finished injection runs matching '*-{args.prefix}-*' in {args.results}.")

    out = args.out or os.path.join(args.results, 'injection_summary.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    cal = [r for r in rows if r['converged'] and r['detected']]
    print(f"{n} finished injection runs ({len(missing)} unfinished/unreadable): "
          f"{sum(r['converged'] for r in rows)} converged, "
          f"{sum(r['detected'] for r in rows)} detected by the screen, "
          f"{sum(r['precise'] for r in rows)} with P within {100*P_TOL:g}%, "
          f"{sum(r['in95_P'] for r in rows)} with the true P inside the 95% interval, "
          f"{sum(r['wrong_mode'] for r in rows)} in a wrong mode (true P outside 99%).")
    for m in missing:
        print(f"  not used: {m}")

    print("\nBy total S:            runs  detected  P within tol  wrong mode  converged")
    for lo, hi in zip(S_BINS[:-1], S_BINS[1:]):
        b = [r for r in rows if lo <= r['S'] < hi]
        if b:
            f = lambda k: sum(r[k] for r in b) / len(b)
            print(f"  S {lo:>3g}-{min(hi, 100):<3g}          {len(b):4d}  {f('detected'):7.0%}  "
                  f"{f('precise'):11.0%}  {f('wrong_mode'):10.0%}  {f('converged'):9.0%}")

    print(f"\nCalibration over the {len(cal)} converged runs with a detection "
          f"(expected: 68% and 95% coverage, median z ~ 0, rms z ~ 1):")
    print(f"  {'param':6s} {'in 68%':>7s} {'in 95%':>7s} {'median z':>9s} {'rms z':>7s}")
    for p in PARAMS + (['s_rel'] if cal and all('z_s_rel' in r for r in cal) else []):
        if not cal:
            break
        zz = np.array([r[f'z_{p}'] for r in cal])
        print(f"  {p:6s} {np.mean([r[f'in68_{p}'] for r in cal]):7.0%} "
              f"{np.mean([r[f'in95_{p}'] for r in cal]):7.0%} {np.median(zz):9.2f} "
              f"{np.sqrt(np.mean(zz**2)):7.2f}")

    print(f"\nPosterior restricted to the injected ranges (P <= {P_MAX_FRAC:g} x baseline, "
          f"e <= {E_MAX_INJ:g}); mass removed: median "
          f"{np.median([r['frac_outside_injected'] for r in rows]):.0%}, max "
          f"{np.max([r['frac_outside_injected'] for r in rows]):.0%}.")
    print("Posterior quantile of the true period by P/baseline (calibrated: mean 0.50,"
          " spread ~0.29; 'full' = unrestricted posterior):")
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.01)]:
        b = [r['q_P'] for r in cal if lo <= r['P_frac'] < hi]
        if b:
            bf = [r['q_P_full'] for r in cal if lo <= r['P_frac'] < hi]
            print(f"  P/baseline {lo:.1f}-{min(hi, 2.0):.1f}: {len(b):3d} runs, mean q {np.mean(b):.2f} "
                  f"(full {np.mean(bf):.2f}), "
                  f"fraction q < 0.16: {np.mean(np.array(b) < 0.16):.0%}, q > 0.84: "
                  f"{np.mean(np.array(b) > 0.84):.0%}")

    bad = [r for r in rows if not r['converged'] or r['wrong_mode']]
    if bad:
        print("\nRuns to inspect (not converged, or true P outside the posterior's 99%):")
        for r in bad:
            print(f"  {r['run']}: S={r['S']:.1f}, P_true={r['P_true']:.1f} d, "
                  f"P_med={r['P_med']:.1f} d, converged={r['converged']}, "
                  f"q(P_true)={r['q_P']:.3f}")
    frozen = sorted({r['run'].rsplit('-', 2)[-2] + '-' + r['run'].rsplit('-', 1)[-1]
                     for r in rows if r['joint_acc'] < 1e-3})
    if frozen:
        print(f"\nFrozen runs (acceptance 0; re-run them): "
              + ', '.join(frozen))
        if args.rerun_file:
            lines = [l for l in open(args.jobs) if l.split() and l.split()[0] in frozen]
            with open(args.rerun_file, 'w') as fh:
                fh.write('# Re-run of frozen injection runs\n' + ''.join(lines))
            print(f"Wrote {len(lines)} jobs to {args.rerun_file}: "
                  f"python3 Tools/run_batch.py {args.rerun_file}")
    print(f"\nPer-run table: {out}")


if __name__ == '__main__':
    main()
