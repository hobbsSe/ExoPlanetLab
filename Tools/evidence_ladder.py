#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Collect an evidence ladder: how many companions do the data support?

Each source is fitted with a fixed number of companions N = 1, 2, 3, ... in
separate runs (EXOPLANETLAB_N_FIT=N, EXOPLANETLAB_EVIDENCE=1; job names ending
in '-N<k>', see jobs_evidence.txt). Every such run stores in its
posterior_samples.npz

  logz, logzerr   ln Z(N) from the parallel-tempered chains (bridge sampling,
                  solver_ptemcee.evidence_estimates; the ladder reaches beta = 0,
                  so this is the evidence for the normalised prior, including
                  the period ordering and EXOPLANETLAB_ORBIT_SEPARATION)
  logz0           ln Z(0), no companion, same data and noise model, integrated
                  directly over the noise parameters (identical in every run of
                  a source up to rounding; checked here)

Per source the script prints ln Z(N), the log Bayes factor between
consecutive N, ln B(N vs N - 1) = ln Z(N) - ln Z(N - 1) (B the ratio of the
two evidences), the posterior probability of each N (uniform prior over the N
present, or --prior-decay r: p(N) ~ r^N), and the adopted N: the largest N
whose ln B over N - 1 exceeds --threshold (default 5, odds of about 150:1,
'strong' on the Jeffreys / Kass & Raftery scale), stepping up from N = 0. The
error of the bridge estimate is statistical only;
a run whose chains did not converge (run.log) is flagged, since its ln Z may
be biased (a missed mode lowers it).

Usage:
    python3 Tools/evidence_ladder.py EV              # every run whose tag contains 'EV-'
    python3 Tools/evidence_ladder.py EV --threshold 3 --prior-decay 0.5
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from scipy.special import logsumexp

# the main code, data and Results/ live one level up (this script is in Tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'Results')


def collect(prefix, res=RES):
    """{source tag: {N: dict(logz, logzerr, logz0, converged, dir)}}; the latest
    run wins where a (source, N) was run more than once. Runs of the same
    (source, N) from other starting candidates (tags '...-N<k>r<rank>',
    EXOPLANETLAB_SEED_RANK) are alternatives: the one with the highest ln Z is
    kept, and the others are listed with it ('starts')."""
    out, alts = {}, {}
    pat = os.path.join(res, f'*plt_*-{prefix}-*')
    for d in sorted(glob.glob(pat), key=os.path.getmtime):
        f = os.path.join(d, 'posterior_samples.npz')
        m = re.search(rf'-({re.escape(prefix)}-.+)-N(\d+)(?:r(\d+))?$', os.path.basename(d))
        if not (m and os.path.exists(f)):
            continue
        z = np.load(f, allow_pickle=True)
        if 'logz' not in z.files:
            continue
        n = int(z['n_fit']) if 'n_fit' in z.files else int(m.group(2))
        log = open(os.path.join(d, 'run.log')).read() if os.path.exists(
            os.path.join(d, 'run.log')) else ''
        alts.setdefault((m.group(1), n), {})[int(m.group(3) or 1)] = dict(
            logz=float(z['logz']), logzerr=float(z['logzerr']),
            logz0=float(z['logz0']) if 'logz0' in z.files else np.nan,
            accel={m: float(z['logz_' + m]) for m in ('accel7', 'accel9', 'harm2', 'harm23')
                   if 'logz_' + m in z.files and np.isfinite(z['logz_' + m])},
            # the tau verdict exists for the MCMC solvers only; nested samplers
            # run to their evidence tolerance
            converged=('NOT CONVERGED' not in log) and (
                'CONVERGED' in log or ('solver' in z.files and str(z['solver']) in
                                       ('dynesty', 'nessai'))),
            stable=bool(z['logz_stable']) if 'logz_stable' in z.files else None,
            noise=str(z['astro_noise']) if 'astro_noise' in z.files else '?',
            dir=os.path.basename(d))
    for (src, n), by_rank in alts.items():
        best = max(by_rank, key=lambda k: by_rank[k]['logz'])
        out.setdefault(src, {})[n] = dict(
            by_rank[best], starts={k: (v['logz'], v['dir']) for k, v in sorted(by_rank.items())},
            start=best)
    return out


def ladder(runs, threshold, prior_decay):
    """Rows (N, lnZ, err, lnB vs N-1, err, p(N | data), converged) and the adopted N."""
    z0 = np.array([r['logz0'] for r in runs.values() if np.isfinite(r['logz0'])])
    Ns = sorted(runs)
    lz = {n: runs[n]['logz'] for n in Ns}
    le = {n: runs[n]['logzerr'] for n in Ns}
    if len(z0):
        lz[0], le[0] = float(np.median(z0)), 0.0
    Ns = sorted(lz)
    lnp = np.array([lz[n] + n * np.log(prior_decay) for n in Ns])
    post = np.exp(lnp - logsumexp(lnp))
    rows, adopted = [], (0 if 0 in lz else None)
    for k, n in enumerate(Ns):
        prev = n - 1 if (n - 1) in lz else None
        lnb = lz[n] - lz[prev] if prev is not None else np.nan
        lnb_err = np.hypot(le[n], le[prev]) if prev is not None else np.nan
        rows.append((n, lz[n], le[n], lnb, lnb_err, post[k],
                     runs[n]['converged'] if n in runs else True))
        if adopted is not None and prev == adopted and lnb > threshold:
            adopted = n
    spread = float(np.ptp(z0)) if len(z0) > 1 else 0.0
    return rows, adopted, spread


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prefix', nargs='?', default='EV',
                    help="job-name prefix of the ladder runs (default 'EV'): "
                         "tags '<prefix>-<source>-N<k>'")
    ap.add_argument('--threshold', type=float, default=5.0,
                    help='ln B over N - 1 needed to adopt N (default 5)')
    ap.add_argument('--prior-decay', type=float, default=1.0,
                    help='prior p(N) ~ r^N over the N present (default 1: uniform)')
    ap.add_argument('--results', default=RES, help='results directory (default Results/)')
    ap.add_argument('--out', default=None,
                    help='also write the table here (default Results/evidence_ladder_<prefix>.txt)')
    args = ap.parse_args()

    data = collect(args.prefix, args.results)
    if not data:
        sys.exit(f"no evidence runs with tag '{args.prefix}-<source>-N<k>' in {args.results}")
    lines = [f"Evidence ladder '{args.prefix}': ln B threshold {args.threshold:g}, "
             f"prior p(N) ~ {args.prior_decay:g}^N"]
    for src in sorted(data):
        runs = data[src]
        rows, adopted, spread = ladder(runs, args.threshold, args.prior_decay)
        noise = sorted({r['noise'] for r in runs.values()})
        lines.append(f"\n=== {src}  (noise model {', '.join(noise)})")
        lines.append(f"  {'N':>2s} {'ln Z':>14s} {'ln B vs N-1':>18s} {'p(N|data)':>10s}  run")
        for n, z, ze, b, be, p, conv in rows:
            bs = f"{b:+10.2f} +/- {be:4.2f}" if np.isfinite(b) else ''
            flag = '' if conv else '  NOT CONVERGED'
            if n in data[src] and data[src][n]['stable'] is False:
                flag += '  ln Z NOT STABLE'
            if n in data[src] and len(data[src][n].get('starts', {})) > 1:
                st = data[src][n]['starts']
                flag += ('  [best of starts ' + ', '.join(
                    f"r{k}: {v[0]:.2f}" for k, v in st.items()) + ']')
            d = data[src][n]['dir'] if n in data[src] else '(direct integration)'
            lines.append(f"  {n:2d} {z:9.2f} +/- {ze:4.2f} {bs:>18s} {p:10.3g}  {d}{flag}")
        # acceleration models: no companion orbit, same normalisation
        acc = {m: v for r in runs.values() for m, v in r['accel'].items()}
        lz = {n: z for n, z, *_ in rows}
        for m, v in acc.items():
            best = max((n for n in lz if n > 0), key=lambda n: lz[n], default=None)
            lines.append(f"  {m} {v:9.2f}  (direct integration)  ln B vs N = 0 "
                         f"{v - lz[0]:+.2f}" + (f"; N = {best} vs {m} {lz[best] - v:+.2f}"
                                                if best else ''))
        if acc and adopted and max(acc.values()) > lz[adopted]:
            lines.append(f"  NOTE: an acceleration model has a higher ln Z than N = {adopted}: "
                         f"the data show curvature, not an established orbit")
        if spread > 0.05:
            lines.append(f"  WARNING: ln Z(0) differs by {spread:.2f} between the runs "
                         f"(different data or noise model?)")
        if any(r['stable'] is None for r in runs.values()):
            lines.append("  NOTE: runs without the stability test (stepping-stone ln Z, stopped on tau "
                         "alone) may be biased low for strong signals")
        if len(noise) > 1:
            lines.append("  WARNING: the runs use different noise models; ln Z not comparable")
        missing = [n for n in range(1, max(runs) + 1) if n not in runs]
        if missing:
            lines.append(f"  NOTE: no run for N = {missing}; the ladder steps only through consecutive N")
        top = max(runs)
        lines.append(f"  Adopted N = {adopted}" + (
            f"  (reached the largest N fitted, {top}: fit N = {top + 1} to check)"
            if adopted == top else ''))
    text = '\n'.join(lines)
    print(text)
    out = args.out or os.path.join(args.results, f'evidence_ladder_{args.prefix}.txt')
    open(out, 'w').write(text + '\n')
    print(f"\nWritten: {os.path.relpath(out, ROOT)}")


if __name__ == '__main__':
    main()
