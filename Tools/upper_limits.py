#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Upper limits on undetected companions against period, for sources of the Gaia
DR4 pre-release epoch astrometry (limits.py).

    python3 Tools/upper_limits.py 4181040337841125632
    python3 Tools/upper_limits.py 4181040337841125632 --m-star 1.1
    python3 Tools/upper_limits.py all --no-plot

For every period of a logarithmic grid: the upper limit on the semi-major axis
a0 of the photocentre orbit (95% by default; isotropic orientation, prior
uniform in a0), the a0 that would be detected at S = 8, the evidence ratio of
a companion at that period to none, and -- with --m-star -- the corresponding
mass of a dark companion, using the parallax of the single-star fit. A period
where ln B is clearly positive holds a signal: the limit there is not an
upper limit on an undetected companion. No sampling; a second or two per
source.

'all' runs every source of the file. Output: Results/upper_limits/<id>.txt,
.npz and .png.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gaia_dr4_epoch   # noqa: E402
import limits           # noqa: E402

MJUP = 1047.57          # Jupiter masses per solar mass
BLUE, ORANGE = '#2a78d6', '#eb6834'


def run(source_id, args):
    d = gaia_dr4_epoch.load_source_epoch_data(source_id)
    y, X, sig, t, psi = (np.asarray(d[k], dtype=float) for k in
                         ('proj_data', 'proj_design', 'sigma_mas', 't_obs_day', 'scan_angle_rad'))
    lam = d.get('proj_prior_prec')
    span = float(t.max() - t.min())
    periods = np.exp(np.linspace(np.log(args.p_min), np.log(args.p_max_span * span), args.n_periods))
    e_grid = tuple(float(v) for v in args.e_grid.split(','))
    r = limits.upper_limits(y, X, sig, t, psi, periods, lam=lam, level=args.level,
                            s_det=args.s_det, noise=args.noise, e_grid=e_grid, seed=args.seed)
    names = list(d['proj_param_names'])
    w = 1.0 / sig ** 2
    lam_ = np.zeros(X.shape[1]) if lam is None else np.asarray(lam, float)
    plx = float(np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(lam_), X.T @ (w * y))
                [names.index('parallax')])
    mass = sens_mass = None
    if args.m_star and plx > 0:
        mass = limits.companion_mass(r['a0_limit'], periods, plx, args.m_star) * MJUP
        sens_mass = limits.companion_mass(r['a0_sens'], periods, plx, args.m_star) * MJUP

    head = (f"# Upper limits for source {source_id}: G = {d.get('g_mag', float('nan')):.2f}, "
            f"{len(t)} measurements over {span:.0f} d, single-star parallax {plx:.3f} mas\n"
            f"# noise model {args.noise}: nu = {r['nu']:.2f}, f = {r['f']:.3f}; "
            f"eccentricities {e_grid}; level {args.level:.2f}; sensitivity at S = {args.s_det:g}"
            + (f"; host mass {args.m_star:g} Msun" if args.m_star else '') + '\n'
            f"# {'P [d]':>8s} {'a0 limit':>10s} {'a0 at S_det':>12s} {'ln B':>8s} {'S best':>7s}"
            + (f" {'M limit':>10s} {'M at S_det':>11s}" if mass is not None else '') + '\n'
            f"# {'':>8s} {'[mas]':>10s} {'[mas]':>12s} {'':>8s} {'':>7s}"
            + (f" {'[MJup]':>10s} {'[MJup]':>11s}" if mass is not None else ''))
    lines = [head]
    for j, P in enumerate(periods):
        lines.append(f"  {P:8.2f} {r['a0_limit'][j]:10.4f} {r['a0_sens'][j]:12.4f} "
                     f"{r['ln_b'][j]:+8.2f} {r['s_best'][j]:7.2f}"
                     + (f" {mass[j]:10.3f} {sens_mass[j]:11.3f}" if mass is not None else ''))
    out = os.path.join(args.out, str(source_id))
    os.makedirs(args.out, exist_ok=True)
    with open(out + '.txt', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    np.savez_compressed(out + '.npz', source_id=source_id, parallax=plx, span=span,
                        mass_limit_mjup=np.nan if mass is None else mass,
                        mass_sens_mjup=np.nan if sens_mass is None else sens_mass,
                        e_grid=np.array(e_grid), noise=args.noise, **r)
    j_max = int(np.argmax(r['ln_b']))
    print(f"{source_id}  G {d.get('g_mag', float('nan')):5.2f}  N {len(t):4d}  nu {r['nu']:6.2f} f {r['f']:5.2f}  "
          f"a0 limit median {np.median(r['a0_limit']):.4f} mas "
          f"({r['a0_limit'].min():.4f}-{r['a0_limit'].max():.4f}), at S = {args.s_det:g} "
          f"{np.median(r['a0_sens']):.4f};  largest ln B {r['ln_b'][j_max]:+.1f} at {periods[j_max]:.1f} d"
          + (f";  M limit at {periods[len(periods) // 2]:.0f} d: {mass[len(periods) // 2]:.2f} MJup"
             if mass is not None else ''))

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(periods, r['a0_limit'], color=BLUE, lw=2, label=f"{100 * args.level:.0f}% upper limit")
        ax.plot(periods, r['a0_sens'], color=ORANGE, lw=2, ls='--',
                label=f"detected at $S$ = {args.s_det:g}")
        ax.axvline(span, color='0.6', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.text(span, ax.get_ylim()[0], 'time span ', va='bottom', ha='right', color='0.4',
                fontsize=8, rotation=90)
        ax.set_xlabel('Period [d]')
        ax.set_ylabel(r'$a_0$ [mas]')
        ax.set_title(f"Gaia DR4 {source_id} ($G$ = {d.get('g_mag', float('nan')):.1f})", fontsize=10)
        ax.grid(True, which='both', color='0.92', lw=0.5)
        ax.legend(frameon=False)
        if mass is not None:
            ax2 = ax.twinx()                # same curve in other units: a secondary axis
            ax2.set_yscale('log')
            k = len(periods) // 2
            sc = mass[k] / r['a0_limit'][k]
            ax2.set_ylim(np.array(ax.get_ylim()) * sc)
            ax2.set_ylabel(f"$M$ [M$_{{\\rm Jup}}$] at {periods[k]:.0f} d (approx.)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out + '.png', dpi=150)
        plt.close(fig)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_id', help="Gaia source_id, or 'all'")
    ap.add_argument('--m-star', type=float, default=None, help='host mass [Msun] for mass limits')
    ap.add_argument('--level', type=float, default=0.95)
    ap.add_argument('--s-det', type=float, default=8.0, help='S of the sensitivity curve')
    ap.add_argument('--noise', choices=['student', 'gauss'], default='student')
    ap.add_argument('--e-grid', default='0,0.3,0.6',
                    help="eccentricities marginalised over ('0' for circular orbits)")
    ap.add_argument('--p-min', type=float, default=10.0, help='shortest period [d]')
    ap.add_argument('--p-max-span', type=float, default=2.0,
                    help='longest period in units of the time span')
    ap.add_argument('--n-periods', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'Results', 'upper_limits'))
    args = ap.parse_args()
    if args.source_id == 'all':
        ids = sorted(set(int(v) for v in gaia_dr4_epoch.load_prerelease_table()['source_id']))
    else:
        ids = [int(args.source_id)]
    for sid in ids:
        run(sid, args)
    print(f"\nWritten: {os.path.relpath(args.out)}/<source_id>.txt, .npz"
          + ('' if args.no_plot else ', .png'))


if __name__ == '__main__':
    main()
