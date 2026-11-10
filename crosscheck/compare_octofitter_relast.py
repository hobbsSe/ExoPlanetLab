#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Compare Octofitter.jl with ExoPlanetLab on the same simulated Gaia epoch
astrometry with relative astrometry of the companion.

Data: crosscheck/export_relast_injection.py. Octofitter chain:
Results/crosscheck/octofitter_relast_<case>_chain.csv (octofitter_relast.jl).
ExoPlanetLab: the run tagged XRA-<case> (Tools/jobs_relast_xcheck.txt).

The scale s = a_rel / a0 is a free parameter of ExoPlanetLab and
(M_A + M_b) / M_b in Octofitter; a_rel = s a0. Angles are folded as in
compare_octofitter.py, and offsets are in units of the combined half-width
of the two 68% intervals.

omega. ExoPlanetLab's omega is that of the photocentre orbit (from its
Thiele-Innes constants), Octofitter's b_omega that of the companion's orbit
about its host, which is the photocentre orbit reversed: at the same Omega
the two differ by 180 deg. Octofitter's omega is therefore shifted by 180 deg
before the comparison, as in compare_octofitter.py.

Usage:  python3 crosscheck/compare_octofitter_relast.py [short long]
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from generate_sources import campbell_elements  # noqa: E402
from compare_octofitter import q, wrap, fold_node  # noqa: E402

RES = os.path.join(ROOT, 'Results')


def ours(case):
    runs = sorted(d for d in glob.glob(os.path.join(RES, f'1plt_emcee_*-XRA-{case}'))
                  if os.path.exists(os.path.join(d, 'posterior_samples.npz')))
    if not runs:
        sys.exit(f"no ExoPlanetLab run XRA-{case} in {RES}")
    d = runs[-1]
    z = np.load(os.path.join(d, 'posterior_samples.npz'))
    n = [str(x) for x in z['names']]
    col = lambda k: z['samples'][:, n.index(k)]
    a0, om, Om, inc = campbell_elements(col('A'), col('B'), col('F'), col('G'))
    Om_f, om_f = fold_node(np.degrees(Om), np.degrees(om))
    out = dict(P=col('P'), e=col('e'), i=np.degrees(inc), Omega=Om_f, omega=om_f, a0=a0,
               s=col('s_rel'), a_rel=col('s_rel') * a0)
    m = re.search(r'^parallax\s*:\s*([\d.]+) \[([\d.]+), ([\d.]+)\]',
                  open(os.path.join(d, 'summary.txt')).read(), re.M)
    plx = (float(m.group(2)), float(m.group(1)), float(m.group(3))) if m else None
    return os.path.basename(d), out, plx


def octo(case):
    path = os.path.join(RES, 'crosscheck', f'octofitter_relast_{case}_chain.csv')
    c = pd.read_csv(path)
    pick = lambda *keys: next(c[k].to_numpy(float) for k in keys if k in c.columns)
    Om_f, om_f = fold_node(np.degrees(pick('b_Ω', 'b_Omega')),
                           np.degrees(pick('b_ω', 'b_omega')))
    return path, dict(P=pick('P_day'), e=pick('b_e'), i=np.degrees(pick('b_i')),
                      Omega=Om_f, omega=(om_f + 180.0) % 360.0, a0=pick('a0_mas'), s=pick('s'),
                      a_rel=pick('a_rel_mas'), plx=pick('plx'))


def main():
    for case in sys.argv[1:] or ['short', 'long']:
        T = json.load(open(os.path.join(RES, 'crosscheck', f'relast_{case}_truth.json')))
        Om_t, om_t = fold_node(T['Omega_deg'], T['omega_deg'])
        truth = dict(P=T['P'], e=T['e'], i=T['inc_deg'], Omega=float(Om_t), omega=float(om_t),
                     a0=T['a0_mas'], s=T['s'], a_rel=T['a_rel_mas'])
        run, E, plx_ours = ours(case)
        path, O = octo(case)
        for k, per in (('Omega', 180.0), ('omega', 360.0)):
            E[k] = wrap(E[k], per, truth[k])
            O[k] = wrap(O[k], per, truth[k])
        print(f"\n=== {case}:  ExoPlanetLab {run}  vs  Octofitter {os.path.basename(path)}"
              f" ({len(O['P'])} draws)")
        print(f"  {'':10s} {'true':>10s} {'ExoPlanetLab':>28s} {'Octofitter':>28s}   offset"
              f"   width O/E")
        fmt = lambda v, nd: f"{v[1]:.{nd}f} +{v[2]-v[1]:.{nd}f} -{v[1]-v[0]:.{nd}f}"
        for k, nd in (('P', 2), ('e', 4), ('i', 2), ('Omega', 2), ('omega', 2), ('a0', 4),
                      ('s', 2), ('a_rel', 2)):
            a, b = q(E[k]), q(O[k])
            w = np.hypot((a[2] - a[0]) / 2, (b[2] - b[0]) / 2)
            print(f"  {k:10s} {truth[k]:10.4f} {fmt(a, nd):>28s} {fmt(b, nd):>28s}   "
                  f"{(b[1] - a[1]) / w:+.2f} sigma   {(b[2] - b[0]) / (a[2] - a[0]):.2f}")
        b = q(O['plx'])
        if plx_ours:
            w = np.hypot((plx_ours[2] - plx_ours[0]) / 2, (b[2] - b[0]) / 2)
            print(f"  {'parallax':10s} {T['plx']:10.4f} {fmt(plx_ours, 4):>28s} {fmt(b, 4):>28s}   "
                  f"{(b[1] - plx_ours[1]) / w:+.2f} sigma   "
                  f"{(b[2] - b[0]) / (plx_ours[2] - plx_ours[0]):.2f}")


if __name__ == '__main__':
    main()
