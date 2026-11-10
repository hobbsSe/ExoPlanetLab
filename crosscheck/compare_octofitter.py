#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Compare an Octofitter.jl posterior with ExoPlanetLab's for the same source:
astrometry-only fits of the DR4 pre-release epoch astrometry.

Octofitter chain: Results/crosscheck/octofitter_<name>_chain.csv (from
crosscheck/octofitter_fit.jl). ExoPlanetLab: the astrometry-only reference run
of the source in Results/ (posterior_samples.npz and summary.txt).

Conventions. ExoPlanetLab's omega is that of the photocentre orbit: the one
which, with Omega, reproduces the Thiele-Innes constants (generate_sources.
campbell_elements), and with which the RV model has K > 0 at the true
ascending node (Gaia BH3: 77.9, 136.2 deg; published 77.3, 136.2).
Octofitter's b_omega is that of the companion's orbit about its host, the
photocentre orbit reversed: at the same Omega the two differ by 180 deg, and
Octofitter's omega is shifted by 180 deg here. Astrometry alone cannot tell (Omega,
omega) from (Omega + 180, omega + 180), so both codes' draws are folded to
Omega in [0, 180) with omega shifted along. Offsets are in units of the
combined half-width of the two 68% intervals. The jitters are not comparable: Octofitter's is per transit (after
reducing the nine CCDs to one point), ExoPlanetLab's per CCD.

Usage:  python3 crosscheck/compare_octofitter.py gaia4 [gaia_bh3 hd114762]
"""

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from generate_sources import campbell_elements  # noqa: E402

RES = os.path.join(ROOT, 'Results')
RUN_TAG = {'gaia4': 'Gaia-4-NoRV', 'gaia_bh3': 'Gaia-BH3-NoRV', 'hd114762': 'HD-114762-NoRV'}


def q(x):
    return np.percentile(np.asarray(x, float), [16, 50, 84])


def wrap(x, period, centre):
    return centre + ((np.asarray(x, float) - centre + period / 2) % period) - period / 2


def fold_node(Omega_deg, omega_deg):
    """Astrometry alone cannot tell (Omega, omega) from (Omega + 180, omega + 180):
    map every draw to Omega in [0, 180), shifting omega with it."""
    Om = np.asarray(Omega_deg, float) % 360
    om = np.asarray(omega_deg, float)
    flip = Om >= 180
    return np.where(flip, Om - 180, Om), np.where(flip, om + 180, om) % 360


def ours(name):
    runs = sorted(d for d in glob.glob(os.path.join(RES, f'1plt_emcee_*-{RUN_TAG[name]}'))
                  if os.path.exists(os.path.join(d, 'posterior_samples.npz')))
    if not runs:
        sys.exit(f"no ExoPlanetLab run for {RUN_TAG[name]} in {RES}")
    d = runs[-1]
    z = np.load(os.path.join(d, 'posterior_samples.npz'))
    n = [str(x) for x in z['names']]
    S = z['samples']
    col = lambda k: S[:, n.index(k)]
    a0, om, Om, inc = campbell_elements(col('A'), col('B'), col('F'), col('G'))
    Om_f, om_f = fold_node(np.degrees(Om), np.degrees(om))
    out = dict(P=col('P'), e=col('e'), i=np.degrees(inc), Omega=Om_f, omega=om_f, a0=a0)
    if 'jitter_astro' in n:
        out['jitter'] = col('jitter_astro')
    m = re.search(r'^parallax\s*:\s*([\d.]+) \[([\d.]+), ([\d.]+)\]',
                  open(os.path.join(d, 'summary.txt')).read(), re.M)
    plx = (float(m.group(2)), float(m.group(1)), float(m.group(3))) if m else None
    return os.path.basename(d), out, plx


def octo(name):
    import pandas as pd
    path = os.path.join(RES, 'crosscheck', f'octofitter_{name}_chain.csv')
    c = pd.read_csv(path)
    pick = lambda *keys: next(c[k].to_numpy(float) for k in keys if k in c.columns)
    Om_f, om_f = fold_node(np.degrees(pick('b_Ω', 'b_Omega')),
                           np.degrees(pick('b_ω', 'b_omega')))
    return path, dict(P=pick('P_day'), e=pick('b_e'), i=np.degrees(pick('b_i')),
                      Omega=Om_f, omega=(om_f + 180.0) % 360.0,   # companion -> photocentre
                      a0=pick('a0_mas'), plx=pick('plx'),
                      jitter=pick('GaiaDR4_astrometric_jitter'))


def main():
    for name in sys.argv[1:] or ['gaia4']:
        run, E, plx_ours = ours(name)
        path, O = octo(name)
        # put the angles on a common branch, centred on ExoPlanetLab's median
        for k, per in (('Omega', 180.0), ('omega', 360.0)):
            c = np.median(E[k])
            E[k] = wrap(E[k], per, c)
            O[k] = wrap(O[k], per, c)
        if np.median(O['i']) - 90 and np.sign(np.median(O['i']) - 90) != np.sign(np.median(E['i']) - 90):
            print(f"  NOTE: inclinations on opposite sides of 90 deg "
                  f"(Octofitter {np.median(O['i']):.1f}, ExoPlanetLab {np.median(E['i']):.1f})")
        print(f"\n=== {name}:  ExoPlanetLab {run}  vs  Octofitter {os.path.basename(path)}"
              f" ({len(O['P'])} draws)")
        print(f"  {'':10s} {'ExoPlanetLab':>30s} {'Octofitter':>30s}   offset")
        fmt = lambda v, nd: f"{v[1]:.{nd}f} +{v[2]-v[1]:.{nd}f} -{v[1]-v[0]:.{nd}f}"
        for k, nd in (('P', 2), ('e', 4), ('i', 2), ('Omega', 2), ('omega', 2), ('a0', 4)):
            a, b = q(E[k]), q(O[k])
            w = np.hypot((a[2] - a[0]) / 2, (b[2] - b[0]) / 2)
            print(f"  {k:10s} {fmt(a, nd):>30s} {fmt(b, nd):>30s}   {(b[1] - a[1]) / w:+.2f} sigma")
        b = q(O['plx'])
        if plx_ours:
            w = np.hypot((plx_ours[2] - plx_ours[0]) / 2, (b[2] - b[0]) / 2)
            print(f"  {'parallax':10s} {fmt(plx_ours, 4):>30s} {fmt(b, 4):>30s}   "
                  f"{(b[1] - plx_ours[1]) / w:+.2f} sigma")
        if 'jitter' in E:
            print(f"  {'jitter':10s} {fmt(q(E['jitter']), 4):>30s} {fmt(q(O['jitter']), 4):>30s}"
                  f"   (per CCD vs per transit: not comparable)")


if __name__ == '__main__':
    main()
