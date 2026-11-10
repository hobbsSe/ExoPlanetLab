#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Compare ExoPlanetLab with kima and kepmodel on the same DR4 pre-release data.
Reads

    Results/<run>/posterior_samples.npz        ExoPlanetLab (latest run with the tag)
    Results/crosscheck/kima_<id>/kima_posterior.npz    crosscheck/kima_fit.py
    Results/crosscheck/kepmodel_<id>.npz               crosscheck/kepmodel_fit.py

and prints, per parameter, median and 68% half-width for the two samplers, the
maximum-likelihood value and Hessian error for kepmodel, and the differences
from ExoPlanetLab in units of the combined width. P, e, a0 and the inclination
folded to 0-90 deg (the three codes use different conventions for the node and
the sense of motion); parallax and proper motion.

Usage:
    python3 crosscheck/compare_kima.py                       # the three companions
    python3 crosscheck/compare_kima.py --tag EV-{name}-N1
"""

import argparse
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {'Gaia-4': 1457486023639239296, 'Gaia-BH3': 4318465066420528000,
           'HD-114762': 3937211745905473024}


def campbell(A, B, F, G):
    """a0 and inclination folded to 0-90 deg from Thiele-Innes constants."""
    u = A * A + B * B + F * F + G * G
    v = A * G - B * F
    a2 = 0.5 * (u + np.sqrt(np.maximum(u * u - 4 * v * v, 0.0)))
    return np.sqrt(a2), np.degrees(np.arccos(np.clip(np.abs(v) / a2, 0.0, 1.0)))


def med(x):
    q = np.percentile(x, [16, 50, 84])
    return q[1], 0.5 * (q[2] - q[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='EV-{name}-N1', help="ExoPlanetLab run tag ({name})")
    ap.add_argument('--out', default=os.path.join(ROOT, 'Results', 'crosscheck',
                                                  'compare_kima_kepmodel.txt'))
    args = ap.parse_args()
    cc = os.path.join(ROOT, 'Results', 'crosscheck')
    lines = []
    for name, sid in SOURCES.items():
        runs = sorted(glob.glob(os.path.join(ROOT, 'Results', f'1plt_*-{args.tag.format(name=name)}')),
                      key=os.path.getmtime)
        if not runs:
            continue
        z = np.load(os.path.join(runs[-1], 'posterior_samples.npz'), allow_pickle=True)
        S = z['samples']
        a0, inc = campbell(S[:, 3], S[:, 4], S[:, 5], S[:, 6])
        ours = {'P [d]': S[:, 0], 'e': S[:, 1], 'a0 [mas]': a0, 'i [deg]': inc}
        if 'single_star' in z.files:
            ss = dict(zip(z['single_star_names'], z['single_star'].T))
            ours.update({'parallax [mas]': ss['parallax'], 'pmra [mas/yr]': ss['pmra'],
                         'pmdec [mas/yr]': ss['pmdec']})
        nn = list(z['names'])
        if 'nu_astro' in nn:
            ours['nu'] = S[:, nn.index('nu_astro')]
        cols = {'ExoPlanetLab': {k: med(v) for k, v in ours.items()}}

        f = os.path.join(cc, f'kima_{sid}', 'kima_posterior.npz')
        note = ''
        if os.path.exists(f):
            k = np.load(f, allow_pickle=True)
            ks = dict(zip(k['names'], k['samples'].T))
            cols['kima'] = {k_: med(v) for k_, v in {
                'P [d]': ks['P1'], 'e': ks['ecc1'], 'a0 [mas]': ks['a1'],
                'i [deg]': np.degrees(np.arccos(np.abs(ks['cosi1']))),
                'parallax [mas]': ks['parallax'], 'pmra [mas/yr]': ks['mua'],
                'pmdec [mas/yr]': ks['mud'], 'nu': ks['nu']}.items()}
            note += (f"kima: ln Z {float(k['logz']):.2f}, {k['samples'].shape[0]} samples "
                     f"(ESS {float(k['ess']):.0f}), jitter {med(ks['jitter'])[0]:.3f} mas.  ")
        f = os.path.join(cc, f'kepmodel_{sid}.npz')
        if os.path.exists(f):
            k = np.load(f, allow_pickle=True)
            kv = dict(zip(k['names'], zip(k['values'], k['errors'])))
            rng = np.random.default_rng(0)       # a0, i errors from draws of the TI constants
            ti = [rng.normal(*kv[f'kep.0.TI{c}'], 4000) for c in 'ABFG']
            ka0, kinc = campbell(*ti)
            cols['kepmodel'] = {'P [d]': kv['kep.0.P'], 'e': kv['kep.0.e'], 'a0 [mas]': med(ka0),
                                'i [deg]': med(kinc), 'parallax [mas]': kv['lin.plx'],
                                'pmra [mas/yr]': kv['lin.pmra'], 'pmdec [mas/yr]': kv['lin.pmdec']}
            note += (f"kepmodel: peak {float(k['p_peak']):.2f} d, FAP {float(k['fap']):.2g}, "
                     f"Gaussian, jitter {float(k['jitter']):g} mas.")
        lines.append(f"\n=== {name} ({sid}); ExoPlanetLab run {os.path.basename(runs[-1])}")
        lines.append(f"  {'parameter':15s}" + ''.join(f"{c:>26s}" for c in cols)
                     + ''.join(f"{c + ' - EPL':>16s}" for c in list(cols)[1:]))
        for p in cols['ExoPlanetLab']:
            row = f"  {p:15s}"
            for c in cols:
                row += (f"{cols[c][p][0]:15.4f} +/- {cols[c][p][1]:<7.4f}" if p in cols[c]
                        else f"{'':26s}")
            m0, s0 = cols['ExoPlanetLab'][p]
            for c in list(cols)[1:]:
                row += (f"{(cols[c][p][0] - m0) / np.hypot(s0, cols[c][p][1]):+13.2f} sig"
                        if p in cols[c] else f"{'':16s}")
            lines.append(row)
        lines.append('  ' + note)
    print('\n'.join(lines))
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\nWritten: {os.path.relpath(args.out)}")


if __name__ == '__main__':
    main()
