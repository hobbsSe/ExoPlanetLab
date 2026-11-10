#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
chi2 of the published orbits of Gaia-4 and Gaia BH3, held fixed (no refit),
against the DR4 pre-release epoch residuals, with and without the two
corrections of the model:

  projection   the orbit is passed through the single-star fit,
               s -> (I - H) s, as the published residuals are (Sect. 3.2)
  orientation  the Thiele-Innes constants are put on the correct axes,
               dDec = A X + F Y, dRA* = B X + G Y; the exchanged rows are
               dRA* = A X + F Y, dDec = B X + G Y

The chi2 uses the formal errors. "no orbit" is the chi2 of the residuals
themselves.

    python3 Tools/published_orbit_chi2.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gaia_dr4_epoch   # noqa: E402
import orbit            # noqa: E402

SOURCES = {1457486023639239296: 'Gaia-4', 4318465066420528000: 'Gaia BH3'}


def along_scan(t, psi, P, e, T0, A, B, F, G, exchanged):
    dra, ddec = orbit.astrometric_orbit(t, P, e, T0, A, B, F, G)
    if exchanged:
        dra, ddec = ddec, dra
    return dra * np.sin(psi) + ddec * np.cos(psi)


def main():
    lines = [f"{'source':>9s} {'N':>4s} {'no orbit':>10s} {'neither':>10s} "
             f"{'proj. only':>10s} {'orient. only':>12s} {'both':>10s}"]
    for sid, name in SOURCES.items():
        d = gaia_dr4_epoch.load_source_epoch_data(sid)
        t = np.asarray(d['t_obs_day'], float)
        psi = np.asarray(d['scan_angle_rad'], float)
        r = np.asarray(d['along_scan_mas'], float)
        s2 = np.asarray(d['sigma_mas'], float) ** 2
        X = np.asarray(d['proj_design'], float)
        gain = np.asarray(d['proj_gain'], float)
        truth = gaia_dr4_epoch.get_known_truth(sid)
        chi = {'no orbit': float(np.sum(r ** 2 / s2))}
        for proj in (False, True):
            for exch in (True, False):
                s = along_scan(t, psi, *truth, exchanged=exch)
                if proj:
                    s = s - X @ (gain @ s)
                chi[(proj, exch)] = float(np.sum((r - s) ** 2 / s2))
        lines.append(f"{name:>9s} {len(t):4d} {chi['no orbit']:10.4g} {chi[(False, True)]:10.4g} "
                     f"{chi[(True, True)]:10.4g} {chi[(False, False)]:12.4g} "
                     f"{chi[(True, False)]:10.4g}")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(ROOT, 'Results', 'published_orbit_chi2.txt'), 'w') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main()
