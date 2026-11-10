#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Export DR4 pre-release epoch astrometry for kima's GAIAmodel.

kima (v6, GAIAdata / GAIAmodel) reads one row per along-scan measurement,

    t  w  sigw  psi  pf

and models  w = (da + mua tau) sin(psi) + (dd + mud tau) cos(psi) + plx pf
+ the orbit, tau in years from GAIAdata.M0_epoch -- the convention of
ExoPlanetLab's single-star design matrix (checked here: its position columns
are sin(psi) and cos(psi) exactly). The rows written are the ones ExoPlanetLab
fits (gaia_dr4_epoch.compute_al_residuals: used_by_agis_al, after the
single-star fit's own rejections):

    t     days from the reference epoch of the epoch data (J2017.5), as t_obs;
          use M0_epoch = 0 in kima
    w     raw along-scan position [mas] ('proj_data')
    sigw  quoted error [mas]
    psi   scan angle [rad]
    pf    along-scan parallax factor (the parallax column of the design matrix)

kima has five single-star parameters, ExoPlanetLab's DR4-like model a sixth
(pseudo-colour, prior 8.5e-5): with --colour COEFF its term is subtracted from
w (it is 0.004-0.007 mas rms for the three companions, quoted errors 0.08-0.13).
The data are ESA's; the output goes to Results/crosscheck/ (not tracked by git).

Usage:
    python3 crosscheck/export_kima.py 1457486023639239296 --colour=-2.24e-6
"""

import argparse
import os
import sys

import numpy as np

# the main code and data live one level up (this script is in crosscheck/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gaia_dr4_epoch        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_id', type=int, nargs='+')
    ap.add_argument('--colour', type=float, default=0.0,
                    help='pseudo-colour coefficient whose term is subtracted (default 0)')
    ap.add_argument('--out', default=os.path.join('Results', 'crosscheck'))
    args = ap.parse_args()

    df = gaia_dr4_epoch.load_prerelease_table(os.environ.get('EXOPLANETLAB_DR4_PRERELEASE_FILE'))
    os.makedirs(args.out, exist_ok=True)
    for sid in args.source_id:
        ep = gaia_dr4_epoch.compute_al_residuals(df, sid)
        X, psi = ep['proj_design'], ep['scan_angle_rad']
        names = list(ep['proj_param_names'])
        if max(np.max(np.abs(X[:, 0] - np.sin(psi))), np.max(np.abs(X[:, 1] - np.cos(psi)))) > 1e-9:
            sys.exit(f"{sid}: the position columns are not sin / cos of the scan angle")
        w = np.asarray(ep['proj_data'], float)
        if args.colour and 'pseudocolour' in names:
            w = w - args.colour * X[:, names.index('pseudocolour')]
        f = os.path.join(args.out, f'{sid}.gaia')
        with open(f, 'w') as fh:
            fh.write('t\tw\tsigw\tpsi\tpf\n-\t-\t----\t---\t--\n')
            for row in zip(ep['t_obs_day'], w, ep['sigma_mas'], psi, X[:, names.index('parallax')]):
                fh.write('\t'.join(f'{v:.9f}' for v in row) + '\n')
        print(f"{sid}: {len(w)} rows -> {f}"
              + (f"  (colour term {args.colour:g} subtracted)" if args.colour else ''))


if __name__ == '__main__':
    main()
