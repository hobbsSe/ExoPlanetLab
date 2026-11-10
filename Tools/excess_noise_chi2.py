#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Excess noise of the three DR4 pre-release companions, measured with the
formal errors: the chi2/dof of the orbit fit and the chi2/dof of the scatter
within a transit.

For each source the orbit (the mean model of posterior draws of the latest
EV-<name>-N1 run, as in Tools/noise_diagnostics.py) is subtracted and the
single-star terms are refitted by weighted least squares with the formal
errors. Then:

  orbit fit       chi2 / (N - p - 7) of these residuals
  within transit  chi2 / (N - n_transits) of the residuals about their
                  weighted mean in each field-of-view transit (transit_id of
                  the raw table)
  clipped         the orbit-fit chi2/dof without the measurements beyond
                  5 sigma, to show how much of the excess is the heavy tail

If the two first numbers agree, the whole excess of the orbit fit is already
present among the CCD measurements of one transit, so it is not a
per-transit (attitude-like) error.

Note that (rms residual / typical formal error)^2 is not chi2/dof: it ignores
that each residual has its own error, and for HD 114762 it gives 16 against
the correct 9.9.

    python3 Tools/excess_noise_chi2.py
    python3 Tools/excess_noise_chi2.py --tag 'EV-{name}-N1' --clip 5
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Tools'))

import gaia_dr4_epoch       # noqa: E402
import noise_diagnostics    # noqa: E402


def excess(d, model, transit, clip):
    """chi2/dof of the orbit fit, within transits and with |z| > clip removed,
    the number of clipped measurements and the largest |z|."""
    y = np.asarray(d['proj_data'], float) - model
    X = np.asarray(d['proj_design'], float)
    s = np.asarray(d['sigma_mas'], float)
    w = 1.0 / s ** 2
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ beta
    n_par = X.shape[1] + 7
    chi_orbit = float(np.sum(r ** 2 * w) / (len(r) - n_par))
    chi, dof = 0.0, 0
    for i in np.unique(transit):
        k = transit == i
        m = np.sum(w[k] * r[k]) / np.sum(w[k])
        chi += float(np.sum((r[k] - m) ** 2 * w[k]))
        dof += int(k.sum()) - 1
    z = r / s
    keep = np.abs(z) < clip
    chi_clip = float(np.sum(z[keep] ** 2) / (keep.sum() - n_par))
    return chi_orbit, chi / dof, chi_clip, int((~keep).sum()), float(np.abs(z).max()), len(np.unique(transit))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='EV-{name}-N1', help="run tag of the companions' orbits")
    ap.add_argument('--clip', type=float, default=5.0, help='|z| limit of the clipped chi2/dof')
    ap.add_argument('--results', default=os.path.join(ROOT, 'Results'))
    args = ap.parse_args()
    df = gaia_dr4_epoch.load_prerelease_table()
    lines = [f"{'source':>10s} {'transits':>8s} {'orbit fit':>9s} {'within tr.':>10s} "
             f"{'clipped':>8s} {'n clip':>6s} {'max |z|':>7s}  run"]
    for sid, name in noise_diagnostics.COMPANIONS.items():
        d = gaia_dr4_epoch.load_source_epoch_data(sid)
        t = np.asarray(d['t_obs_day'], float)
        psi = np.asarray(d['scan_angle_rad'], float)
        model, run = noise_diagnostics.orbit_model(sid, t, psi, args.results, args.tag)
        if model is None:
            lines.append(f"{name:>10s}  no run with tag {args.tag}")
            continue
        transit = noise_diagnostics.metadata(df, sid, t, psi)['transit']
        c_orb, c_tr, c_clip, n_clip, zmax, n_tr = excess(d, model, transit, args.clip)
        lines.append(f"{name:>10s} {n_tr:8d} {c_orb:9.2f} {c_tr:10.2f} {c_clip:8.2f} "
                     f"{n_clip:6d} {zmax:7.0f}  {run}")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.results, 'excess_noise_chi2.txt'), 'w') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main()
