#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Effect of the pseudo-colour prior on the projection.

For the three DR4 pre-release companions, with no orbit and with the orbit of
the latest EV-<name>-N1 run (mean model of posterior draws, as in
Tools/noise_diagnostics.py), compare

  projected   ||r - (I - H) s||^2_W, with r the published residuals and H the
              pipeline's projector (which includes the pseudo-colour prior)
  fitted      min_beta ||d - s - D beta||^2_W, a joint fit without the prior

with W the pipeline's weights, i.e. the weights of H. Without the prior the
two are identical; the difference is the effect of the prior. The relative
data-to-prior precision of the pseudo-colour is printed as well, for the
pipeline weights and for the formal errors.

    python3 Tools/projection_prior_check.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Tools'))

import gaia_dr4_epoch       # noqa: E402
import noise_diagnostics    # noqa: E402


def main():
    lines = [f"{'source':>10s} {'model':>8s} {'projected':>12s} {'fitted':>12s} {'diff':>9s} "
             f"{'rel':>8s}   pseudo-colour data/prior precision (pipeline W, formal W)"]
    for sid, name in noise_diagnostics.COMPANIONS.items():
        d = gaia_dr4_epoch.load_source_epoch_data(sid)
        t = np.asarray(d['t_obs_day'], float)
        psi = np.asarray(d['scan_angle_rad'], float)
        r = np.asarray(d['along_scan_mas'], float)
        y0 = np.asarray(d['proj_data'], float)
        X = np.asarray(d['proj_design'], float)
        gain = np.asarray(d['proj_gain'], float)
        W = np.asarray(d['proj_weights'], float)
        sig = np.asarray(d['sigma_mas'], float)
        lam = np.atleast_1d(d['proj_prior_prec'])[-1]
        ratio = [float((X[:, -1] ** 2 * w).sum() / lam) for w in (W, 1.0 / sig ** 2)]
        model, run = noise_diagnostics.orbit_model(sid, t, psi, os.path.join(ROOT, 'Results'),
                                                   'EV-{name}-N1')
        for lab, s in (('no orbit', np.zeros_like(r)), ('orbit', model)):
            if s is None:
                continue
            c_proj = float(np.sum((r - (s - X @ (gain @ s))) ** 2 * W))
            y = y0 - s
            b = np.linalg.lstsq(X * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
            c_fit = float(np.sum((y - X @ b) ** 2 * W))
            lines.append(f"{name:>10s} {lab:>8s} {c_proj:12.6g} {c_fit:12.6g} {c_proj - c_fit:9.2g} "
                         f"{(c_proj - c_fit) / c_fit:8.1e}   {ratio[0]:.3g}, {ratio[1]:.3g}")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(ROOT, 'Results', 'projection_prior_check.txt'), 'w') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main()
