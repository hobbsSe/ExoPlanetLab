#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Single star vs acceleration (7- / 9-parameter) on DR4 epoch astrometry
(accel.py). No sampling: the three evidences are direct integrals,
seconds to a minute per source.

For every source: ln Z of the single-star model and ln B of 'accel7'
(constant acceleration), 'accel9' (acceleration + jerk), 'harm2' and 'harm23'
(scan-angle harmonics k = 2 and k = 2, 3) over it, the fitted terms
with a flat prior, and their significance. The orbit evidences of
Tools/evidence_ladder.py (EXOPLANETLAB_EVIDENCE=1 runs) have the same
normalisation and compare directly.

Usage
-----
    python3 Tools/acceleration_test.py                    # all pre-release sources
    python3 Tools/acceleration_test.py 4318465066420528000
    # synthetic data in the geometry of a source (its measurements are not used):
    python3 Tools/acceleration_test.py 4181040337841125632 --inject-accel 0.05,-0.02
    python3 Tools/acceleration_test.py 4181040337841125632 \\
        --inject-orbit 6000,0.3,0.4,60,30,90,2.0 --noise-nu 3 --seed 1

--inject-orbit is "P,e,phase,inc_deg,Omega_deg,omega_deg,a0_mas" (as
EXOPLANETLAB_INJECT, with a0); --inject-accel "accel_ra,accel_dec[,jerk_ra,
jerk_dec]" in mas/yr^2 (mas/yr^3); --inject-harm "c2,s2[,c3,s3]" in mas.
"""

import argparse
import os
import sys
import time

import numpy as np

# the main code and data live one level up (this script is in Tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import accel                 # noqa: E402
import gaia_dr4_epoch        # noqa: E402
import orbit                 # noqa: E402
import robust                # noqa: E402

NU_LIM, F_LIM = (1.0, 100.0), (0.1, 10.0)      # as ExoPlanetLab.py (NU_MIN, ...)


def run_source(df, source_id, args):
    ep = gaia_dr4_epoch.compute_al_residuals(df, source_id)
    X, sig = ep['proj_design'], np.asarray(ep['sigma_mas'], float)
    names = list(ep['proj_param_names'])
    ipa, ipd = names.index('pmra'), names.index('pmdec')
    tau = np.asarray(ep['t_obs_day'], float) / 365.25
    d, what = ep['proj_data'], 'real data'
    if args.inject_orbit or args.inject_accel or args.inject_harm:
        rng = np.random.default_rng(args.seed)
        eps = (rng.standard_t(args.noise_nu, len(sig)) if args.noise_nu > 0
               else rng.standard_normal(len(sig)))
        d, what = sig * eps, 'SYNTHETIC noise' + (f' (Student-t nu {args.noise_nu:g})'
                                                  if args.noise_nu > 0 else ' (Gaussian)')
        if args.inject_orbit:
            P, e, ph, inc, Om, om, a0 = (float(v) for v in args.inject_orbit.split(','))
            t = ep['t_obs_day']
            par = np.array([P, e, t.min() + (ph % 1.0) * P,
                            *orbit.ti_from_campbell(a0, np.radians(om), np.radians(Om),
                                                    np.radians(inc))])
            d = d + orbit.orbital_model_AL(t, par, ep['scan_angle_rad'], projector=None)
            what += f' + orbit P {P:g} d, e {e:g}, a0 {a0:g} mas'
        if args.inject_accel:
            v = [float(x) for x in args.inject_accel.split(',')]
            v = v + [0.0] * (4 - len(v))
            pm = X[:, [ipa, ipd]]
            d = d + (pm * tau[:, None] / 2.0) @ v[:2] + (pm * tau[:, None] ** 2 / 6.0) @ v[2:]
            what += f' + acceleration {v[:2]} mas/yr^2' + (f', jerk {v[2:]} mas/yr^3'
                                                           if any(v[2:]) else '')
        if args.inject_harm:
            v = [float(x) for x in args.inject_harm.split(',')]
            v = v + [0.0] * (4 - len(v))
            psi = ep['scan_angle_rad']
            d = d + (v[0] * np.cos(2 * psi) + v[1] * np.sin(2 * psi)
                     + v[2] * np.cos(3 * psi) + v[3] * np.sin(3 * psi))
            what += f' + scan-angle harmonics {v} mas'
    rob = robust.StudentTSingleStar(d, X, sig, ep['proj_prior_prec'])
    t0 = time.time()
    lz0 = rob.log_evidence(np.zeros(len(d)), NU_LIM, F_LIM)
    out = [f"\n=== {source_id}  G = {ep['g_mag']:.2f}, {len(d)} points over "
           f"{tau.max() - tau.min():.2f} yr; {what}",
           f"  single star   ln Z = {lz0:.2f}"]
    for model in accel.MODELS:
        lz = accel.log_evidence(rob, tau, ipa, ipd, model, NU_LIM, F_LIM, psi=ep['scan_angle_rad'])
        est = accel.estimate(rob, tau, ipa, ipd, model, NU_LIM, F_LIM, psi=ep['scan_angle_rad'])
        out.append(f"  {model:13s} ln Z = {lz:.2f}   ln B vs single = {lz - lz0:+.2f}   "
                   f"significance {est['significance']:.1f}  (nu {est['nu']:.2f}, f {est['f']:.2f})")
        for n, u, v, s in zip(est['names'], est['units'], est['value'], est['error']):
            out.append(f"      {n:10s} = {v:+9.4f} +/- {s:.4f} {u}")
    out.append(f"  ({time.time() - t0:.1f} s)")
    print('\n'.join(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_id', nargs='*', type=int,
                    help='Gaia source ids (default: every source of the pre-release file)')
    ap.add_argument('--inject-orbit', default='', help='"P,e,phase,inc,Omega,omega,a0"')
    ap.add_argument('--inject-accel', default='', help='"ara,adec[,jra,jdec]"')
    ap.add_argument('--inject-harm', default='', help='"c2,s2[,c3,s3]" (mas)')
    ap.add_argument('--noise-nu', type=float, default=0.0,
                    help='Student-t noise of the synthetic data (default 0: Gaussian)')
    ap.add_argument('--seed', type=int, default=0, help='noise seed of the synthetic data')
    args = ap.parse_args()

    df = gaia_dr4_epoch.load_prerelease_table(os.environ.get('EXOPLANETLAB_DR4_PRERELEASE_FILE'))
    ids = args.source_id or list(gaia_dr4_epoch.list_prerelease_sources(df))
    print(f"Acceleration / scan-angle test: scales log-uniform on [{accel.ACCEL_MIN:g}, "
          f"{accel.ACCEL_MAX:g}] mas/yr^2, [{accel.JERK_MIN:g}, {accel.JERK_MAX:g}] mas/yr^3, "
          f"[{accel.HARM_MIN:g}, {accel.HARM_MAX:g}] mas")
    for sid in ids:
        run_source(df, int(sid), args)


if __name__ == '__main__':
    main()
