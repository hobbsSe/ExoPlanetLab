#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Simulated data for the Octofitter cross-check of relative astrometry.

A dark companion of a star with real single-star motion is injected into the
observing geometry of a quiet DR4 pre-release source, exactly as
ExoPlanetLab.py does with EXOPLANETLAB_INJECT / _INJECT_A0 / _INJECT_SS /
_INJECT_SEED (same signal, same random numbers, Gaussian noise at the formal
errors), and relative positions of the companion are simulated at
s = a_rel / a0 with Gaussian errors. The system is physical, which Octofitter
needs: with host mass M1 and parallax plx, a dark companion at scale s has
M2 = M1 / (s - 1), a_rel = plx (M_tot P^2)^(1/3) and a0 = a_rel / s.

Written to Results/crosscheck/ (not tracked by git):
    relast_<case>_epoch_astrometry.csv   Octofitter's DR4 format, one row per
        CCD observation with its own transit_id, so that octofitter_relast.jl
        fits the same points with the same errors as ExoPlanetLab (no
        reduction to transits). epoch = MJD of ExoPlanetLab's time (TCB).
    relast_<case>_rel_astrometry.csv     ExoPlanetLab's RELAST file (radec),
        read by both codes.
    relast_<case>_truth.json             the injected values
and the job line for Tools/run_batch.py is printed.

The cases 'jitter' (quoted imaging errors too small) and 'lum' (a luminous
companion) test EXOPLANETLAB_RELAST_JITTER and _RELAST_FLUX_RATIO.

Usage:  python3 crosscheck/export_relast_injection.py [short long jitter lum]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import gaia_dr4_epoch  # noqa: E402
import hipparcos       # noqa: E402
import orbit           # noqa: E402

OUT_DIR = os.path.join(ROOT, 'Results', 'crosscheck')
SOURCE_ID = 2309425390592896          # geometry only (a quiet source)
M1, PLX, PMRA, PMDEC = 1.0, 20.0, 10.0, -5.0   # Msun, mas, mas/yr
SCALE, SIG_REL, N_REL, SPAN_REL = 50.0, 2.0, 8, 3.0
# case: P [d], e, phase, inc, Omega, omega [deg], noise seed
CASES = {'short': (600.0, 0.30, 0.20, 60.0, 40.0, 100.0, 1),
         'long':  (3000.0, 0.40, 0.55, 120.0, 130.0, 250.0, 2),
         # tests of the imaging jitter and of the flux ratio (not for Octofitter)
         'jitter': (600.0, 0.30, 0.20, 60.0, 40.0, 100.0, 3),
         'lum':    (600.0, 0.30, 0.20, 60.0, 40.0, 100.0, 4)}
# Departures from the defaults above. 'jitter': 20 positions whose quoted
# error (0.5 mas) is a quarter of the true one (2 mas), so a jitter of
# sqrt(2^2 - 0.5^2) = 1.94 mas is missing. 'lum': a luminous companion,
# M2 = 0.3 Msun with flux ratio L2 / L1 = 0.05: beta = r / (1 + r),
# s = 1 / (M2 / M_tot - beta).
OPTIONS = {'jitter': dict(n_rel=20, sig_quoted=0.5),
           'lum': dict(m2=0.3, flux_ratio=0.05)}


def make(case):
    P, e, phase, inc, Om, om, seed = CASES[case]
    d = gaia_dr4_epoch.load_source_epoch_data(SOURCE_ID)
    t = np.asarray(d['t_obs_day'], float)
    sig = np.asarray(d['sigma_mas'], float)
    ang = np.asarray(d['scan_angle_rad'], float)
    names = list(d['proj_param_names'])
    plx_factor = d['proj_design'][:, names.index('parallax')]

    opt = OPTIONS.get(case, {})
    n_rel, sig_quoted = opt.get('n_rel', N_REL), opt.get('sig_quoted', SIG_REL)
    flux_ratio = opt.get('flux_ratio', 0.0)
    if 'm2' in opt:
        m2 = opt['m2']
        scale = 1.0 / (m2 / (M1 + m2) - flux_ratio / (1.0 + flux_ratio))
    else:
        scale, m2 = SCALE, M1 / (SCALE - 1.0)
    a_rel = PLX * ((M1 + m2) * (P / 365.25) ** 2) ** (1.0 / 3.0)
    a0 = a_rel / scale
    T0 = float(t.min() + (phase % 1.0) * P)
    truth = np.array([P, e, T0, *orbit.ti_from_campbell(a0, np.radians(om), np.radians(Om),
                                                        np.radians(inc))])
    # the Gaia data: as the INJECT block of ExoPlanetLab.py
    noise = sig * np.random.default_rng(seed).standard_normal(len(t))
    s = orbit.orbital_model_AL(t, truth, ang, projector=None)
    xi, eta = hipparcos.sky_path(
        d['ra0_deg'], d['dec0_deg'], gaia_dr4_epoch.DR4_EPOCH_REF_JD, 'tcb',
        PLX, PMRA, PMDEC, 0.0, gaia_dr4_epoch.DR4_EPOCH_REF_JD + t, 'tcb',
        d['ra0_deg'], d['dec0_deg'])
    raw = s + xi * np.sin(ang) + eta * np.cos(ang) + PLX * plx_factor + noise
    # the relative positions
    rng = np.random.default_rng(47 + seed)
    t_rel = np.sort(rng.uniform(t.min(), t.min() + SPAN_REL * (t.max() - t.min()), n_rel))
    xr, yr = orbit.relative_orbit_vec(t_rel, truth[None, :], [scale])
    x_rel = xr[0] + SIG_REL * rng.standard_normal(n_rel)
    y_rel = yr[0] + SIG_REL * rng.standard_normal(n_rel)

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f'relast_{case}')
    mjd = gaia_dr4_epoch.DR4_EPOCH_REF_JD - 2400000.5 + t
    with open(base + '_epoch_astrometry.csv', 'w') as fh:
        fh.write(f"# SYNTHETIC along-scan data in the observing geometry of {SOURCE_ID}\n"
                 f"# (crosscheck/export_relast_injection.py, case {case}); one row per CCD.\n")
        fh.write('source_id,transit_id,epoch,scan_pos_angle,parallax_factor_al,centroid_pos_al,'
                 'centroid_pos_error_al,ipd_error_al,used_by_agis_al,outlier_flag\n')
        for k in range(len(t)):
            fh.write(f"{SOURCE_ID},{k + 1},{mjd[k]:.9f},{np.degrees(ang[k]):.9f},"
                     f"{plx_factor[k]:.10g},{raw[k]:.9f},{sig[k]:.8g},{sig[k]:.8g},true,0\n")
    rel_file = base + '_rel_astrometry.csv'
    with open(rel_file, 'w') as fh:
        fh.write(f"# SYNTHETIC relative astrometry (case {case}): s = {scale:g}, "
                 f"true sigma = {SIG_REL:g} mas, quoted {sig_quoted:g} mas\n"
                 f"# TIME_REF_JD = {gaia_dr4_epoch.DR4_EPOCH_REF_JD}\n# FORMAT = radec\n"
                 f"# time(d), dra(mas), ddec(mas), sigma_ra, sigma_dec, rho\n")
        for k in range(n_rel):
            fh.write(f"{t_rel[k]:.6f}, {x_rel[k]:.6f}, {y_rel[k]:.6f}, {sig_quoted:g}, "
                     f"{sig_quoted:g}, 0\n")
    S = float(np.sqrt(np.sum((orbit.orbital_model_AL(
        t, truth, ang, projector=orbit.SingleStarProjector(d['proj_design'], d['proj_gain']))
        / sig) ** 2)))
    json.dump(dict(case=case, source_id=SOURCE_ID, P=P, e=e, T0=T0, inc_deg=inc, Omega_deg=Om,
                   omega_deg=om, a0_mas=a0, a_rel_mas=a_rel, s=scale, M1=M1, M2=m2,
                   flux_ratio=flux_ratio, sigma_rel_true=SIG_REL, sigma_rel_quoted=sig_quoted,
                   plx=PLX, pmra=PMRA, pmdec=PMDEC, seed=seed, snr_total=S, n_obs=len(t),
                   baseline_d=float(t.max() - t.min())),
              open(base + '_truth.json', 'w'), indent=1)
    print(f"# {case}: P = {P:g} d, a0 = {a0:.4f} mas, a_rel = {a_rel:.2f} mas, M2 = {m2:.5f} Msun, "
          f"total S = {S:.0f}, {len(t)} points over {t.max() - t.min():.0f} d, "
          f"median sigma {np.median(sig):.3f} mas")
    print(f"XRA-{case}  SOURCE_MODE=gaia_dr4_prerelease SOURCE_ID={SOURCE_ID} RV_MODE=none "
          f"SOLVER=emcee ASTRO_NOISE=gauss INJECT={P:g},{e:g},{phase:g},{inc:g},{Om:g},{om:g} "
          f"INJECT_A0={a0!r} INJECT_SEED={seed} INJECT_SS={PLX:g},{PMRA:g},{PMDEC:g},0 "
          f"RELAST_MODE=external RELAST_FILE={os.path.relpath(rel_file, ROOT)}"
          + (f" RELAST_FLUX_RATIO={flux_ratio:g}" if flux_ratio else ""))


if __name__ == '__main__':
    for c in sys.argv[1:] or ['short', 'long']:
        make(c)
