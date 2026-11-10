#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Write a run_batch.py jobs file for an injection-recovery campaign.

Each job injects one companion into the real observing geometry of a quiet DR4
pre-release source (epoch times, scan angles, per-epoch errors and single-star
projection; its own measurements are not used) and fits it as real data -- see
EXOPLANETLAB_INJECT in ExoPlanetLab.py. The injected orbits are drawn by Latin
hypercube sampling in (log P, e, log S), so a few dozen jobs cover the ranges
evenly, with an isotropic orientation (cos i, Omega, omega uniform) and a
uniform periastron phase:

    P     log-uniform in [P_MIN_FRAC, P_MAX_FRAC] x the geometry's baseline
    e     uniform in [0, E_MAX]
    S     log-uniform in [S_MIN, S_MAX]; S is the total signal-to-noise
          sqrt(sum_i [((I-H)s)_i / sigma_i]^2) of the injected orbit, which sets a0

Usage:
    python3 Tools/make_injection_jobs.py --seed 4 --prefix INJ5   # 60 jobs -> Tools/jobs_INJ5.txt
    python3 Tools/run_batch.py Tools/jobs_INJ5.txt --dry-run
    python3 Tools/run_batch.py Tools/jobs_INJ5.txt
    python3 Tools/injection_summary.py --prefix INJ5              # collect the results

The campaigns INJ2-INJ7 are merged in jobs_injection.txt (one section each, with
the generator settings); append a new campaign's lines there once it has been
run, and run one campaign with run_batch.py --only '^INJ5-'. Use a new seed AND a new
prefix for every campaign (the same seed repeats the same injections).
"""

import argparse
import datetime
import os
import numpy as np

# Quiet DR4 pre-release sources (single-star residual RMS ~ their errors) used
# as observing geometries, with their observation spans in days (from the
# pre-release file).
GEOMETRIES = {
    2309425390592896:    1844.0,   # G 14.0, 597 epochs
    20694084440761600:   1843.0,   # G 14.1, 576 epochs
    435469040545191680:  1739.0,   # G 14.0, 649 epochs
    1663617687609809280: 1902.0,   # G 14.3, 781 epochs
    4181040337841125632: 1733.0,   # G  9.0, 637 epochs
}


def latin_hypercube(n, d, rng):
    """n points in [0, 1)^d, one per stratum in every dimension."""
    u = (np.arange(n)[:, None] + rng.uniform(size=(n, d))) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=60, help='number of jobs (default 60)')
    ap.add_argument('--seed', type=int, default=1, help='seed of the draws (default 1)')
    ap.add_argument('--p-min-frac', type=float, default=0.1)
    ap.add_argument('--p-max-frac', type=float, default=2.0)
    ap.add_argument('--e-max', type=float, default=0.9)
    ap.add_argument('--s-min', type=float, default=5.0)
    ap.add_argument('--s-max', type=float, default=100.0)
    ap.add_argument('--extra-noise', type=float, default=0.0,
                    help='white noise (mas) added to the data but not to the errors')
    ap.add_argument('--solver', default='emcee')
    ap.add_argument('--orient-prior', default='isotropic', choices=['isotropic', 'ti'],
                    help='orientation prior of the fits, written into every job so that '
                         'the campaign does not depend on the current default (default '
                         'isotropic, which matches the isotropic injected orientations)')
    ap.add_argument('--astro-noise', default='student', choices=['student', 'gauss'],
                    help='astrometric noise model of the fits, written into every job '
                         '(default student, as for DR4 data)')
    ap.add_argument('--noise-nu', type=float, default=0.0,
                    help='inject Student-t noise with this nu (default 0 = Gaussian)')
    ap.add_argument('--noise-f', type=float, default=1.0,
                    help='scale of the injected noise in units of the quoted errors')
    ap.add_argument('--pt-astro', type=int, default=1, choices=[0, 1],
                    help='tempered joint fit (1, default) or plain emcee (0), written '
                         'into every job')
    ap.add_argument('--relast', action='store_true',
                    help='add simulated relative astrometry of the companion '
                         '(RELAST_MODE=simulated, with its default scale, error and epochs)')
    ap.add_argument('--prefix', default='INJ', help='job-name prefix (default INJ)')
    ap.add_argument('--out', default=None,
                    help='output jobs file (default Tools/jobs_<prefix>.txt, so that the '
                         'merged jobs_injection.txt is never overwritten)')
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f'jobs_{args.prefix}.txt')
    rng = np.random.default_rng(args.seed)
    u = latin_hypercube(args.n, 3, rng)                    # log P, e, log S
    geoms = list(GEOMETRIES.items())
    lines = [
        f"# Injection-recovery jobs for run_batch.py, written by make_injection_jobs.py",
        f"# on {datetime.datetime.now():%Y-%m-%d %H:%M} with: --n {args.n} --seed {args.seed} "
        f"--p-min-frac {args.p_min_frac} --p-max-frac {args.p_max_frac} --e-max {args.e_max} "
        f"--s-min {args.s_min} --s-max {args.s_max} --extra-noise {args.extra_noise} "
        f"--solver {args.solver} --orient-prior {args.orient_prior} --pt-astro {args.pt_astro} "
        f"--astro-noise {args.astro_noise} --noise-nu {args.noise_nu:g} --noise-f {args.noise_f:g} "
        f"{'--relast ' if args.relast else ''}--prefix {args.prefix}",
        "# INJECT = P[d],e,phase,inc[deg],Omega[deg],omega[deg]; INJECT_SNR = total S.",
        "# Collect the results with: python3 Tools/injection_summary.py",
    ]
    for k in range(args.n):
        sid, span = geoms[k % len(geoms)]
        P = span * args.p_min_frac * (args.p_max_frac / args.p_min_frac) ** u[k, 0]
        e = args.e_max * u[k, 1]
        S = args.s_min * (args.s_max / args.s_min) ** u[k, 2]
        inc = np.degrees(np.arccos(rng.uniform(-1.0, 1.0)))
        Om, om, ph = rng.uniform(0, 180), rng.uniform(0, 360), rng.uniform(0, 1)
        name = f"{args.prefix}-{k + 1:03d}"
        lines.append(
            f"{name:9s} SOURCE_MODE=gaia_dr4_prerelease SOURCE_ID={sid} RV_MODE=none "
            f"SOLVER={args.solver} ORIENT_PRIOR={args.orient_prior} PT_ASTRO={args.pt_astro} "
            f"ASTRO_NOISE={args.astro_noise} "
            f"INJECT={P:.3f},{e:.4f},{ph:.4f},{inc:.2f},{Om:.2f},{om:.2f} "
            f"INJECT_SNR={S:.2f} INJECT_SEED={args.seed * 100000 + k + 1}"
            + (f" INJECT_EXTRA_NOISE={args.extra_noise:g}" if args.extra_noise > 0 else "")
            + (f" INJECT_NOISE_NU={args.noise_nu:g}" if args.noise_nu > 0 else "")
            + (f" INJECT_NOISE_F={args.noise_f:g}" if args.noise_f != 1.0 else "")
            + (" RELAST_MODE=simulated" if args.relast else ""))
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"Wrote {args.n} jobs to {args.out}")


if __name__ == '__main__':
    main()
