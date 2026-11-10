#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Fit DR4 pre-release epoch astrometry with kima's GAIAmodel.

Run with the Python of crosscheck/venv (kima is not in the main environment):

    crosscheck/venv/bin/python crosscheck/kima_fit.py 1457486023639239296 --steps 50000

on the file written by crosscheck/export_kima.py. One Keplerian, the
single-star terms (five: kima has no pseudo-colour), Student-t likelihood with
an additive jitter, diffusive nested sampling (DNest4). Priors chosen to match
ExoPlanetLab's:

    P      log-uniform 1-7305 d          e      uniform 0-1
    a0     log-uniform 1e-3-1000 mas     cos i  uniform -1..1 (isotropic)
    Omega, omega, mean anomaly at the reference epoch: uniform
    nu     log-uniform 1-100             jitter log-uniform 1e-3-10 mas
    position, parallax, proper motion: uniform, wide, centred on a
    least-squares single-star fit; or, with --baseline kepmodel, uniform
    within +/-30 sigma of kepmodel's maximum-likelihood values (crosscheck/
    kepmodel_fit.py) -- still far wider than the posterior, but narrow enough
    for DNest4 to reach a strong signal: with the wide priors it ended with
    1-16 effective samples for Gaia BH3 and HD 114762 after 400 000 steps

Differences that remain: kima's noise is t_nu with sigma^2 + jitter^2,
ExoPlanetLab's t_nu with (f sigma)^2; kima samples the single-star terms,
ExoPlanetLab fits them for every evaluation.

Output in Results/crosscheck/kima_<source_id>/: kima's own files and
kima_posterior.npz (names, samples, logz) for crosscheck/compare_kima.py.
"""

import argparse
import os

import numpy as np

import kima
from kima.distributions import LogUniform, Uniform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_id', type=int)
    ap.add_argument('--steps', type=int, default=50000, help='DNest4 saved samples')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--baseline', choices=['lsq', 'kepmodel'], default='lsq',
                    help='centre and width of the single-star priors (see above)')
    ap.add_argument('--dir', default=os.path.join(ROOT, 'Results', 'crosscheck'))
    args = ap.parse_args()

    datafile = os.path.join(args.dir, f'{args.source_id}.gaia')
    work = os.path.join(args.dir, f'kima_{args.source_id}')
    os.makedirs(work, exist_ok=True)
    os.chdir(work)                      # kima writes its files to the current directory

    d = kima.GAIAdata(datafile, skip=2)
    d.M0_epoch = 0.0                    # t is in days from the reference epoch (J2017.5)
    t, w, s, psi, pf = (np.array(getattr(d, k)) for k in ('t', 'w', 'wsig', 'psi', 'pf'))
    tau = t / 365.25
    X = np.column_stack([np.sin(psi), np.cos(psi), pf, tau * np.sin(psi), tau * np.cos(psi)])
    b = np.linalg.lstsq(X / s[:, None], w / s, rcond=None)[0]

    m = kima.GAIAmodel(True, 1, d)
    m.studentt = True
    if args.baseline == 'kepmodel':
        k = np.load(os.path.join(args.dir, f'kepmodel_{args.source_id}.npz'), allow_pickle=True)
        kv = dict(zip(k['names'], zip(k['values'], k['errors'])))
        pri = [Uniform(kv[n][0] - 30 * kv[n][1], kv[n][0] + 30 * kv[n][1])
               for n in ('lin.ra', 'lin.dec', 'lin.plx', 'lin.pmra', 'lin.pmdec')]
        m.da_prior, m.dd_prior, m.parallax_prior, m.mua_prior, m.mud_prior = pri
    else:
        m.da_prior, m.dd_prior = Uniform(b[0] - 100, b[0] + 100), Uniform(b[1] - 100, b[1] + 100)
        m.parallax_prior = Uniform(b[2] - 20, b[2] + 20)
        m.mua_prior, m.mud_prior = Uniform(b[3] - 50, b[3] + 50), Uniform(b[4] - 50, b[4] + 50)
    m.Jprior = LogUniform(1e-3, 10.0)
    m.nu_prior = LogUniform(1.0, 100.0)
    c = m.conditional
    c.Pprior = LogUniform(1.0, 7305.0)
    c.eprior = Uniform(0, 1)
    c.aprior = LogUniform(1e-3, 1000.0)

    kima.run(m, steps=args.steps, num_threads=args.threads)
    res = kima.load_results(m)
    with open('posterior_sample.txt') as fh:
        names = fh.readline().lstrip('#').split()
    samples = np.atleast_2d(np.loadtxt('posterior_sample.txt'))
    np.savez_compressed('kima_posterior.npz', names=np.array(names), samples=samples,
                        logz=float(res.evidence), ess=float(res.ESS),
                        lsq=b, kima_version=kima.__version__)
    print(f"kima {kima.__version__}: ln Z = {res.evidence:.2f}, ESS = {res.ESS}, "
          f"{samples.shape[0]} posterior samples -> {os.path.join(work, 'kima_posterior.npz')}")


if __name__ == '__main__':
    main()
