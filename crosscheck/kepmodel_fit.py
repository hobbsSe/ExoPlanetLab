#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Periodogram and maximum-likelihood orbit of DR4 pre-release epoch astrometry
with kepmodel (Delisle & Segransan).

Run with the Python of crosscheck/venv (kepmodel is not in the main environment):

    crosscheck/venv/bin/python crosscheck/kepmodel_fit.py 1457486023639239296

on the file written by crosscheck/export_kima.py. kepmodel is not a sampler:
it gives the periodogram of the residuals of the linear (single-star) model
with an analytic false-alarm probability, and a maximum-likelihood Keplerian
with errors from the Hessian. Gaussian noise: the quoted errors plus a jitter
in quadrature (--jitter, mas; ExoPlanetLab's Gaussian-model jitter of the
source is the like-for-like choice).

Linear terms: position, parallax and proper motion (five, as kima; the
pseudo-colour term is removed by the export). Prints the highest periodogram
peak, its FAP, and the fitted orbit; writes kepmodel_<source_id>.npz to
Results/crosscheck/ for crosscheck/compare_kima.py.
"""

import argparse
import os

import numpy as np
from kepmodel import astro
from spleaf import term

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_id', type=int)
    ap.add_argument('--jitter', type=float, default=0.0, help='white jitter [mas]')
    ap.add_argument('--pmin', type=float, default=10.0, help='shortest period searched [d]')
    ap.add_argument('--pmax', type=float, default=7305.0, help='longest period searched [d]')
    ap.add_argument('--dir', default=os.path.join(ROOT, 'Results', 'crosscheck'))
    args = ap.parse_args()

    t, w, sig, psi, pf = np.loadtxt(os.path.join(args.dir, f'{args.source_id}.gaia'),
                                    skiprows=2, unpack=True)
    o = np.argsort(t, kind='stable')        # S+LEAF needs increasing times
    t, w, sig, psi, pf = t[o], w[o], sig[o], psi[o], pf[o]
    cth, sth = np.cos(psi), np.sin(psi)
    tau = t / 365.25
    kw = dict(err=term.Error(sig))
    if args.jitter > 0:
        kw['jit'] = term.Jitter(args.jitter)
    m = astro.AstroModel(t, w, cth, sth, **kw)
    for d, name in ((sth, 'ra'), (cth, 'dec'), (pf, 'plx'), (tau * sth, 'pmra'),
                    (tau * cth, 'pmdec')):
        m.add_lin(d, name)
    m.fit_lin()
    chi2_0 = m.chi2()

    span = t.max() - t.min()
    nu0, numax = 2 * np.pi / args.pmax, 2 * np.pi / args.pmin
    dnu = 2 * np.pi / span / 10.0
    nfreq = int((numax - nu0) / dnu) + 1
    nu, power = m.periodogram(nu0, dnu, nfreq)
    k = int(np.argmax(power))
    p_peak, fap = 2 * np.pi / nu[k], m.fap(power[k], nu.max())
    print(f"{args.source_id}: {len(t)} points, chi2 of the single-star fit {chi2_0:.1f}")
    print(f"periodogram ({nfreq} frequencies, P {args.pmin:g}-{args.pmax:g} d): highest peak "
          f"P = {p_peak:.2f} d, power {power[k]:.4f}, FAP = {fap:.3g}")

    m.add_keplerian_from_period(p_peak)
    m.fit()
    par = list(m.fit_param)
    val, err = m.get_param_error(par)
    print(f"maximum-likelihood orbit: chi2 {m.chi2():.1f} (Delta chi2 {chi2_0 - m.chi2():.1f}), "
          f"ln L {m.loglike():.2f}")
    for n, v, e in zip(par, val, err):
        print(f"  {n:12s} {v:14.6f} +/- {e:.6f}")
    np.savez_compressed(os.path.join(args.dir, f'kepmodel_{args.source_id}.npz'),
                        names=np.array(par), values=val, errors=err, p_peak=p_peak, fap=fap,
                        power=power[k], chi2_single=chi2_0, chi2_orbit=m.chi2(),
                        jitter=args.jitter, periods=2 * np.pi / nu, powers=power)


if __name__ == '__main__':
    main()
