#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
What would a radial-velocity amplitude coupled to the astrometric orbit give
? For a dark companion the photocentre orbit is the star's, so

    K = 2 pi (a0 / parallax) sin i / (P sqrt(1 - e^2)),

but the joint fit leaves K free. This script imposes the coupling on the
existing joint astrometry + RV fits of the three DR4 pre-release companions by
importance sampling, without a new fit.

For every posterior draw the single-star terms, and with them the parallax,
are drawn from their conditional distribution given the orbit (as the fit does
for the reported parallax), which gives the implied K of the draw. The RV
model is linear in K and in the velocity offsets, which have uniform priors,
so with the offsets integrated out the RV likelihood of a draw is Gaussian in
K, N(K; K_hat, s_K^2), and the weight of the draw under the coupled model is
N(K_implied; K_hat, s_K^2). (Holding the offsets at their drawn values instead
is also exact but useless: K and the offsets are strongly correlated and the
weights collapse.) The mean weight gives the evidence ratio,

    Z(coupled) / Z(free) = (K_max - K_min) / 2 * < N(K_implied; K_hat, s_K^2) >,

the factor 1/2 being the choice of node, which the sign of the free K makes.

    python3 Tools/coupled_k.py                 # the runs of the default --batch stamp
    python3 Tools/coupled_k.py --batch 20260926_1003
    python3 Tools/coupled_k.py --batch 20261105_1257 --name 'EVRV-{tag}-N1'

Writes Results/coupled_k.txt.
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gaia_dr4_epoch                               # noqa: E402
import mass_estimate                                # noqa: E402
import orbit                                        # noqa: E402
import robust                                       # noqa: E402
from generate_sources import campbell_elements      # noqa: E402

SOURCES = {'Gaia-BH3': 4318465066420528000, 'Gaia-4': 1457486023639239296,
           'HD-114762': 3937211745905473024}
AU_KM, DAY_S, MJUP_PER_MSUN = 1.495978707e8, 86400.0, 1047.57


def read_rv(source_id):
    """Times (days from the DR4 reference epoch), RVs, errors, instrument
    index, names, and the header's star values."""
    path = os.path.join(ROOT, 'RVdata', str(source_id), 'rv_data.csv')
    rows, head = [], {}
    with open(path) as fh:
        for line in fh:
            raw = line.strip()
            if raw.startswith('#'):
                kv = raw.lstrip('#').split('=')
                if len(kv) == 2:
                    try:
                        head[kv[0].strip()] = float(kv[1].strip())
                    except ValueError:
                        pass
                continue
            f = [x.strip() for x in line.split('#', 1)[0].split(',')]
            if len(f) >= 4:
                rows.append((float(f[0]), float(f[1]), float(f[2]), f[3]))
    names = sorted(set(r[3] for r in rows))
    t = np.array([r[0] for r in rows]) + head['TIME_REF_JD'] - gaia_dr4_epoch.DR4_EPOCH_REF_JD
    return (t, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
            np.array([names.index(r[3]) for r in rows]), names, head)


def wq(x, w, q=(0.16, 0.5, 0.84)):
    """Weighted quantiles."""
    o = np.argsort(x)
    c = np.cumsum(w[o]) / np.sum(w)
    return np.interp(q, c, x[o])


def fmt(v, nd):
    lo, md, hi = v
    return f"{md:.{nd}f} +{hi - md:.{nd}f} -{md - lo:.{nd}f}"


def coupled(tag, source_id, batch, results, rng, chunk=2000, name='{tag}-RV'):
    run_name = name.format(tag=tag)
    runs = sorted(glob.glob(os.path.join(results, f'1plt_emcee_{batch}*-{run_name}')))
    if not runs:
        return [f"{tag}: no run 1plt_emcee_{batch}*-{run_name}"]
    z = np.load(os.path.join(runs[-1], 'posterior_samples.npz'), allow_pickle=True)
    names = [str(x) for x in z['names']]
    S = np.asarray(z['samples'], float)
    col = lambda k: S[:, names.index(k)]                              # noqa: E731
    P, e, T0, A, B, F, G, K = (col(k) for k in ('P', 'e', 'T0', 'A', 'B', 'F', 'G', 'K_p1'))
    a0, _, _, inc = campbell_elements(A, B, F, G)

    # parallax of every draw: the robust single-star fit with the orbit removed
    d = gaia_dr4_epoch.load_source_epoch_data(source_id)
    t, psi = np.asarray(d['t_obs_day'], float), np.asarray(d['scan_angle_rad'], float)
    rob = robust.StudentTSingleStar(np.asarray(d['proj_data'], float), np.asarray(d['proj_design'], float),
                                    np.asarray(d['sigma_mas'], float), lam=d.get('proj_prior_prec'))
    j = list(d['proj_param_names']).index('parallax')
    plx = np.concatenate([rob.beta_draws(orbit.orbital_model_AL_vec(t, S[k:k + chunk, :7], psi),
                                         col('nu_astro')[k:k + chunk], col('f_astro')[k:k + chunk], rng)[:, j]
                          for k in range(0, len(S), chunk)])
    K_imp = (2.0 * np.pi * a0 / plx * AU_KM * np.sin(inc) / (P * DAY_S * np.sqrt(1.0 - e ** 2))) * 1e3

    # the RV likelihood as a Gaussian in K: offsets integrated out, jitters as drawn
    t_rv, rv, rv_err, inst, inst_names, head = read_rv(source_id)
    wt = 1.0 / (rv_err[None, :] ** 2
                + np.column_stack([col('jitter_' + n) for n in inst_names])[:, inst] ** 2)
    shape = orbit.rv_model_vec(t_rv, P, e, T0, np.ones(len(S)), orbit.omega_from_ti(A, B, F, G))
    y = np.broadcast_to(rv, shape.shape).copy()
    for k in range(len(inst_names)):                                   # weighted mean per instrument
        m = inst == k
        sw = np.sum(wt[:, m], axis=1, keepdims=True)
        shape[:, m] -= np.sum(wt[:, m] * shape[:, m], axis=1, keepdims=True) / sw
        y[:, m] -= np.sum(wt[:, m] * y[:, m], axis=1, keepdims=True) / sw
    Af = np.sum(wt * shape ** 2, axis=1)
    K_hat, s_K = np.sum(wt * shape * y, axis=1) / Af, 1.0 / np.sqrt(Af)
    # the node the RVs choose: the sign of K_hat (not of the stored K, whose
    # sign follows the omega convention of the run)
    K_c = np.sign(K_hat) * K_imp
    lnw = -0.5 * ((K_c - K_hat) / s_K) ** 2 - np.log(s_K * np.sqrt(2.0 * np.pi))
    w = np.exp(lnw - lnw.max())
    ess = np.sum(w) ** 2 / np.sum(w ** 2)
    K_max = max(5000.0, float(rv.max() - rv.min()))
    lnB = np.log(np.mean(w)) + lnw.max() + np.log(2.0 * K_max / 2.0)
    # its Monte Carlo error from the scatter of the weights
    lnB_err = np.std(w) / np.mean(w) / np.sqrt(len(w))

    m_star = rng.normal(head['M_STAR_MSUN'], head['M_STAR_SIGMA'], len(S))
    unit, sc = ('Msun', 1.0) if tag == 'Gaia-BH3' else ('Mjup', MJUP_PER_MSUN)
    mass = lambda Kv: sc * np.array([mass_estimate.rv_planet_mass(*v)         # noqa: E731
                                     for v in zip(P, Kv, e, inc, m_star)])
    m_free, m_coup = mass(K), mass(K_c)

    one = np.ones(len(S))
    L = [f"=== {tag}  ({os.path.basename(runs[-1])}, {len(S)} draws)",
         f"  fitted K {fmt(wq(np.abs(K), one), 1)} m/s;  implied K {fmt(wq(K_imp, one), 1)} m/s;  "
         f"width of the RV likelihood in K at fixed orbit {np.median(s_K):.1f} m/s",
         f"  effective sample size under the coupling: {ess:.0f} of {len(S)}",
         f"  ln [Z(coupled) / Z(free)] = {lnB:+.2f} +/- {lnB_err:.2f}   (K prior +/-{K_max:.0f} m/s)",
         f"  {'':10s} {'K free':>28s} {'K coupled':>28s} {'shift / sigma':>13s} {'width ratio':>11s}"]
    for lab, v, nd in (('P [d]', P, 4 if tag == 'HD-114762' else 1), ('e', e, 4), ('a0 [mas]', a0, 4),
                       ('i [deg]', np.degrees(inc), 3), ('plx [mas]', plx, 4), ('K [m/s]', np.abs(K), 1)):
        q0 = wq(v, one)
        q1 = wq(np.abs(K_c) if lab.startswith('K') else v, w)
        s0, s1 = (q0[2] - q0[0]) / 2.0, (q1[2] - q1[0]) / 2.0
        L.append(f"  {lab:10s} {fmt(q0, nd):>28s} {fmt(q1, nd):>28s} {(q1[1] - q0[1]) / s0:+13.2f} {s1 / s0:11.2f}")
    q0, q1 = wq(m_free, one), wq(m_coup, w)
    L.append(f"  {'M2 [' + unit + ']':10s} {fmt(q0, 2):>28s} {fmt(q1, 2):>28s} "
             f"{(q1[1] - q0[1]) / ((q0[2] - q0[0]) / 2):+13.2f} {(q1[2] - q1[0]) / (q0[2] - q0[0]):11.2f}")
    L.append(f"  (M_star {head['M_STAR_MSUN']} +/- {head['M_STAR_SIGMA']} Msun)")
    return L + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', default='20260926_1003', help='time stamp of the joint runs')
    ap.add_argument('--name', default='{tag}-RV',
                    help="run name after the time stamp, e.g. 'EVRV-{tag}-N1'")
    ap.add_argument('--results', default=os.path.join(ROOT, 'Results'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    L = ["K coupled to the astrometric orbit, by importance sampling of the joint fits",
         "(Tools/coupled_k.py). Medians and 68% intervals.", ""]
    for tag, sid in SOURCES.items():
        L += coupled(tag, sid, args.batch, args.results, rng, name=args.name)
    text = '\n'.join(L)
    print(text)
    out = os.path.join(args.results, 'coupled_k.txt')
    with open(out, 'w') as fh:
        fh.write(text + '\n')
    print(f"Written: {os.path.relpath(out)}")


if __name__ == '__main__':
    main()
