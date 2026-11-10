#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Exoplanet astrometric orbit parameter estimation — single and multi-planet
# joint inference.
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.
#
# This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
# This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

###############################################################################

"""
Upper limits on undetected companions, as a function of period.

For a source without a detection the question is what companion the data
exclude. At a fixed period P, eccentricity e and periastron time T0 the orbit
is linear in the Thiele-Innes constants, so it can be fitted together with the
single-star terms like any other linear term: the data give the constants
m +/- C (a 4 x 4 covariance, the single-star terms marginalised). No sampling
is needed.

Upper limit. With an isotropic orientation (cos i, Omega, omega uniform) and a
prior uniform in the semi-major axis a0 of the photocentre orbit, an orbit is
a0 u, with u the four constants of a unit orbit of that orientation, and the
likelihood along it is Gaussian in a0,

    L(a0) / L(0) = exp(b a0 - q a0^2 / 2),   q = u' C^-1 u,   b = u' C^-1 m.

The posterior of a0 at that period is the mixture of these truncated Gaussians
over orientations (and over e and the phase, when e > 0), each weighted by its
integral over a0 >= 0 -- its evidence relative to no companion. The limit is
the a0 below which a fraction `level` of that mixture lies. A uniform prior in
a0 is the conventional choice for an upper limit: the log-uniform prior of the
orbit fits would make the limit depend on its lower bound.

Sensitivity. The signal-to-noise ratio of that orbit after the single-star fit
is S = a0 sqrt(q), so a0 = S_det / sqrt(q) is the orbit that would be detected
at S_det (half of the companions are detected at S = 7-8 with ln B > 5).
Its median over orientations is the sensitivity curve: what the data
could have detected, as opposed to what they exclude.

Noise model: 'student' fits t_nu(0, (f sigma)^2) with the single-star terms by
IRLS (robust.StudentTSingleStar), nu and f at their best values without a
companion, and takes m and C from the fit and its Fisher matrix (a Gaussian
approximation in the constants); 'gauss' is weighted least squares with
errors f sigma, f from the reduced chi2 of the single-star fit.
"""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import erfc, erfcx, log_ndtr

import orbit
import robust

# julian year [d]
YEAR = 365.25


def ti_columns(t, psi, P, e=0.0, T0=0.0):
    """(N, 4) columns of the along-scan orbit model for A, B, F, G."""
    t = np.asarray(t, dtype=float)
    E = orbit.kepler_eq(2.0 * np.pi * (t - T0) / P, e)
    X = np.cos(E) - e
    Y = np.sqrt(1.0 - e * e) * np.sin(E)
    cp, sp = np.cos(psi), np.sin(psi)
    return np.column_stack([X * cp, X * sp, Y * cp, Y * sp])


def unit_orbits(n, rng):
    """(n, 4) Thiele-Innes constants of orbits with a0 = 1 and isotropic
    orientation (cos i, Omega, omega uniform)."""
    cosi = rng.uniform(-1.0, 1.0, n)
    Om = rng.uniform(0.0, 2.0 * np.pi, n)
    om = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack(orbit.ti_from_campbell(np.ones(n), om, Om, np.arccos(cosi)))


def noise_fit(d, X, sigma, lam=None, noise='student'):
    """Noise parameters of the single-star model (no companion).

    Returns (nu, f): the maximum-likelihood Student-t parameters, or
    (inf, f) with f^2 the reduced chi2 for noise='gauss'."""
    d, X, sigma = (np.asarray(v, dtype=float) for v in (d, X, sigma))
    if noise == 'gauss':
        lam_ = np.zeros(X.shape[1]) if lam is None else np.asarray(lam, float)
        w = 1.0 / sigma ** 2
        beta = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(lam_), X.T @ (w * d))
        chi2 = float(np.sum(((d - X @ beta) / sigma) ** 2))
        return np.inf, float(np.sqrt(max(chi2 / max(len(d) - X.shape[1], 1), 1e-12)))
    rob = robust.StudentTSingleStar(d, X, sigma, lam=lam)
    zero = np.zeros((1, len(d)))
    lo, hi = np.log([1.0, 0.05]), np.log([200.0, 1.0e3])

    def nll(x):
        if np.any(x < lo) or np.any(x > hi):
            return np.inf
        return -float(rob.loglike(zero, np.exp(x[:1]), np.exp(x[1:]))[0])

    best = min((minimize(nll, np.log([nu0, 1.0]), method='Nelder-Mead',
                         options=dict(xatol=1e-3, fatol=1e-4)) for nu0 in (3.0, 30.0)),
               key=lambda r: r.fun)
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def ti_fit(d, X, sigma, cols, nu, f, lam=None):
    """Thiele-Innes constants m (4,) and their covariance C (4, 4) from the
    linear fit of the single-star terms and the four orbit columns."""
    d, X, sigma, cols = (np.asarray(v, dtype=float) for v in (d, X, sigma, cols))
    p = X.shape[1]
    Xa = np.hstack([X, cols])
    lam_a = np.concatenate([np.zeros(p) if lam is None else np.asarray(lam, float), np.zeros(4)])
    if np.isfinite(nu):
        rob = robust.StudentTSingleStar(d, Xa, sigma, lam=lam_a)
        _, beta, H = rob._terms(np.zeros((1, len(d))), np.array([nu]), np.array([f]))
        beta, H = beta[0], H[0]
    else:
        w = 1.0 / (f * sigma) ** 2
        H = Xa.T @ (Xa * w[:, None]) + np.diag(lam_a)
        beta = np.linalg.solve(H, Xa.T @ (w * d))
    C = np.linalg.inv(H)[p:, p:]
    return beta[p:], C


def _log_weight(q, b):
    """ln of the integral of exp(b a - q a^2 / 2) over a >= 0."""
    z = b / np.sqrt(q)
    return 0.5 * z * z + log_ndtr(z) + 0.5 * np.log(2.0 * np.pi / q)


def _mixture_cdf(a, q, b, w):
    """Posterior probability that a0 < a for the mixture of truncated
    Gaussians (q, b) with weights w (summing to 1)."""
    # P(a0 < a) = 1 - Phi(-(a - mu) / s) / Phi(mu / s), mu = b / q, s = q^-1/2
    z0 = b / np.sqrt(q)
    za = (a * q - b) / np.sqrt(q)
    # erfc ratio, stable in both tails: erfc(x) = erfcx(x) exp(-x^2)
    xa, x0 = za / np.sqrt(2.0), -z0 / np.sqrt(2.0)
    with np.errstate(over='ignore', invalid='ignore'):
        ratio = np.where(
            (xa > 0) & (x0 > 0),
            erfcx(np.maximum(xa, 0)) / erfcx(np.maximum(x0, 0)) * np.exp(-(xa * xa - x0 * x0)),
            erfc(xa) / erfc(x0))
    return float(np.sum(w * (1.0 - np.clip(ratio, 0.0, 1.0))))


def limit_at_period(m_list, C_list, units, level=0.95, s_det=8.0):
    """Upper limit and sensitivity for one period.

    m_list, C_list : the fits (m, C) for each (e, T0) component
    units : (n, 4) isotropic unit orbits
    Returns dict(a0_limit, a0_sens, ln_b, s_best): the a0 below which `level`
    of the posterior lies; the median a0 detected at S = s_det; ln of the
    evidence ratio of a companion at this period (uniform a0 prior, per mas)
    to none; and the largest S of the fitted constants (sqrt of m' C^-1 m)."""
    qs, bs, lw, s_best = [], [], [], 0.0
    for m, C in zip(m_list, C_list):
        Ci = np.linalg.inv(C)
        q = np.einsum('ni,ij,nj->n', units, Ci, units)
        b = units @ (Ci @ m)
        qs.append(q)
        bs.append(b)
        lw.append(_log_weight(q, b))
        s_best = max(s_best, float(np.sqrt(max(m @ Ci @ m, 0.0))))
    q, b, lw = np.concatenate(qs), np.concatenate(bs), np.concatenate(lw)
    ln_b = float(np.logaddexp.reduce(lw) - np.log(len(lw)))
    w = np.exp(lw - lw.max())
    w /= w.sum()
    hi = float(np.max(np.abs(b) / q + 10.0 / np.sqrt(q)))
    a_lim = brentq(lambda a: _mixture_cdf(a, q, b, w) - level, 0.0, hi, xtol=1e-9 * hi)
    return dict(a0_limit=float(a_lim), a0_sens=float(np.median(s_det / np.sqrt(q))),
                ln_b=ln_b, s_best=s_best)


def upper_limits(d, X, sigma, t, psi, periods, lam=None, level=0.95, s_det=8.0,
                 noise='student', e_grid=(0.0,), n_phase=8, n_orient=2000, seed=0):
    """Upper limits on a companion's photocentre semi-major axis against period.

    d : (N,) raw along-scan data [mas] (gaia_dr4_epoch 'proj_data'), X : (N, p)
    single-star design matrix, sigma : (N,) errors [mas], lam : (p,) prior
    precisions of the single-star terms, t : (N,) times [d], psi : (N,) scan
    angles [rad].
    e_grid : eccentricities marginalised over with equal weight (with n_phase
    periastron phases each for e > 0); the default is circular orbits.

    Returns a dict of arrays over `periods`: 'a0_limit' [mas], 'a0_sens' [mas],
    'ln_b', 's_best', and the scalars 'nu', 'f', 'level', 's_det'."""
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    rng = np.random.default_rng(seed)
    units = unit_orbits(n_orient, rng)
    nu, f = noise_fit(d, X, sigma, lam=lam, noise=noise)
    t = np.asarray(t, dtype=float)
    out = {k: np.empty(len(periods)) for k in ('a0_limit', 'a0_sens', 'ln_b', 's_best')}
    for j, P in enumerate(periods):
        ms, Cs = [], []
        for e in e_grid:
            for ph in (np.arange(n_phase) / n_phase if e > 0 else (0.0,)):
                m, C = ti_fit(d, X, sigma, ti_columns(t, psi, P, e, t.min() + ph * P), nu, f,
                              lam=lam)
                ms.append(m)
                Cs.append(C)
        r = limit_at_period(ms, Cs, units, level=level, s_det=s_det)
        for k in out:
            out[k][j] = r[k]
    out.update(periods=periods, nu=nu, f=f, level=level, s_det=s_det)
    return out


def companion_mass(a0_mas, P_days, parallax_mas, m_star_msun):
    """Mass [Msun] of a dark companion whose host's orbit has semi-major axis
    a0 [mas]: m^3 / (M + m)^2 = (a0 / parallax)^3 / P^2 [AU, yr]."""
    a0_mas, P_days = np.broadcast_arrays(np.asarray(a0_mas, float), np.asarray(P_days, float))
    rhs = (a0_mas / parallax_mas) ** 3 / (P_days / YEAR) ** 2
    out = np.empty(rhs.shape)
    for idx in np.ndindex(rhs.shape):
        r = float(rhs[idx])
        out[idx] = brentq(lambda m: m ** 3 / (m_star_msun + m) ** 2 - r, 0.0,
                          max(10.0 * m_star_msun, 10.0 * r + 10.0), xtol=1e-14)
    return out
