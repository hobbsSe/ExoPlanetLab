# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Acceleration models for Gaia epoch astrometry.

Why
---
A companion with a period much longer than the data shows only a curved
stretch of its orbit: the star's proper motion changes at a constant rate (an
acceleration), or at a linearly changing rate (acceleration plus jerk). These
are the 7- and 9-parameter solutions of the Gaia non-single-star processing.
They are the simpler hypotheses an orbit has to beat when P >> baseline, and a
detection in their own right when no orbit can be determined.

Model
-----
The single-star model of robust.StudentTSingleStar with extra linear terms:

    position += a tau^2 / 2 [+ j tau^3 / 6],  tau in years from the reference
                                               epoch of the epoch data

for both coordinates: a = (accel_ra, accel_dec) in mas/yr^2 ('accel7'), and
j = (jerk_ra, jerk_dec) in mas/yr^3 as well ('accel9'). The design columns are
the proper-motion columns times tau / 2 and tau^2 / 6.

Prior and evidence
------------------
The evidence needs a proper prior on the new terms. Each pair has a Gaussian
prior N(0, sigma_p^2) per component with the scale sigma_p log-uniform over
[ACCEL_MIN, ACCEL_MAX] (and [JERK_MIN, JERK_MAX]): isotropic in direction and
close to log-uniform in size, like the a0 prior of an orbit. For given nu, f
and sigma_p the Gaussian prior integrates analytically (in the same Laplace
approximation as the single-star terms): with b the fitted terms and C their
covariance under a flat prior,

    ln Z(nu, f, sigma_p) = ln L_flat(nu, f) - ln det(C + S) / 2
                           - b'(C + S)^-1 b / 2,     S = sigma_p^2 I

where ln L_flat is StudentTSingleStar.loglike with the extra columns (the
likelihood with the new terms under a flat prior). The last two terms,
together with the volume ln det(C) / 2 contained in ln L_flat (GaussAccel adds
it explicitly), are the price of the prior: the quadratic term penalises
fitted terms that are large compared with the width of C + S, and the
determinant term, combined with that volume, gives ln det[C (C + S)^-1] / 2,
about ln(sigma_b / sigma_p) per component that the data measure well
(sigma_b << sigma_p its uncertainty) and nothing for a component that they do
not constrain (an Occam penalty). The
scales are integrated on a grid here (the term is StudentTSingleStar's
'extra', so it is part of ln L wherever the model is used), nu and f by
StudentTSingleStar.log_evidence. ln Z has the normalisation of the N = 0
evidence of the ladder and of the tempered orbit fits, so the
single-star, acceleration and orbit models compare directly. As sigma_p -> 0
the model is the single star, so the log Bayes factor ln B(acceleration vs
single) = ln Z(acceleration) - ln Z(single) cannot be strongly
negative (about -ln of the number of decades in which sigma_p is undetectable): the
comparison can favour an acceleration, not argue against one.

In an orbit fit
---------------
EXOPLANETLAB_ASTRO_ACCEL=7|9 uses the same augmented model, with the prior
term, as the single-star part of the orbit likelihood: an orbit plus the
curvature of a wider companion. Its ln Z is then comparable with the orbit
without the terms and with the acceleration models without an orbit.

Gaussian noise model
--------------------
With EXOPLANETLAB_ASTRO_NOISE=gauss the data are single-star residuals; the
same terms and prior, with the columns projected like the orbit (GaussAccel,
at the end of this module). Exact there: the likelihood is Gaussian in them.

Scan-angle harmonics
------------------------------
The same machinery tests for spurious, instrumental or source-structure
signals: an along-scan offset that depends on the direction of the scan, the
scan angle psi, but not on time, sum_n [u_n cos(n psi) + v_n sin(n psi)] (mas),
with the amplitudes (u_n, v_n) fitted directly and n = 2 ('harm2') or n = 2
and 3 ('harm23'). The n = 1 harmonic, u_1 cos psi + v_1 sin psi, is a fixed
offset in delta and alpha*, already in the single-star model. These models are
an alternative to an orbit or an acceleration, not a reparametrisation: unlike
'accel7' / 'accel9', whose fitted terms are the acceleration and jerk
themselves, they have no time dependence, and (u_n, v_n) say nothing about a
or j. Gaia's scanning law ties psi to time (the 63-day precession), so such a
signal can mimic an orbit; if 'harm2' / 'harm23' explain the data as well as
an orbit, the orbit is not to be trusted, and with the terms inside the orbit
fit (EXOPLANETLAB_ASTRO_HARM) a spurious orbit loses its evidence. Each
(u_n, v_n) pair has the Gaussian scale-mixture prior above, scale log-uniform
on [EXOPLANETLAB_HARM_MIN, _HARM_MAX] mas (isotropic in phase).
"""

import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import robust

# Prior ranges of the scales [mas/yr^2, mas/yr^3]. An orbit of semi-major axis
# a0 and period P accelerates the star by up to a0 (2 pi / P)^2: 0.1-1.6 a0 per
# yr^2 for P = 20-5 yr, so the range covers the a0 range of the orbit prior
# (from ISO_A0_MIN = 1e-3 mas) for the periods an acceleration stands for.
ACCEL_MIN = float(os.environ.get('EXOPLANETLAB_ACCEL_MIN', '1e-3'))
ACCEL_MAX = float(os.environ.get('EXOPLANETLAB_ACCEL_MAX', '1e2'))
JERK_MIN = float(os.environ.get('EXOPLANETLAB_JERK_MIN', '1e-3'))
JERK_MAX = float(os.environ.get('EXOPLANETLAB_JERK_MAX', '1e2'))
N_SCALE = 48         # grid points per scale (spacing 0.24 in ln s; the integrand
                     # is smooth on a scale of ~1)

HARM_MIN = float(os.environ.get('EXOPLANETLAB_HARM_MIN', '1e-3'))      # mas
HARM_MAX = float(os.environ.get('EXOPLANETLAB_HARM_MAX', '1e2'))

# name -> number of (2-column) term pairs; their names, units and scale ranges
MODELS = {'accel7': 1, 'accel9': 2, 'harm2': 1, 'harm23': 2}
TERM_NAMES = ['accel_ra', 'accel_dec', 'jerk_ra', 'jerk_dec']
TERM_UNITS = ['mas/yr^2', 'mas/yr^2', 'mas/yr^3', 'mas/yr^3']
HARMONICS = {'harm2': (2,), 'harm23': (2, 3)}
NAMES = {'accel7': TERM_NAMES[:2], 'accel9': TERM_NAMES,
         'harm2': ['harm2_cos', 'harm2_sin'],
         'harm23': ['harm2_cos', 'harm2_sin', 'harm3_cos', 'harm3_sin']}
UNITS = {'accel7': TERM_UNITS[:2], 'accel9': TERM_UNITS,
         'harm2': ['mas'] * 2, 'harm23': ['mas'] * 4}
SCALES = {'accel7': [(ACCEL_MIN, ACCEL_MAX)],
          'accel9': [(ACCEL_MIN, ACCEL_MAX), (JERK_MIN, JERK_MAX)],
          'harm2': [(HARM_MIN, HARM_MAX)], 'harm23': [(HARM_MIN, HARM_MAX)] * 2}
DESCRIPTION = {'accel7': 'acceleration', 'accel9': 'acceleration + jerk',
               'harm2': 'scan-angle harmonic k = 2', 'harm23': 'scan-angle harmonics k = 2, 3'}


def columns(model, X, tau_yr, ipa, ipd, psi=None):
    """The extra design columns of a model, (N, 2 x pairs): the proper-motion
    columns of X times tau / 2 (and tau^2 / 6), or cos / sin of n psi."""
    if model in HARMONICS:
        if psi is None:
            raise ValueError(f'{model}: needs the scan angles psi')
        psi = np.asarray(psi, dtype=float)
        return np.column_stack([f(k * psi) for k in HARMONICS[model] for f in (np.cos, np.sin)])
    tau = np.asarray(tau_yr, dtype=float)[:, None]
    pm = np.asarray(X, dtype=float)[:, [ipa, ipd]]
    return np.hstack([pm * tau / 2.0] + ([pm * tau ** 2 / 6.0] if MODELS[model] == 2 else []))


def augment(rob, tau_yr, ipa, ipd, model, prior=False, psi=None):
    """StudentTSingleStar with the model's columns appended (see columns).

    rob: the single-star model (one noise group); tau_yr: (N,) years from the
    reference epoch; ipa, ipd: its proper-motion columns (position times tau,
    exactly); psi: (N,) scan angles [rad] (harmonic models). prior=True: with
    the proper prior of the new terms in ln L (for evidences and orbit fits);
    False: flat (estimates)."""
    if rob.groups is not None:
        raise NotImplementedError('accel: one noise group only')
    X = np.hstack([rob.X, columns(model, rob.X, tau_yr, ipa, ipd, psi)])
    lam = np.concatenate([rob.lam, np.zeros(X.shape[1] - rob.p)])
    return robust.StudentTSingleStar(rob.d, X, np.sqrt(rob.sig2), lam,
                                     extra=prior_term(model, rob.p) if prior else None)


def _scale_grid(lo, hi):
    """Prior variances s^2 over the grid of one scale and the ln of the
    trapezoid weights times the normalised log-uniform prior, both (S,)."""
    g = np.linspace(np.log(lo), np.log(hi), N_SCALE)
    w = np.full(N_SCALE, g[1] - g[0])
    w[[0, -1]] *= 0.5
    return np.exp(2.0 * g), np.log(w) - np.log(g[-1] - g[0])


def _iso2(C, b, var):
    """ln det(C + v I) and b'(C + v I)^-1 b for symmetric 2 x 2 matrices C
    (..., 2, 2), vectors b (..., 2) and every v of var (S,): (..., S) each, in
    closed form from the eigen-decomposition of C."""
    a, d, c = C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]
    m, r = 0.5 * (a + d), np.hypot(0.5 * (a - d), c)
    th = 0.5 * np.arctan2(2.0 * c, a - d)
    y1 = np.cos(th) * b[..., 0] + np.sin(th) * b[..., 1]
    y2 = -np.sin(th) * b[..., 0] + np.cos(th) * b[..., 1]
    l1, l2 = (m + r)[..., None] + var, (m - r)[..., None] + var
    return np.log(l1) + np.log(l2), y1[..., None] ** 2 / l1 + y2[..., None] ** 2 / l2


def _mix(model):
    """mix(C, b) -> (K,): ln of the integral over the prior scales of
    det(C + S)^-1/2 exp(-b'(C + S)^-1 b / 2), for the fitted terms b (K, k) and
    their flat-prior covariance C (K, k, k). 2 x 2 blocks in closed form: it
    is evaluated for every walker of an orbit fit."""
    va, lwa = _scale_grid(*SCALES[model][0])
    if MODELS[model] == 1:
        def mix(C, b):
            logdet, q = _iso2(C, b, va)                          # (K, S)
            return logsumexp(-0.5 * logdet - 0.5 * q + lwa[None, :], axis=1)
        return mix
    vj, lwj = _scale_grid(*SCALES[model][1])
    lw = (lwj[:, None] + lwa[None, :])[None, :, :]               # (1, Sj, Sa)

    def mix(C, b):
        ba, bj = b[:, :2], b[:, 2:]
        Caa, Caj, Cjj = C[:, :2, :2], C[:, :2, 2:], C[:, 2:, 2:]
        # the jerk block for every jerk scale, inverted in closed form
        a = Cjj[:, None, 0, 0] + vj                              # (K, Sj)
        d = Cjj[:, None, 1, 1] + vj
        c = np.broadcast_to(Cjj[:, None, 0, 1], a.shape)
        det = a * d - c * c
        Mi = np.stack([np.stack([d, -c], -1), np.stack([-c, a], -1)], -2) / det[..., None, None]
        W = Caj[:, None] @ Mi                                    # (K, Sj, 2, 2)
        # the acceleration block given the jerk scale (Schur complement), then
        # every acceleration scale
        Sa = Caa[:, None] - W @ np.swapaxes(Caj, 1, 2)[:, None]
        ra = ba[:, None, :] - np.einsum('ksij,kj->ksi', W, bj)
        logdet, q = _iso2(Sa, ra, va)                            # (K, Sj, Sa)
        logdet = logdet + np.log(det)[..., None]
        q = q + np.einsum('ki,ksij,kj->ks', bj, Mi, bj)[..., None]
        return logsumexp((-0.5 * logdet - 0.5 * q + lw).reshape(len(b), -1), axis=1)
    return mix


def prior_term(model, p0):
    """extra(beta, H) for StudentTSingleStar: the Gaussian prior of the terms
    after the first p0, integrated over them and over its scales."""
    mix = _mix(model)

    def extra(beta, H):
        return mix(np.linalg.inv(H)[:, p0:, p0:], beta[:, p0:])
    return extra


def log_evidence(rob, tau_yr, ipa, ipd, model, nu_lim, f_lim, psi=None):
    """ln Z of a model of MODELS without an orbit: nu and f over their
    log-uniform priors, the scales over theirs, the linear terms analytically."""
    aug = augment(rob, tau_yr, ipa, ipd, model, prior=True, psi=psi)
    return aug.log_evidence(np.zeros(len(aug.d)), nu_lim, f_lim)


def estimate(rob, tau_yr, ipa, ipd, model, nu_lim, f_lim, psi=None):
    """The fitted terms at the best nu and f, with a flat prior on them.

    Returns dict(names, units, value, error, cov, significance, nu, f):
    significance = sqrt(b' C^-1 b), the size of the terms in units of their
    uncertainty (for 'accel9': of the four terms together)."""
    aug = augment(rob, tau_yr, ipa, ipd, model, psi=psi)
    m0 = np.zeros((1, len(aug.d)))
    lo, hi = np.log([nu_lim[0], f_lim[0]]), np.log([nu_lim[1], f_lim[1]])

    def nll(x):
        if np.any(x < lo) or np.any(x > hi):
            return np.inf
        return -float(aug.loglike(m0, np.exp(x[:1]), np.exp(x[1:]))[0])
    x = minimize(nll, np.log([min(max(3.0, nu_lim[0]), nu_lim[1]), 1.0]),
                 method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-4)).x
    nu, f = float(np.exp(x[0])), float(np.exp(x[1]))
    _, beta, H = aug._terms(m0, np.array([nu]), np.array([f]))
    C = np.linalg.inv(H[0])[rob.p:, rob.p:]
    b = beta[0, rob.p:]
    return dict(names=NAMES[model], units=UNITS[model], value=b,
                error=np.sqrt(np.diag(C)), cov=C,
                significance=float(np.sqrt(b @ np.linalg.solve(C, b))), nu=nu, f=f)


# ----------------------------------------------------------------------------
# Gaussian noise model (EXOPLANETLAB_ASTRO_NOISE=gauss): the data are the
# residuals (I - H) d of the single-star fit, so the acceleration columns are
# projected the same way, Y = (I - H) X_acc, and the terms are linear in a
# Gaussian likelihood: fitted and integrated exactly, for every model.
# ----------------------------------------------------------------------------

def projected_columns(X, tau_yr, ipa, ipd, model, projector, psi=None):
    """Y (N, k): the model's columns (see columns), seen through the
    single-star projector (orbit.SingleStarProjector)."""
    return projector.apply(columns(model, X, tau_yr, ipa, ipd, psi))


def gauss_terms(resid, sig2, Y):
    """Least-squares terms for K residual vectors (K, N) with variances sig2
    (K, N) or (N,): b (K, k), their covariance C (K, k, k) and the residuals
    after removing them (K, N)."""
    resid = np.atleast_2d(np.asarray(resid, dtype=float))
    w = 1.0 / np.broadcast_to(np.asarray(sig2, dtype=float), resid.shape)
    k = Y.shape[1]
    YY = np.einsum('np,nq->npq', Y, Y).reshape(len(Y), -1)
    A = (w @ YY).reshape(-1, k, k)
    b = np.linalg.solve(A, ((w * resid) @ Y)[..., None])[..., 0]
    return b, np.linalg.inv(A), resid - b @ Y.T


class GaussAccel:
    """Gaussian ln L with the acceleration terms integrated over their prior:

        ln L = -sum(r^2 / sig2 + ln sig2) / 2 + ln det(C) / 2 + mix(C, b),

    r the residuals after the fitted terms b. Without the terms it is the
    Gaussian ln L of ExoPlanetLab.py, so the evidences compare."""

    def __init__(self, Y, model):
        self.Y = np.asarray(Y, dtype=float)
        self.model = model
        self._mix = _mix(model)

    def loglike(self, resid, sig2):
        """ln L (K,) for residuals data - orbit (K, N), variances (K, N) or (N,)."""
        b, C, r = gauss_terms(resid, sig2, self.Y)
        s2 = np.broadcast_to(np.asarray(sig2, dtype=float), r.shape)
        return (-0.5 * np.sum(r * r / s2 + np.log(s2), axis=1)
                + 0.5 * np.linalg.slogdet(C)[1] + self._mix(C, b))

    def draws(self, resid, sig2, rng):
        """The terms for K residual vectors: fitted values plus a draw from
        their (flat-prior) covariance, (K, k)."""
        b, C, _ = gauss_terms(resid, sig2, self.Y)
        return b + np.einsum('kpq,kq->kp', np.linalg.cholesky(C),
                             rng.standard_normal(b.shape))
