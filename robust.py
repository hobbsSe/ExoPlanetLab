# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Outlier-robust (Student-t) likelihood for Gaia epoch astrometry.

Why
---
The excess noise of the DR4 pre-release epoch astrometry is not Gaussian.
Within one transit the nine CCD positions scatter with a core at the quoted
errors (median |x - median| / sigma = 0.62-0.67, Gaussian 0.67) and heavy
tails: 2-2.5% beyond 5 sigma for Gaia BH3 and HD 114762, up to 46 sigma. A
single white jitter, the Gaussian model's answer, inflates every error to
accommodate a few outliers; for HD 114762 that makes each transit's error
2.4x the robust (median) value and lets the outliers pull the orbit
(astrometry-only P and e further from the RV orbit than Octofitter's
median-based fit). Fitted to the within-transit deviations, a Student-t with
scale f * sigma_i beats the Gaussian-plus-jitter by Delta ln L = 39 / 135 / 478
(Gaia-4 / BH3 / HD 114762; nu = 5.3 / 2.5 / 1.9, f = 0.95 / 0.92 / 0.76), and
an extra jitter adds little.

Model
-----
    d_i = m_i(theta) + (X beta)_i + eps_i,   eps_i ~ t_nu(0, f^2 sigma_i^2)

d: raw along-scan positions; m: the orbit's along-scan signal (unprojected);
X: the single-star design matrix (position, parallax, proper motion,
pseudo-colour) with a Gaussian prior of precision Lambda on its terms
(the colour prior). The Gaussian model uses the residuals of a least-squares
single-star fit, (I - H) d, which is exact only for Gaussian noise: outliers
also pull a least-squares single-star fit. Here beta is fitted robustly for
every likelihood evaluation, by iteratively reweighted least squares (the
Student-t weights w_i = (nu + 1) / (nu + r_i^2 / s_i^2)), and marginalised
with a Laplace term that uses the Fisher information of the t,
(nu + 1) / (nu + 3) / s_i^2 per point:

    ln L(theta, nu, f) = sum_i ln t_nu(r_i; f^2 sigma_i^2) - beta'Lambda beta / 2
                         - ln det(X' diag(c / s^2) X + Lambda) / 2,
    r = d - m - X beta_hat,  c = (nu + 1) / (nu + 3),  s_i = f sigma_i

(constants dropped). As nu -> infinity with f = 1 this is the exact Gaussian
marginal likelihood over the single-star terms (checked: 1e-3 in ln L at
nu = 1e9 against an independent least-squares calculation, the difference
being rounding in gammaln), so the model contains the Gaussian one. Note that
the Gaussian model used elsewhere projects with the weights of gaiasupdate's
single-star fit, which for sources with a large AGIS excess noise (the
companion hosts) are not 1/sigma^2; this model refits the single-star terms
with its own weights instead. Everything is vectorised over K parameter
vectors (walkers) at once; ~10-20 ms per 64 walkers on one core.
"""

import numpy as np
from scipy.special import gammaln

# Reweighting iterations: ln L is converged to ~1e-5 after 12 even for the most
# outlier-rich source (HD 114762, nu ~ 2; 5e-4 after 8, 3e-10 after 20).
N_IRLS = 15          # maximum reweighting iterations
IRLS_TOL = 1e-9      # stop earlier when max |delta beta| < IRLS_TOL (mas)


class StudentTSingleStar:
    """Robust likelihood of raw along-scan data given orbit models.

    Parameters
    ----------
    d : (N,) raw along-scan positions [mas] (gaia_dr4_epoch 'proj_data')
    X : (N, p) single-star design matrix ('proj_design')
    sigma : (N,) quoted along-scan errors [mas]
    lam : (p,) prior precision of the single-star terms ('proj_prior_prec');
          0 for the unconstrained ones
    groups : (N,) int, optional -- data sets with their own noise parameters
          (Gaia = 0, Hipparcos = 1). nu and f are then (K, G) arrays,
          one column per group; without groups they are (K,).
    extra : callable(beta, H) -> (K,), optional -- a term added to ln L from
          the fitted linear terms and their Fisher matrix: the proper prior of
          some of them, integrated analytically (accel.py).
    """

    def __init__(self, d, X, sigma, lam=None, groups=None, extra=None):
        self.extra = extra
        self.d = np.asarray(d, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.sig2 = np.asarray(sigma, dtype=float) ** 2
        self.p = self.X.shape[1]
        self.lam = np.zeros(self.p) if lam is None else np.asarray(lam, dtype=float)
        self._L = np.diag(self.lam)
        self.groups = None if groups is None else np.asarray(groups, dtype=int)
        self.n_groups = 1 if groups is None else int(self.groups.max()) + 1
        # X_i X_i' for every point, so the normal matrices are one contraction
        self._XX = np.einsum('np,nq->npq', self.X, self.X)
        # the same as an (N, p*p) matrix: the weighted normal matrices of K
        # models are then one matrix product (BLAS), 4x faster than the
        # einsum contraction and identical to rounding
        self._XXf = self._XX.reshape(len(self.X), -1)

    def _per_row(self, v, K):
        """nu or f for K models as (K, 1) (one group) or (K, N) (per row)."""
        v = np.asarray(v, dtype=float)
        if self.groups is None:
            return np.broadcast_to(v.reshape(-1), (K,)).reshape(K, 1).copy()
        v = np.broadcast_to(v.reshape(-1, self.n_groups), (K, self.n_groups))
        return v[:, self.groups]

    def _lgamma_term(self, nu, K):
        """gammaln((nu+1)/2) - gammaln(nu/2), per group, as (K, 1) or (K, N)."""
        v = np.asarray(nu, dtype=float)
        if self.groups is None:
            v = np.broadcast_to(v.reshape(-1), (K,)).reshape(K, 1)
            return gammaln((v + 1) / 2) - gammaln(v / 2)
        v = np.broadcast_to(v.reshape(-1, self.n_groups), (K, self.n_groups))
        return (gammaln((v + 1) / 2) - gammaln(v / 2))[:, self.groups]

    def fit_beta(self, e0, s2, nu):
        """IRLS for beta. e0: (K, N) data minus orbit; s2: (K, N) scale^2;
        nu: (K, 1) or (K, N) per row. Returns beta (K, p), final residuals
        (K, N), iterations."""
        K = e0.shape[0]
        w = np.ones_like(e0)
        beta = np.zeros((K, self.p))
        if self.p == 0:                 # no single-star terms (simulated data)
            return beta, e0, 0
        for it in range(1, N_IRLS + 1):
            ws = w / s2
            A = (ws @ self._XXf).reshape(K, self.p, self.p) + self._L
            b = (ws * e0) @ self.X
            beta_new = np.linalg.solve(A, b[..., None])[..., 0]
            r = e0 - beta_new @ self.X.T
            w = (nu + 1.0) / (nu + r * r / s2)
            done = np.max(np.abs(beta_new - beta)) < IRLS_TOL
            beta = beta_new
            if done:
                break
        return beta, r, it

    def _terms(self, models, nu, f):
        """ln L, beta_hat (K, p) and the Fisher matrix H (K, p, p; None without
        single-star terms) for K orbit models."""
        models = np.atleast_2d(np.asarray(models, dtype=float))
        K = models.shape[0]
        nu_ = self._per_row(nu, K)
        s2 = self._per_row(f, K) ** 2 * self.sig2[None, :]
        beta, r, _ = self.fit_beta(self.d[None, :] - models, s2, nu_)
        lt = (self._lgamma_term(nu, K) - 0.5 * np.log(nu_ * np.pi * s2)
              - (nu_ + 1) / 2 * np.log1p(r * r / (nu_ * s2)))
        if self.p == 0:
            return lt.sum(axis=1), beta, None
        c = (nu_ + 1.0) / (nu_ + 3.0)
        H = (np.broadcast_to(c / s2, s2.shape) @ self._XXf).reshape(K, self.p, self.p) + self._L
        _, logdet = np.linalg.slogdet(H)
        prior_beta = 0.5 * np.einsum('kp,p,kp->k', beta, self.lam, beta)
        ll = lt.sum(axis=1) - prior_beta - 0.5 * logdet
        if self.extra is not None:
            ll = ll + self.extra(beta, H)
        return ll, beta, H

    def loglike(self, models, nu, f):
        """ln L for K orbit models (K, N) [mas, unprojected], nu (K,), f (K,)."""
        return self._terms(models, nu, f)[0]

    def log_evidence(self, models, nu_lim, f_lim, n=60, tol=2e-3, chunk=400):
        """ln of the integral of L(models; nu, f) over nu and f with log-uniform
        priors on nu_lim and f_lim, for one fixed orbit model (N,) -- e.g. zeros,
        the N = 0 evidence of the evidence ladder.

        Trapezoid on a grid in (ln nu, ln f) that zooms onto the peak: with many
        points the posterior in ln f is ~0.01 wide, finer than any fixed grid
        over the prior, so each pass re-grids the box where ln L > max - 40 (plus
        a cell) until the estimate changes by less than tol."""
        if self.groups is not None:
            raise NotImplementedError('log_evidence: one noise group only')
        m = np.asarray(models, dtype=float).ravel()[None, :]
        lo = np.log([nu_lim[0], f_lim[0]])
        hi = np.log([nu_lim[1], f_lim[1]])
        norm = np.log(hi[0] - lo[0]) + np.log(hi[1] - lo[1])
        box_lo, box_hi, prev = lo.copy(), hi.copy(), None
        for _ in range(8):
            g0 = np.linspace(box_lo[0], box_hi[0], n)
            g1 = np.linspace(box_lo[1], box_hi[1], n)
            A, B = np.meshgrid(np.exp(g0), np.exp(g1), indexing='ij')
            a, b = A.ravel(), B.ravel()
            ll = np.concatenate([self.loglike(np.repeat(m, len(a[j:j + chunk]), axis=0),
                                              a[j:j + chunk], b[j:j + chunk])
                                 for j in range(0, len(a), chunk)]).reshape(n, n)
            w = np.full(n, 1.0)
            w[[0, -1]] = 0.5
            lw = np.log(np.outer(w, w)) + np.log((g0[1] - g0[0]) * (g1[1] - g1[0]))
            mx = ll.max()
            est = float(mx + np.log(np.sum(np.exp(ll - mx + lw))) - norm)
            if prev is not None and abs(est - prev) < tol:
                return est
            prev = est
            i0, i1 = np.nonzero(ll > mx - 40.0)
            box_lo = np.maximum(lo, [g0[max(i0.min() - 1, 0)], g1[max(i1.min() - 1, 0)]])
            box_hi = np.minimum(hi, [g0[min(i0.max() + 1, n - 1)], g1[min(i1.max() + 1, n - 1)]])
        return est

    def beta_draws(self, models, nu, f, rng):
        """Robust single-star terms for K orbit models: beta_hat plus a draw
        from the inverse Fisher matrix (the Laplace covariance), (K, p)."""
        models = np.atleast_2d(np.asarray(models, dtype=float))
        K = models.shape[0]
        nu_ = self._per_row(nu, K)
        s2 = self._per_row(f, K) ** 2 * self.sig2[None, :]
        beta, _, _ = self.fit_beta(self.d[None, :] - models, s2, nu_)
        c = (nu_ + 1.0) / (nu_ + 3.0)
        H = (np.broadcast_to(c / s2, s2.shape) @ self._XXf).reshape(K, self.p, self.p) + self._L
        L = np.linalg.cholesky(np.linalg.inv(H))
        return beta + np.einsum('kpq,kq->kp', L, rng.standard_normal((K, self.p)))
