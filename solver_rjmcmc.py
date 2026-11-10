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
RJMCMC backend for trans-dimensional parameter estimation (using the eryn package).

Unlike the fixed-N solvers (emcee, dynesty, nessai), this solver treats the
number of planets N as an unknown jointly sampled with the orbit parameters.
ExoPlanetLab has two trans-dimensional solvers:
  rjmcmc  — Gaussian + KDE in-model proposals (this file)
  rjflow  — RealNVP normalising-flow in-model proposals (solver_rjflow.py)

At each MCMC step the chain can:
  - propose new source parameters via an in-model move, or
  - propose a birth (add a source drawn from the prior) or death (remove one)
    via a Reversible-Jump move, accepted via a Metropolis-Hastings ratio that
    correctly accounts for the change in dimensionality.

The marginal posterior over N emerges from the fraction of MCMC steps spent
in each model order.  Posterior samples at the most probable N are extracted
and frequency-sorted for the standard corner plots and diagnostics.

In-model proposals are a mixture: an affine-invariant differential-evolution
move (_PerSourceDEMove) that steps *along* the correlated per-source posterior
ridge, a Gaussian random walk over the regularised full peel covariance
(_PeriodicPerSourceMove), and, when peel_results are available, a KDE
independence sampler (_KDEIndependenceMove).  The whole run may be wrapped in
parallel tempering (EXOPLANETLAB_RJ_NTEMPS).

Public API
----------
run(log_likelihood, log_prior, prior_transform, args, ...) -> SimpleNamespace
    Run RJMCMC.

print_diagnostics(result, active_names)
    Print N-posterior, acceptance fractions, and per-N sample counts.
"""

import copy as _copy
import os
import numpy as np
from types import SimpleNamespace
from config import N_MIN, N_MAX, N_POOL

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform
from eryn.moves import GaussianMove, DistributionGenerateRJ, MHMove

try:
    from scipy.stats import gaussian_kde as _gaussian_kde
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from scipy.stats import truncnorm
from scipy.special import logsumexp
from tight_prior import _tight_stats, period_for

# Parameter indices within the 7-element per-source vector:
#   P(0), e(1), T0(2), A(3), B(4), F(5), G(6)  [Thiele-Innes elements]
#
# None of these are periodic angles (unlike the gravitational-wave
# parameterization this file's boundary-handling logic was adapted from), so
# periodic wrap is unused here (_PERIODIC_IDX is empty).  Reflective fold is
# applied to every bounded, symmetric-range parameter where the MAP can
# plausibly sit at (or near) a prior boundary -- eccentricity e (bounded in
# [0,1)) and the four Thiele-Innes constants A,B,F,G (bounded in a fixed,
# symmetric mas range) -- so a Gaussian proposal centred near a boundary
# still moves freely instead of losing ~half its proposals to simple prior
# rejection.  P and T0 use ordinary prior rejection.
_PERIODIC_IDX = ()               # unused for this parameterization
_REFLECT_IDX  = (1, 3, 4, 5, 6)  # e, A, B, F, G — mirror symmetric → reflect at bounds


def _wrap_reflect(new_c, lo, hi, periodic_idx=_PERIODIC_IDX, reflect_idx=_REFLECT_IDX):
    """In-place periodic wrap / reflective fold of proposed coordinates so a
    parameter sitting at (or driven past) a prior boundary still moves freely.
    `new_c` has trailing axis = ndim; `lo`/`hi` are length-ndim arrays.  Shared
    by every in-model move (Gaussian, DE, KDE fallback)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    for idx in periodic_idx:
        if idx >= new_c.shape[-1]:
            continue
        width = hi[idx] - lo[idx]
        new_c[..., idx] = lo[idx] + (new_c[..., idx] - lo[idx]) % width
    for idx in reflect_idx:
        if idx >= new_c.shape[-1]:
            continue
        span = hi[idx] - lo[idx]
        x = (new_c[..., idx] - lo[idx]) % (2 * span)
        new_c[..., idx] = lo[idx] + np.where(x > span, 2 * span - x, x)
    return new_c


def _regularise_cov(per_source_covs, stds, ndim, shrink=0.10, var_floor_frac=0.05,
                    eig_floor=1.0e-3):
    """Turn a list of per-source empirical covariance matrices into ONE
    well-conditioned, positive-definite proposal covariance.

    Each planet's per-source posterior is a thin, strongly-correlated ridge
    (period P, epoch T0 and the Thiele-Innes constants trade off against each
    other, and P is very tightly constrained relative to its prior).  Passing
    the raw empirical covariance to eryn's GaussianMove makes proposals move
    *along* that ridge -- but the raw matrix is often numerically near-singular
    (a very tightly pinned parameter has a tiny variance), so
    ``np.linalg.cholesky`` raises and a plain diagonal proposal steps
    perpendicular to the ridge -> near-zero in-model acceptance.

    This regulariser: (1) averages the per-source covariances; (2) floors the
    eigenvalues of the *correlation* matrix at ``eig_floor`` so a near-perfect
    correlation no longer implies a zero eigenvalue; (3) floors each variance
    at ``(var_floor_frac * std)**2``; (4) shrinks ``shrink`` toward the
    diagonal (Ledoit-Wolf style); (5) adds escalating jitter until Cholesky
    succeeds.

    Returns ``(cov, ok, cond)``: the matrix, whether it is usable, and the
    2-norm condition number of its correlation matrix (``np.inf`` if the
    fallback diagonal is returned).
    """
    stds = np.asarray(stds, dtype=float).copy()
    stds[~np.isfinite(stds) | (stds <= 0)] = 1.0e-12

    covs = [np.asarray(c, dtype=float) for c in per_source_covs
            if np.asarray(c).shape == (ndim, ndim) and np.all(np.isfinite(c))]
    if not covs:
        return np.diag(stds ** 2), False, np.inf

    c = np.mean(covs, axis=0)
    c = 0.5 * (c + c.T)

    # (2) eigenvalue floor on the correlation matrix
    d = np.sqrt(np.clip(np.diag(c), 1.0e-300, None))
    R = c / np.outer(d, d)
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eig_floor, None)
    R = (V * w) @ V.T
    dR = np.sqrt(np.clip(np.diag(R), 1.0e-300, None))
    R = R / np.outer(dR, dR)                      # renormalise unit diagonal
    c = R * np.outer(d, d)

    # (3) variance floor
    var_floor = (var_floor_frac * stds) ** 2
    dc = np.maximum(np.diag(c), var_floor)
    np.fill_diagonal(c, dc)
    c = 0.5 * (c + c.T)

    # (4) shrink toward the diagonal
    c = (1.0 - shrink) * c + shrink * np.diag(np.diag(c))

    # No cap on the condition number of the matrix in raw units: the period,
    # the epoch and the Thiele-Innes constants have spreads that differ by
    # orders of magnitude, so the raw matrix is badly conditioned by
    # construction.  Capping it inflates the steps of the tightly constrained
    # parameters and the Gaussian proposals are rejected.  The correlation
    # matrix is well conditioned after step (2), and _ScaledProposal draws
    # from the matrix through its correlation form, so the raw scales never
    # enter a decomposition.
    c = 0.5 * (c + c.T)

    # (5) escalating jitter (on the correlation matrix) until PD
    d = np.sqrt(np.clip(np.diag(c), 1.0e-300, None))
    R = c / np.outer(d, d)
    for k in range(-12, -2):
        try:
            np.linalg.cholesky(R)
            cond = float(np.linalg.cond(R))
            return R * np.outer(d, d), True, cond
        except np.linalg.LinAlgError:
            R = R + (10.0 ** k) * np.eye(ndim)
            R = R / (1.0 + 10.0 ** k)
    return np.diag(np.maximum(np.diag(c), var_floor)), False, np.inf


def _source_covs(peel_results, ndim):
    """One proposal covariance per detected planet, from its warm-start samples.

    Unlike the common matrix these are not shrunk towards the diagonal and the
    correlation eigenvalues are floored only at 1e-8: the samples of a single
    planet define its ridge well, and widening it across the ridge is what
    makes a Gaussian step fail.  Returns [] if any planet lacks usable samples
    (the move then uses the common matrix for every leaf).
    """
    out = []
    for pr in (peel_results or []):
        smp = getattr(pr, 'samples', None)
        if smp is None or smp.ndim != 2 or smp.shape[1] != ndim \
                or smp.shape[0] <= 2 * ndim:
            return []
        c, ok, _ = _regularise_cov([np.cov(smp, rowvar=False)], np.std(smp, axis=0),
                                   ndim, shrink=0.0, var_floor_frac=0.0,
                                   eig_floor=1.0e-8)
        if not ok:
            return []
        out.append(c)
    return out


class _SourceConditioner:
    """Linear dependence of each detected planet on the other detected planets.

    The planets share the same measurements, so their parameters are
    correlated in the joint posterior: given the other planets, the allowed
    region of one planet is a small part of its marginal distribution (a few
    per cent of the volume for two planets with periods within a factor of
    two).  An independence proposal drawn from the marginal distribution then
    lands outside that region nearly every time.

    For planet k this class fits x_k = mu_k + (x_others - mu_others) K_k + r_k
    by least squares on the warm-start samples, which must be rows of the
    same joint chain.  The independence moves draw the residual r_k (from a
    KDE or a flow) and add the term in the current x_others, which makes the
    proposal conditional on the other planets.  ``ok`` is False when there is
    one planet only, when the sample sets differ in length, or when
    EXOPLANETLAB_RJ_CONDITIONAL=0.
    """

    def __init__(self, peel_results, ndim):
        self.ok = False
        self.n = 0
        if os.environ.get('EXOPLANETLAB_RJ_CONDITIONAL', '1').strip() == '0':
            return
        smp = [getattr(pr, 'samples', None) for pr in (peel_results or [])]
        if len(smp) < 2 or any(x is None for x in smp):
            return
        smp = [np.asarray(x, dtype=float) for x in smp]
        m = smp[0].shape[0]
        if any(x.ndim != 2 or x.shape != (m, ndim) for x in smp) \
                or m <= 4 * ndim * len(smp):
            return
        self.n = len(smp)
        self._mu_o, self._K, self._res, self.volume = [], [], [], []
        for k in range(self.n):
            xo = np.hstack([smp[j] for j in self.others(k)])
            mu_o = xo.mean(axis=0)
            sd_o = xo.std(axis=0)
            sd_o = np.where(sd_o > 0.0, sd_o, 1.0)
            mu_k = smp[k].mean(axis=0)
            K, *_ = np.linalg.lstsq((xo - mu_o) / sd_o, smp[k] - mu_k, rcond=1.0e-8)
            K = K / sd_o[:, None]
            res = smp[k] - (xo - mu_o) @ K
            # volume of the conditional region relative to the marginal one
            d = smp[k].std(axis=0)
            d = np.where(d > 0.0, d, 1.0)
            _, ld_res = np.linalg.slogdet(np.cov(res / d, rowvar=False))
            _, ld_all = np.linalg.slogdet(np.cov(smp[k] / d, rowvar=False))
            self._mu_o.append(mu_o)
            self._K.append(K)
            self._res.append(res)
            self.volume.append(float(np.exp(0.5 * (ld_res - ld_all))))
        self.ok = True

    def others(self, k):
        return [j for j in range(self.n) if j != k]

    def residuals(self, k):
        """Warm-start samples of planet k with the dependence on the others removed."""
        return self._res[k]

    def shift(self, k, x_others):
        """Term to add to a residual; x_others has shape (..., (n-1)*ndim)."""
        return (np.asarray(x_others, dtype=float) - self._mu_o[k]) @ self._K[k]

    def describe(self):
        return ", ".join(f"planet {k + 1}: {v:.3f}" for k, v in enumerate(self.volume))


class _ScaledProposal:
    """Full-covariance Gaussian proposal that is accurate whatever the units.

    eryn's own proposal calls ``rng.multivariate_normal`` on the raw matrix,
    which decomposes it in absolute units and loses the directions whose
    variance is far below the largest one.  This one factorises the
    correlation matrix and multiplies by the standard deviations afterwards.
    ``scale`` is the covariance.
    """

    def __init__(self, scale, factor=None):
        self.scale = np.asarray(scale, dtype=float)
        self._log_factor = None if factor is None else np.log(factor)

    def __call__(self, x0, rng):
        d = np.sqrt(np.diag(self.scale))
        L = np.linalg.cholesky(self.scale / np.outer(d, d))
        f = (1.0 if self._log_factor is None
             else np.exp(rng.uniform(-self._log_factor, self._log_factor)))
        z = rng.standard_normal(x0.shape)
        return x0 + f * (z @ L.T) * d, np.zeros(len(x0))


def _parse_move_mix(env_value, default):
    """Parse ``"name:weight,name:weight,..."`` (from an env var) into an ordered
    list of ``(name, weight)`` with weights renormalised to sum to 1.  Unknown
    or malformed entries fall back to ``default`` (same format)."""
    text = (env_value or default).strip()
    out = []
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        name, _, w = tok.partition(':')
        try:
            out.append((name.strip().lower(), float(w)))
        except ValueError:
            continue
    if not out:
        out = _parse_move_mix(None, default)
        return out
    tot = sum(w for _, w in out) or 1.0
    return [(n, w / tot) for n, w in out]


class _PeriodicPerSourceMove(GaussianMove):
    """GaussianMove with two in-model acceptance improvements.

    Recommendation 1 — per-source independent proposals: each MCMC step
    perturbs ONE randomly selected active leaf rather than all N*7 parameters
    jointly, reducing effective proposal dimension from N*7 to 7 and
    eliminating compound prior rejections across sources.

    Recommendation 2 — periodic / reflective boundary conditions: angular
    parameters are wrapped (φ, φ₀, φ_L) or reflected (θ, θ_L) so that a
    parameter sitting exactly at a prior boundary still moves freely.
    """

    def __init__(self, cov, param_lo, param_hi, target_accept=0.30,
                 source_covs=None, **kwargs):
        super().__init__(cov, **kwargs)
        for _name, _prop in list(self.all_proposal.items()):
            if np.ndim(_prop.scale) == 2:
                self.all_proposal[_name] = _ScaledProposal(_prop.scale)
        # One covariance per planet (leaf k is planet k, as in the KDE move);
        # leaves beyond the list use the common matrix.  Two planets have
        # different ridges, and their average fits neither.
        self._source_props = [_ScaledProposal(c) for c in (source_covs or [])]
        # Step factor per temperature, set by rescale().  A hot chain accepts
        # much larger steps than the cold one, so a factor tuned on the
        # average over temperatures is far too large for the cold chain.
        self._temp_factor = None
        self._lo = np.asarray(param_lo, dtype=float)
        self._hi = np.asarray(param_hi, dtype=float)
        self._target_accept = target_accept
        self._tune_count = 0

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        new_coords = {}
        for branch_name, orig in branches_coords.items():
            # orig: (ntemps, nwalkers, n_max, ndim)
            new_c = orig.copy()
            inds  = (branches_inds[branch_name] if branches_inds is not None
                     else np.ones(orig.shape[:3], dtype=bool))
            ntemps, nwalkers = orig.shape[:2]
            common = self.all_proposal[branch_name]
            tf = (self._temp_factor if self._temp_factor is not None
                  and len(self._temp_factor) == ntemps else np.ones(ntemps))

            # Rec 1: for each walker move ONE active leaf, with the covariance
            # of its planet and the step factor of its temperature
            for t in range(ntemps):
                for w in range(nwalkers):
                    active = np.where(inds[t, w])[0]
                    if len(active) == 0:
                        continue
                    k = active[int(random.randint(len(active)))]
                    prop = (self._source_props[k] if k < len(self._source_props)
                            else common)
                    x0 = orig[t, w, k][None, :]
                    new_c[t, w, k] = x0[0] + tf[t] * (prop(x0, random)[0][0] - x0[0])

            # Rec 2: periodic wrap for φ/φ₀/φ_L, reflective fold for θ/θ_L
            # (indices beyond ndim are skipped, e.g. when a FIT_FLAGS subset is
            # fitted or in reduced-dimension tests).
            _wrap_reflect(new_c, self._lo, self._hi)
            new_coords[branch_name] = new_c
            factors = np.zeros((ntemps, nwalkers))

        return new_coords, factors

    def rescale(self, accept_rate):
        """Robbins-Monro adaptive scaling of the proposal std toward
        ``target_accept`` (default 0.30, matching the acceptance emcee's
        StretchMove reaches automatically).  The fixed-covariance GaussianMove
        has no such self-tuning; at the un-adapted defaults its in-model
        acceptance measured ~0.6% on a real 2-planet run vs. emcee's ~33%,
        leaving the chain stuck near the biased peel starting point instead
        of escaping it.  (The differential-evolution move added to the
        in-model mixture is the primary fix for this; the Gaussian move's
        rescaling is a secondary tuning of its minority share.)

        Step size decays as 1/sqrt(n) (diminishing adaptation) so the scale
        stabilizes rather than oscillating. Call this manually between short
        burn-in chunks (see run()) with the acceptance rate of each
        temperature over the last chunk (an array, cold chain first).

        Deliberately NOT wired up as this class's ``tune(state, accepted)``
        override, and burn-in deliberately does NOT use eryn's own
        ``sample(..., tune=True)`` mechanism: eryn's ``ReversibleJumpMove``
        (the birth/death move used here) repurposes the name ``tune`` for
        an unrelated boolean config flag (``self.tune = tune`` in its
        ``__init__``), shadowing the callable ``Move.tune()`` method. Passing
        ``tune=True`` to ``sampler.sample``/``run_mcmc`` makes eryn try to
        call ``rj_move.tune(state, accepted)`` on that bool and crash
        (``TypeError: 'bool' object is not callable``). Driving rescaling
        manually sidesteps that entirely.
        """
        self._tune_count += 1
        step = 1.0 / np.sqrt(self._tune_count + 1)
        # one acceptance rate per temperature -> one step factor per temperature
        rate = np.atleast_1d(np.asarray(accept_rate, dtype=float))
        if self._temp_factor is None or len(self._temp_factor) != len(rate):
            self._temp_factor = np.ones(len(rate))
        self._temp_factor = self._temp_factor * np.exp(
            step * (rate - self._target_accept))


class _KDEIndependenceMove(GaussianMove):
    """Per-source independence sampler using KDE of peel posteriors.

    Recommendation 3 — KDE independence sampler: proposes θ* drawn directly
    from a scipy.stats.gaussian_kde built from the peel (emcee) samples for
    each source rather than a Gaussian random walk from the current position.
    When the KDE accurately captures the posterior the acceptance approaches
    unity.  The Metropolis-Hastings correction is

        factors = log KDE(θ_current) − log KDE(θ_proposed)

    which exactly cancels the KDE proposal density, leaving only the
    posterior ratio in the acceptance criterion.

    Only applied to leaf index k when k < len(kdes); other active leaves are
    left unchanged (per-source independence same as Rec 1).
    """

    def __init__(self, kdes, cov_fallback, param_lo, param_hi,
                 conditioner=None, cond_kdes=None, **kwargs):
        super().__init__(cov_fallback, **kwargs)
        self._kdes = kdes
        # KDEs of the residuals of _SourceConditioner: used for leaf k when
        # all the other detected planets are active in the walker
        self._cond      = conditioner if cond_kdes else None
        self._cond_kdes = cond_kdes or []
        self._lo   = np.asarray(param_lo, dtype=float)
        self._hi   = np.asarray(param_hi, dtype=float)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        new_coords = {k: v.copy() for k, v in branches_coords.items()}

        for branch_name in list(new_coords.keys()):
            orig  = branches_coords[branch_name]
            new_c = new_coords[branch_name]
            inds  = branches_inds[branch_name]
            ntemps, nwalkers = orig.shape[:2]
            factors = np.zeros((ntemps, nwalkers))

            for t in range(ntemps):
                for w in range(nwalkers):
                    active = np.where(inds[t, w])[0]
                    valid  = [k for k in active if k < len(self._kdes)]
                    if not valid:
                        continue
                    chosen = valid[int(random.randint(len(valid)))]
                    kde    = self._kdes[chosen]
                    shift  = 0.0
                    if self._cond is not None and chosen < len(self._cond_kdes):
                        others = self._cond.others(chosen)
                        if all(inds[t, w, j] for j in others):
                            # conditional on the other planets of this walker
                            kde   = self._cond_kdes[chosen]
                            shift = self._cond.shift(
                                chosen, orig[t, w, others].ravel()).reshape(-1, 1)

                    θ_cur  = orig[t, w, chosen].reshape(-1, 1)   # (ndim, 1)
                    θ_prop = np.clip(
                        kde.resample(1) + shift,
                        self._lo[:, None], self._hi[:, None],
                    )                                             # (ndim, 1)

                    # kde.logpdf returns a shape-(1,) array; NumPy >=2.0 made
                    # float() on a non-0-d array a hard error, so use .item().
                    log_q_cur  = kde.logpdf(θ_cur - shift).item()
                    log_q_prop = kde.logpdf(θ_prop - shift).item()
                    new_c[t, w, chosen] = θ_prop.flatten()
                    # MH correction: log q(cur|prop) - log q(prop|cur)
                    # = log KDE(cur) - log KDE(prop)  (independence sampler)
                    factors[t, w] = log_q_cur - log_q_prop

        return new_coords, factors


class _PerSourceDEMove(MHMove):
    """Differential-evolution in-model move (affine-invariant, self-tuning).

    For each walker, one active leaf ``k`` is perturbed along the difference
    vector between the same leaf of two other random walkers:

        θ'  =  θ_{w,k}  +  γ · (θ_{a,k} − θ_{b,k})  +  ε

    with ``γ = 2.38 / sqrt(2·ndim)`` most steps and ``γ = 1`` a fraction
    ``p_gamma1`` of the time (mode hopping), plus a tiny Gaussian jitter ``ε``.
    Because the proposal is built from the *current ensemble*, it automatically
    adapts to whatever linear correlation the posterior has -- it steps
    *along* the thin correlated per-source ridge without needing a covariance estimate, which
    is exactly why emcee's affine-invariant stretch move escapes the ridge
    where a fixed-covariance Gaussian cannot.  The move is symmetric, so the
    Hastings factor is zero.
    """

    def __init__(self, param_lo, param_hi, jitter_std, p_gamma1=0.1):
        super().__init__()
        self._lo     = np.asarray(param_lo, dtype=float)
        self._hi     = np.asarray(param_hi, dtype=float)
        self._jit    = np.asarray(jitter_std, dtype=float)
        self._pg1    = float(p_gamma1)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        new_coords = {k: v.copy() for k, v in branches_coords.items()}
        first = next(iter(branches_coords.values()))
        ntemps, nwalkers = first.shape[:2]
        factors = np.zeros((ntemps, nwalkers))

        for bname, orig in branches_coords.items():
            _, _, nleaves_max, ndim = orig.shape
            new_c = new_coords[bname]
            if branches_inds is not None and bname in branches_inds:
                inds = branches_inds[bname]
            else:
                inds = np.ones(orig.shape[:3], dtype=bool)

            gamma0 = 2.38 / np.sqrt(2.0 * ndim)
            for t in range(ntemps):
                for w in range(nwalkers):
                    active = np.where(inds[t, w])[0]
                    if len(active) == 0:
                        continue
                    k = active[int(random.randint(len(active)))]
                    # donors: other walkers with leaf k active
                    donors = [ww for ww in range(nwalkers)
                              if ww != w and inds[t, ww, k]]
                    if len(donors) < 2:
                        continue
                    a, b = (donors[i] for i in
                            random.choice(len(donors), size=2, replace=False))
                    gamma = 1.0 if random.rand() < self._pg1 else gamma0
                    step  = gamma * (orig[t, a, k] - orig[t, b, k])
                    step += self._jit * random.randn(ndim)
                    new_c[t, w, k] = orig[t, w, k] + step

            _wrap_reflect(new_c, self._lo, self._hi)

        return new_coords, factors


# Default tuning parameters.  Overridable via env vars for experimentation
# without editing this file.  Burn-in needs enough steps for the adaptive
# proposal-scale tuning (see _PeriodicPerSourceMove.rescale) to converge
# before it's frozen for production.
NWALKERS    = int(os.environ.get('EXOPLANETLAB_RJ_NWALKERS', 50))   # even, > 2*ndim_per
NSTEPS_BURN = int(os.environ.get('EXOPLANETLAB_RJ_NSTEPS_BURN', 2000))
NSTEPS_PROD = int(os.environ.get('EXOPLANETLAB_RJ_NSTEPS_PROD', 8000))

def _build_prior(param_lo, param_hi, log_uniform_idx):
    """Build a ProbDistContainer from per-parameter bounds.

    Parameters in `log_uniform_idx` (e.g. the period P) use a log-uniform
    (Jeffreys) prior; all others are uniform.
    """
    dists = {}
    for i in range(len(param_lo)):
        lo, hi = float(param_lo[i]), float(param_hi[i])
        if i in log_uniform_idx:
            dists[i] = log_uniform(lo, hi)
        else:
            dists[i] = uniform_dist(lo, hi)
    return ProbDistContainer(dists)


class _1DMixture:
    """Equal-weight mixture of frozen 1D truncated-Gaussian components (one
    per peeled source), exposing the .rvs(size)/.logpdf(x) interface
    ProbDistContainer expects for a per-dimension sub-distribution.

    For circular parameters (`period` given), each component is centred at
    0 and samples/evaluation points are wrapped relative to the component's
    own circular mean `mu` before scoring — mirrors
    tight_prior._tight_log_prior_factory's treatment of the 0/period cut."""
    def __init__(self, dists, lo_phys, period=None):
        self.dists   = dists   # list of (frozen_truncnorm, mu)
        self.k       = len(dists)
        self.lo_phys = lo_phys
        self.period  = period

    def rvs(self, size=1):
        out_shape = size if isinstance(size, tuple) else (size,)
        n = int(np.prod(out_shape)) if len(out_shape) else 1
        choice = np.random.randint(0, self.k, size=n)
        out = np.empty(n)
        for i in range(n):
            d, mu = self.dists[choice[i]]
            v = float(d.rvs(size=1)[0])
            if self.period is not None:
                v = np.mod(v + mu - self.lo_phys, self.period) + self.lo_phys
            out[i] = v
        return out.reshape(out_shape)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        logps = []
        for d, mu in self.dists:
            if self.period is not None:
                xw = np.mod(x - mu + self.period / 2.0, self.period) - self.period / 2.0
                logps.append(d.logpdf(xw))
            else:
                logps.append(d.logpdf(x))
        return logsumexp(np.stack(logps, axis=0), axis=0) - np.log(self.k)


def _build_birth_generate_dist(peel_results, param_lo, param_hi, ndim, fit_idx=None):
    """Tight per-source truncated Gaussians (from the peel posteriors),
    mixed across sources per dimension.  Used as the RJ birth-proposal
    generator instead of the flat/wide physical prior: birth proposals
    drawn from a prior far wider than the true posterior have near-zero
    acceptance, biasing the N-posterior toward lower N regardless of the
    true evidence (Birth/death acceptance = 0.000 symptom).  The true
    prior (model_prior) is unaffected -- DistributionGenerateRJ's factors
    already contain the correct MH proposal-density correction, so using
    an informed generate_dist here is statistically safe.

    `fit_idx` maps local dimension j to the global parameter index used by
    tight_prior.period_for(), for the circular (phi, phi_0, phi_L) angles;
    defaults to the identity map (j itself is the global index)."""
    if fit_idx is None:
        fit_idx = list(range(ndim))
    dists = {}
    for j in range(ndim):
        period = period_for(fit_idx[j])
        per_source = []
        for pr in peel_results:
            mu, sig, lo, hi, a, b = _tight_stats(
                pr.samples[:, j], param_lo[j], param_hi[j], period=period)
            if period is not None:
                per_source.append((truncnorm(a, b, loc=0.0, scale=sig), mu))
            else:
                per_source.append((truncnorm(a, b, loc=mu, scale=sig), mu))
        dists[j] = _1DMixture(per_source, lo_phys=float(param_lo[j]), period=period)
    return ProbDistContainer(dists)


def run(log_likelihood, log_prior, prior_transform, args,
        p0=None, ndim=7, n_max=N_MAX, n_min=N_MIN,
        nwalkers=NWALKERS, nsteps_burn=NSTEPS_BURN, nsteps_prod=NSTEPS_PROD,
        peel_best=None, peel_results=None, birth_results=None,
        param_lo=None, param_hi=None,
        log_uniform_idx=(0,),
        fit_idx=None,
        n_pool=N_POOL,
        **kwargs):
    """Run Eryn Reversible-Jump MCMC.

    Parameters
    ----------
    log_likelihood  : callable  (leaf_params, *args) -> float
                      leaf_params has shape (n_active, ndim): parameters of
                      all active sources for this walker.
    log_prior       : not used; Eryn handles priors via ProbDistContainer.
    prior_transform : not used.
    args            : tuple  forwarded to log_likelihood as extra arguments.
    ndim            : int    number of parameters per source (7 for a full fit).
    n_max           : int    maximum number of sources to consider.
    n_min           : int    minimum number of sources (default 1).
    nwalkers        : int    ensemble size; must be even and > 2 * ndim.
    nsteps_burn     : int    burn-in steps (discarded).
    nsteps_prod     : int    production steps (kept).
    peel_best       : list   of ndarray (ndim,) — peel best-fit per known source.
    peel_results    : list   of SimpleNamespace — emcee peel results used to
                      estimate the in-model proposal covariance.  *This is
                      strongly recommended*: posteriors are typically far
                      narrower than the prior, so proposals based on the prior
                      range will have near-zero acceptance.  ExoPlanetLab.py
                      always supplies peel_results (running a short emcee warm-
                      start for the N=1 case); the fallback log-range heuristic
                      is provided only for external callers.
    param_lo        : list   lower prior bound per parameter (length ndim).
    param_hi        : list   upper prior bound per parameter (length ndim).
    log_uniform_idx : tuple  local parameter indices with log-uniform priors.
    birth_results   : list   of SimpleNamespace, optional — peel posteriors used
                      ONLY to build the RJ birth-proposal mixture.  Defaults to
                      `peel_results`.  ExoPlanetLab.py passes the detected-source
                      posteriors PLUS the sub-threshold survey-probe peel (the
                      strongest unexplained residual feature), so a birth can be
                      proposed where an as-yet-undetected planet would sit
                      without polluting the in-model proposal covariance / KDEs.
    fit_idx         : list   local-index -> global-parameter-index map (see
                      ExoPlanetLab.py's FIT_IDX), used only to identify circular
                      parameters for the RJ birth-proposal distribution (none in
                      the Thiele-Innes parameterization); defaults to range(ndim).

    Environment overrides
    ---------------------
    EXOPLANETLAB_RJ_NTEMPS      int   parallel-tempering ladder size (default 4; set 1
                                to disable).  Hot chains carry birth/death moves
                                and swap them down to the cold chain, curing a
                                stuck N.
    EXOPLANETLAB_RJ_INMODEL_MIX str   ``"name:w,name:w"`` over {de, gauss, kde} (or
                                {flow, de, gauss} for rjflow) — the in-model
                                move mixture.  Default gives the affine-invariant
                                DE move the largest share.
    EXOPLANETLAB_RJ_START       str   initial N per walker: ``mixed`` (a quarter at
                                n_min to probe the birth channel, the rest at
                                the peel count — default), ``nmin`` (all at
                                n_min — strongest test, but the N-posterior is
                                wrong if births never fire), or ``npeel`` (all
                                warm-started at the peel count).

    Returns
    -------
    SimpleNamespace with fields:
        samples                (n_flat, most_probable_n * ndim)
        log_probs              (n_flat,) — log-likelihoods
        best_params            (most_probable_n * ndim,) — MAP sample
        n_chain                (nsteps_prod, nwalkers) — N per step/walker
        most_probable_n        int
        n_distribution         ndarray  — posterior sample count per N = 0, 1, …
        acceptance_fraction    float    — mean in-model move acceptance
        rj_acceptance_fraction float    — mean birth/death acceptance
        chain, chain_burn, logprob_burn, tau, tau_err  — all None
        logz, logzerr          — None (Eryn does not compute evidence)
        nsteps_prod            int
    """
    import multiprocessing as _mp

    if param_lo is None or param_hi is None:
        raise ValueError("param_lo and param_hi must be provided to solver_rjmcmc.run()")

    pool = None
    if n_pool > 1:
        if _mp.get_start_method(allow_none=True) != 'fork':
            _mp.set_start_method('fork', force=True)
        pool = _mp.Pool(n_pool)

    ntemps   = max(1, int(os.environ.get('EXOPLANETLAB_RJ_NTEMPS', '4')))
    start_mode = os.environ.get('EXOPLANETLAB_RJ_START', 'mixed').strip().lower()
    if start_mode not in ('mixed', 'nmin', 'npeel'):
        start_mode = 'mixed'
    nwalkers = max(nwalkers, 2 * ndim + 2)
    if nwalkers % 2 != 0:
        nwalkers += 1

    # ── Prior ────────────────────────────────────────────────────────────────
    model_prior = _build_prior(param_lo, param_hi, log_uniform_idx)
    priors   = {'model': model_prior}

    # ── In-model Gaussian proposal covariance ────────────────────────────────
    # Posteriors are typically far narrower than the prior, so the proposal
    # scale must be matched to the posterior, not the prior.
    #
    # If peel_results is provided (from the emcee warm-start in ExoPlanetLab.py),
    # use the empirical posterior covariance directly.  The per-source posterior
    # is a thin, strongly-correlated ridge (P, T0 and the Thiele-Innes constants
    # trade off; P is very tightly constrained).  A *diagonal* proposal with the
    # right marginal widths steps almost perpendicular to the ridge -> huge
    # negative Delta lnL -> in-model acceptance ~ 0 no matter how the scalar
    # step size is tuned.  Use the regularised full empirical covariance so
    # proposals move *along* the ridge, shrunk 10% toward its diagonal
    # (Ledoit-Wolf style) for positive-definiteness.
    #
    # Fallback (no peel_results): 1% of the log range for log-uniform
    # parameters, 1% of the linear range for the rest.  This is usually too
    # large; the N-posterior from the RJ moves is still usable, but
    # peel_results is strongly preferred.
    cov_mat = None
    if peel_results:
        stds = np.median(
            [np.std(pr.samples, axis=0) for pr in peel_results], axis=0
        )
        _covs = [np.cov(pr.samples, rowvar=False) for pr in peel_results
                 if getattr(pr, 'samples', None) is not None
                 and pr.samples.shape[0] > 2 * pr.samples.shape[1]]
        if _covs:
            cov_mat, _cov_ok, _cov_cond = _regularise_cov(_covs, stds, ndim)
            if _cov_ok:
                print(f"  In-model proposal: full empirical covariance "
                      f"(correlation cond ~ {_cov_cond:.1e}) — proposals track the ridge.")
            else:
                cov_mat = None
                print("  In-model proposal: covariance not positive-definite "
                      "after regularisation — falling back to diagonal.")
    else:
        # Centre values for log-uniform parameters from peel_best
        peel_centres = (
            np.mean(np.array([np.abs(pb) for pb in peel_best]), axis=0)
            if peel_best else
            np.array([np.sqrt(float(param_lo[j]) * float(param_hi[j]))
                      if j in log_uniform_idx
                      else 0.5 * (float(param_lo[j]) + float(param_hi[j]))
                      for j in range(ndim)])
        )
        stds = np.array([
            (float(peel_centres[j]) * 0.01 * np.log(float(param_hi[j]) / float(param_lo[j]))
             if j in log_uniform_idx
             else 0.01 * (float(param_hi[j]) - float(param_lo[j])))
            for j in range(ndim)
        ])
    cov = {'model': cov_mat if cov_mat is not None else np.diag(stds ** 2)}

    param_lo_arr = np.array([float(x) for x in param_lo])
    param_hi_arr = np.array([float(x) for x in param_hi])

    # ── Moves ────────────────────────────────────────────────────────────────
    # DistributionGenerateRJ: birth/death proposals.  Prefer a tight,
    # peel-informed mixture over the flat physical prior -- see
    # _build_birth_generate_dist docstring (Birth/death acceptance = 0.000
    # fix).  Falls back to the flat prior when no peel_results are available
    # (matches the in-model proposal fallback above).
    _birth_src = birth_results if birth_results else peel_results
    if _birth_src:
        _fit_idx = fit_idx if fit_idx is not None else list(range(ndim))
        birth_generate_dist = _build_birth_generate_dist(
            _birth_src, param_lo_arr, param_hi_arr, ndim, fit_idx=_fit_idx)
        if birth_results and len(birth_results) > (len(peel_results) if peel_results else 0):
            print(f"  Birth proposal: {len(birth_results)} components "
                  f"(detected sources + strongest unexplained residual peak).")
    else:
        birth_generate_dist = model_prior
    rj_move = DistributionGenerateRJ(
        {'model': birth_generate_dist},
        nleaves_max={'model': n_max},
        nleaves_min={'model': n_min},
    )

    # Rec 1 + 2: per-source Gaussian with periodic / reflective wrapping
    gauss_move = _PeriodicPerSourceMove(cov, param_lo_arr, param_hi_arr,
                                        source_covs=_source_covs(peel_results, ndim))

    # Rec 3: KDE independence sampler built from peel posteriors.
    # Cap the training set: gaussian_kde scales as O(n_train) per evaluation.
    # Bandwidth: Scott's rule, n^(-1/(d+4)) (about 0.5 sigma at n=2000 in 7
    # dimensions).  An earlier version forced 0.25, where a kernel holds
    # 0.25^7 * 2000 = 0.1 other samples and the estimate is close to 2000
    # separate spikes.  EXOPLANETLAB_RJ_KDE_BW overrides it with a factor.
    _MAX_KDE_SAMPLES = 4000
    _bw_env = os.environ.get('EXOPLANETLAB_RJ_KDE_BW', 'scott').strip().lower()
    _KDE_BW = None if _bw_env == 'scott' else float(_bw_env)
    print(f"  KDE move bandwidth: {'Scott rule' if _KDE_BW is None else _KDE_BW}")
    kdes = []
    if _SCIPY_AVAILABLE and peel_results:
        for pr in peel_results:
            if pr is not None and getattr(pr, 'samples', None) is not None \
                    and len(pr.samples) >= 2 * ndim:
                try:
                    s = pr.samples
                    if len(s) > _MAX_KDE_SAMPLES:
                        idx = np.random.choice(len(s), _MAX_KDE_SAMPLES, replace=False)
                        s = s[idx]
                    kdes.append(_gaussian_kde(s.T, bw_method=_KDE_BW))
                except Exception:
                    pass

    # The same KDE on the residuals of each planet about its linear dependence
    # on the other planets (see _SourceConditioner).
    conditioner = _SourceConditioner(peel_results, ndim)
    cond_kdes = []
    if kdes and conditioner.ok and len(kdes) == conditioner.n:
        try:
            for k in range(conditioner.n):
                s = conditioner.residuals(k)
                if len(s) > _MAX_KDE_SAMPLES:
                    s = s[np.random.choice(len(s), _MAX_KDE_SAMPLES, replace=False)]
                cond_kdes.append(_gaussian_kde(s.T, bw_method=_KDE_BW))
            print("  KDE move: each planet is drawn given the others. Volume of the "
                  f"conditional region relative to the marginal one: {conditioner.describe()}")
        except Exception:
            cond_kdes = []

    # Affine-invariant differential-evolution move: builds its step from the
    # current ensemble, so it tracks the thin correlated per-source ridge without a covariance
    # estimate — the piece that lifts in-model acceptance off 0.000.
    de_move = _PerSourceDEMove(param_lo_arr, param_hi_arr, 1.0e-3 * np.asarray(stds))

    _move_by_name = {'de': de_move, 'gauss': gauss_move}
    if kdes:
        _move_by_name['kde'] = _KDEIndependenceMove(
            kdes, cov, param_lo_arr, param_hi_arr,
            conditioner=conditioner, cond_kdes=cond_kdes)
        _default_mix = 'de:0.10,gauss:0.25,kde:0.65'
    else:
        _default_mix = 'de:0.6,gauss:0.4'
    _mix = _parse_move_mix(os.environ.get('EXOPLANETLAB_RJ_INMODEL_MIX'), _default_mix)
    in_model_moves = [(_move_by_name[n], w) for n, w in _mix if n in _move_by_name]
    if not in_model_moves:
        in_model_moves = [(gauss_move, 1.0)]

    # ── Sampler ──────────────────────────────────────────────────────────────
    # Parallel tempering: hot chains accept birth/death moves that the cold
    # chain would reject and swap the dimension change down, so N is no longer
    # pinned by the warm start.  Tmax=inf makes the hottest chain sample the
    # prior.  Set EXOPLANETLAB_RJ_NTEMPS=1 to disable.
    _tempering_kwargs = (dict(ntemps=ntemps, Tmax=np.inf, adaptive=True)
                         if ntemps > 1 else {})
    sampler = EnsembleSampler(
        nwalkers,
        {'model': ndim},
        log_likelihood,
        priors,
        branch_names=['model'],
        nleaves_max={'model': n_max},
        nleaves_min={'model': n_min},
        args=args,
        moves=in_model_moves,
        rj_moves=[(rj_move, 1.0)],
        tempering_kwargs=_tempering_kwargs,
        pool=pool,
    )

    # ── Initial state ─────────────────────────────────────────────────────────
    # Seed positions for every known source from the peel best-fit; which of
    # those leaves start ACTIVE depends on `start_mode`:
    #   npeel  — all walkers start at the peel count (pure warm start)
    #   nmin   — all walkers start at n_min: the birth channel must climb to
    #            the true N (the strongest trans-dimensional test, but the
    #            N-posterior is wrong if births never fire)
    #   mixed  — a quarter of the walkers start at n_min, the rest at the peel
    #            count (default): the n_min walkers probe whether the birth
    #            channel works without putting the MAP at risk
    n_peel = len(peel_best) if peel_best else 1
    n_start_lo = max(n_min, 1)

    coords_gw = np.zeros((ntemps, nwalkers, n_max, ndim))
    inds_gw   = np.zeros((ntemps, nwalkers, n_max), dtype=bool)
    if start_mode == 'npeel':
        inds_gw[:, :, :n_peel] = True
    elif start_mode == 'nmin':
        inds_gw[:, :, :n_start_lo] = True
    else:  # mixed
        _lo_n = max(1, nwalkers // 4)
        inds_gw[:, :_lo_n, :n_start_lo] = True
        inds_gw[:, _lo_n:, :n_peel]     = True
    print(f"  Initial N per walker: start_mode='{start_mode}' "
          f"(n_min={n_min}, peel count={n_peel}); {ntemps} temperature(s).")

    for j in range(n_peel):
        pb    = np.asarray(peel_best[j], dtype=float)
        noise = 0.005 * stds * np.random.randn(ntemps, nwalkers, ndim)
        coords_gw[:, :, j, :] = pb[np.newaxis, np.newaxis, :] + noise

    # Inactive slots beyond the peel count: random draws from uniform prior bounds
    for j in range(n_peel, n_max):
        u = np.random.uniform(0.0, 1.0, (ntemps, nwalkers, ndim))
        coords_gw[:, :, j, :] = (param_lo_arr
                                  + u * (param_hi_arr - param_lo_arr))

    # Clip all coords to prior bounds
    coords_gw = np.clip(
        coords_gw,
        param_lo_arr[np.newaxis, np.newaxis, np.newaxis, :],
        param_hi_arr[np.newaxis, np.newaxis, np.newaxis, :],
    )

    state = State({'model': coords_gw}, inds={'model': inds_gw})

    # ── Burn-in + production ─────────────────────────────────────────────────
    print(f"Running RJMCMC: {nwalkers} walkers, "
          f"{ndim} params/source, N ∈ [{n_min}, {n_max}], {n_pool} workers")
    print(f"Burn-in: {nsteps_burn} steps (adaptive proposal tuning), "
          f"production: {nsteps_prod} steps (fixed proposal)")
    _mix_str = ', '.join(f"{n} {w:.2f}" for n, w in _mix if n in _move_by_name)
    print(f"In-model moves: {_mix_str}"
          + (f"  ({len(kdes)} source KDE(s))" if kdes else ""))
    if ntemps > 1:
        print(f"Parallel tempering: {ntemps} temperatures "
              f"(~{ntemps}x the single-chain cost).")
    print("----------------------------------------------------------------------")
    try:
        # Two-stage run: tune the Gaussian move's step size to a target
        # in-model acceptance during (unstored) burn-in, then freeze it for
        # the stored production phase. Tuning is driven manually in chunks
        # (see _PeriodicPerSourceMove.rescale docstring for why eryn's own
        # tune=True cannot be used with the RJ birth/death move here).
        if nsteps_burn > 0:
            n_chunks   = 10
            chunk_size = max(1, nsteps_burn // n_chunks)
            remaining  = nsteps_burn
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
                gauss_move.accepted, gauss_move.num_proposals = \
                    np.zeros((ntemps, nwalkers)), 0
                state = sampler.run_mcmc(
                    state, this_chunk, store=False, progress=False)
                rate = (np.mean(np.asarray(gauss_move.accepted, dtype=float), axis=1)
                        / max(gauss_move.num_proposals, 1))
                gauss_move.rescale(rate)
                remaining -= this_chunk
            _tf = gauss_move._temp_factor
            print(f"  Burn-in done. Gaussian proposal step factor per temperature "
                  f"(cold first): {', '.join(f'{x:.3f}' for x in _tf)}")

        # Reset acceptance counters so reported diagnostics reflect only the
        # (tuned, stationary) production phase, not the adapting burn-in.
        for mv, _ in in_model_moves:
            mv.accepted, mv.num_proposals = np.zeros((ntemps, nwalkers)), 0
        rj_move.accepted, rj_move.num_proposals = np.zeros((ntemps, nwalkers)), 0

        sampler.run_mcmc(state, nsteps_prod, progress=True)
    except KeyboardInterrupt:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    if pool is not None:
        pool.close()
        pool.join()

    # ── Extract results ────────────────────────────────────────────────────────
    chain_coords = sampler.get_chain()['model']    # (nsteps, ntemps, nwalkers, n_max, ndim)
    chain_inds   = sampler.get_inds()['model']     # (nsteps, ntemps, nwalkers, n_max)
    n_chain_full = sampler.get_nleaves()['model']  # (nsteps, ntemps, nwalkers)
    chain_logl   = sampler.get_log_like()       # (nsteps, ntemps, nwalkers)

    # Squeeze the ntemps=1 axis
    n_chain    = n_chain_full[:, 0, :]   # (nsteps_prod, nwalkers)
    logl_chain = chain_logl[:, 0, :]     # (nsteps_prod, nwalkers)

    # Posterior distribution over N
    n_flat         = n_chain.flatten().astype(int)
    n_distribution = np.bincount(n_flat, minlength=n_max + 1)
    most_probable_n = int(np.argmax(n_distribution))

    # Collect flat samples at most_probable_n, sorted by frequency within each
    # step so that source indices are consistently ordered across samples.
    samples_list = []
    logl_list    = []
    for step in range(nsteps_prod):
        for w in range(nwalkers):
            if n_chain[step, w] == most_probable_n:
                active = chain_inds[step, 0, w]             # (n_max,) bool
                leaf_p = chain_coords[step, 0, w, active]   # (n_active, ndim)
                # Sort by frequency (column 0 = f, the first fitted parameter)
                order  = np.argsort(leaf_p[:, 0])
                samples_list.append(leaf_p[order].flatten())
                logl_list.append(logl_chain[step, w])

    if samples_list:
        samples     = np.array(samples_list)    # (n_flat, most_probable_n * ndim)
        log_probs   = np.array(logl_list)
        best_params = samples[np.argmax(log_probs)]
    else:
        # Fallback: no samples at most_probable_n (should not happen)
        samples     = np.zeros((0, most_probable_n * ndim))
        log_probs   = np.zeros(0)
        best_params = np.zeros(most_probable_n * ndim)

    # Acceptance fractions (cold chain only when tempering)
    def _cold_mean(x):
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        return float(np.mean(x[0])) if x.ndim > 1 else float(np.mean(x))

    af_inmodel = _cold_mean(sampler.acceptance_fraction)
    af_rj      = _cold_mean(sampler.rj_acceptance_fraction)
    rj_tau, rj_tau_means, rj_tau_all, rj_tau_walkers = _cold_chain_tau(chain_coords, chain_inds, n_chain,
                                                           most_probable_n)

    # Acceptance of each in-model move on the cold chain (production only).
    # The overall figure hides a move that is never accepted.
    for _mv, _w in in_model_moves:
        try:
            _acc = np.asarray(_mv.accepted, dtype=float)
            _np = max(int(_mv.num_proposals), 1)
            print(f"  Cold-chain acceptance of {type(_mv).__name__.lstrip('_'):24s}"
                  f" (weight {_w:.2f}): {float(np.mean(_acc[0])) / _np:.3f}")
        except Exception:
            pass

    # Absolute birth/death counts (cold chain, production only — the counters
    # were reset after burn-in).  The fraction rounds to 0.000 even when the
    # channel is active, because one accepted jump seeds a long N-excursion.
    try:
        _rj_acc  = np.asarray(rj_move.accepted, dtype=float)
        rj_n_acc = int(np.sum(_rj_acc[0] if _rj_acc.ndim > 1 else _rj_acc))
        rj_n_prop = int(rj_move.num_proposals) * nwalkers
    except Exception:
        rj_n_acc, rj_n_prop = None, None

    return SimpleNamespace(
        samples                = samples,
        log_probs              = log_probs,
        best_params            = best_params,
        chain                  = None,
        chain_burn             = None,
        logprob_burn           = None,
        acceptance_fraction    = af_inmodel,
        rj_acceptance_fraction = af_rj,
        tau                    = None,
        tau_err                = None,
        nsteps_prod            = nsteps_prod,
        logz                   = None,
        logzerr                = None,
        n_chain                = n_chain,
        most_probable_n        = most_probable_n,
        n_distribution         = n_distribution,
        rj_accepted_count      = rj_n_acc,
        rj_proposed_count      = rj_n_prop,
        rj_tau                 = rj_tau,
        rj_tau_means           = rj_tau_means,
        rj_tau_all             = rj_tau_all,
        rj_tau_walkers         = rj_tau_walkers,
    )


def _cold_chain_tau(chain_coords, chain_inds, n_chain, n_mp):
    """Autocorrelation time of the cold chain at n_mp planets, three ways.

    Uses the walkers that hold n_mp planets for at least 99% of the stored
    steps, with the planets ordered by period as in the reported samples;
    during a short excursion to another number of planets (these come from
    swaps with the hotter chains) the last value at n_mp is repeated.  T0 is
    folded to one orbital cycle.

    1. emcee's estimator, averaged over walkers.  It subtracts the mean of
       each walker, so a variation slower than the chain is not seen.
    2. From the scatter of the walker means: for independent walkers of
       nsteps steps, Var(walker mean) = Var(pooled) * tau / nsteps.  This
       one sees slow variations, but with W walkers its relative error is
       about sqrt(2/(W-1)).

    3. From the series of the mean over the walkers at each step.  The
       walkers are not independent (the differential-evolution move builds a
       step from other walkers, and all walkers share the proposals), and a
       variation common to them is seen in neither 1 nor 2.  With m the
       series of the mean over the W walkers and tau_m its autocorrelation
       time, the variance of the mean of all samples is Var(m) tau_m / nsteps,
       which corresponds to tau_all = W Var(m) tau_m / Var(pooled) steps per
       walker.  This is the one that gives the error of a reported value.

    Returns (tau_emcee, tau_means, tau_all, number of walkers used) or
    (None, None, None, 0).
    EXOPLANETLAB_RJ_SAVE_CHAIN=<file.npy> saves the chain used.
    """
    try:
        from emcee.autocorr import integrated_time
    except ImportError:
        return None, None, None, 0
    nsteps = n_chain.shape[0]
    at_n = (n_chain == n_mp)
    keep = np.flatnonzero(at_n.mean(axis=0) >= 0.99)
    if len(keep) < 2 or nsteps < 100 or n_mp < 1:
        return None, None, None, 0
    ndim = chain_coords.shape[-1]
    x = np.empty((nsteps, len(keep), n_mp * ndim))
    for i, w in enumerate(keep):
        first = int(np.argmax(at_n[:, w]))
        last = None
        for step in list(range(first, nsteps)) + list(range(first - 1, -1, -1)):
            if at_n[step, w]:
                leaf_p = chain_coords[step, 0, w, chain_inds[step, 0, w]]
                last = leaf_p[np.argsort(leaf_p[:, 0])].ravel()
            x[step, i] = last
            if step == nsteps - 1:
                last = x[first, i]
    # T0 is defined modulo P, and a walker can sit one period away from the
    # others.  The reported T0 is folded, so fold it here as well; otherwise
    # the jump of a whole period dominates its variance.
    for k in range(n_mp):
        period = np.exp(x[:, :, k * ndim])
        t0 = x[:, :, k * ndim + 2]
        t0 -= np.round((t0 - np.median(t0)) / period) * period
    _path = os.environ.get('EXOPLANETLAB_RJ_SAVE_CHAIN', '').strip()
    if _path:
        np.save(_path, x)
    tau, tau_means, tau_all = _tau_from_chain(x)
    return tau, tau_means, tau_all, (len(keep) if tau is not None else 0)


def _tau_from_chain(x):
    """The three estimates of _cold_chain_tau for x of shape (nsteps, W, npar)."""
    from emcee.autocorr import integrated_time
    nsteps, nw = x.shape[:2]
    try:
        tau = integrated_time(x, tol=0, quiet=True)
        m = x.mean(axis=1)
        tau_m = integrated_time(m[:, None, :], tol=0, quiet=True)
    except Exception:
        return None, None, None
    pooled = x.reshape(-1, x.shape[-1]).var(axis=0)
    pooled = np.where(pooled > 0, pooled, 1.0)
    tau_means = nsteps * x.mean(axis=0).var(axis=0, ddof=1) / pooled
    tau_all = nw * m.var(axis=0) * tau_m / pooled
    return tau, tau_means, tau_all


def _print_tau(result, active_names):
    """Autocorrelation time and effective number of samples of the RJ chain."""
    tau = getattr(result, 'rj_tau', None)
    if tau is None:
        return
    tau_m, tau_a = result.rj_tau_means, result.rj_tau_all
    nw = result.rj_tau_walkers
    nst = result.nsteps_prod
    nsamp = len(result.samples)
    names = list(active_names) if active_names is not None else []
    print(f"  Autocorrelation time of the cold chain at N = {result.most_probable_n} "
          f"({nw} walkers, {nst} steps), in steps:")
    print("    parameter     per walker   walker means   all walkers together")
    for i, t in enumerate(tau):
        nm = names[i] if i < len(names) else f"par {i}"
        print(f"    {nm:10s}   {t:10.1f}   {tau_m[i]:12.1f}   {tau_a[i]:12.1f}")
    t1, t3 = float(np.nanmax(tau)), float(np.nanmax(tau_a))
    print(f"  Largest: {t1:.1f} steps per walker; {t3:.1f} steps for all walkers together "
          f"(the walkers are not independent).")
    print(f"  Effective samples: {nsamp / t3:.0f} of {nsamp}; a reported median has a "
          f"sampling error of about {1.25 * np.sqrt(t3 / nsamp):.3f} sigma.")
    if nst < 50.0 * max(t1, t3):
        print("  WARNING: the chain is shorter than 50 autocorrelation times; "
              "the estimates above are not reliable.")


def print_diagnostics(result, active_names):
    """Print RJMCMC diagnostics to stdout."""
    print("\n--- RJMCMC diagnostics ---")
    total = result.n_distribution.sum()
    print("  Posterior over number of sources N:")
    for n, count in enumerate(result.n_distribution):
        frac   = count / total if total > 0 else 0.0
        marker = '  ← MAP' if n == result.most_probable_n else ''
        print(f"    N = {n}: {count:7d} samples  ({frac * 100:5.1f}%){marker}")
    print(f"  Most probable N         : {result.most_probable_n}")
    print(f"  Samples at most prob. N : {len(result.samples)}")
    if result.acceptance_fraction is not None:
        print(f"  In-model acceptance     : {result.acceptance_fraction:.3f}")
    else:
        print("  In-model acceptance     : N/A (RJ run skipped)")
    if result.rj_acceptance_fraction is not None:
        _cnt = getattr(result, 'rj_accepted_count', None)
        _tot = getattr(result, 'rj_proposed_count', None)
        _extra = (f"  ({_cnt} / {_tot} accepted)"
                  if _cnt is not None and _tot else "")
        print(f"  Birth/death acceptance  : "
              f"{result.rj_acceptance_fraction:.4f}{_extra}")
    _print_tau(result, active_names)
    if result.logz is not None:
        print(f"  Log evidence (peel sum) : {result.logz:.2f}")
    else:
        print("  Note: Eryn does not compute a log-evidence; use the N-posterior")
        print("        occupation fractions for model comparison.")
