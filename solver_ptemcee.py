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
Parallel-tempered, fixed-dimension EnsembleSampler for the joint
(astrometry [+ RV]) fit, built on eryn.

Why this exists: combining Gaia epoch astrometry with RV data taken years
later (often several orbital cycles after the astrometry baseline ends)
adds a real period/phase degeneracy on top of the astrometry-only problem
-- slightly different (P, T0) can be nearly indistinguishable in the
astrometry alone but project to very different RV phases several cycles
later, so the joint posterior can be genuinely multimodal. Plain emcee's
stretch move mixes slowly between well-separated modes; hot tempered
chains can cross the low-probability valley between them and feed a
corrected distribution down to the cold (beta=1) chain, which is what
gets reported.

Reuses eryn rather than adding ptemcee (emcee's own PTSampler was removed
in emcee 3, and ptemcee isn't a dependency here) -- eryn is already a hard
dependency via solver_rjmcmc.py and its tempering is proven on this exact
codebase. This module strips out all of solver_rjmcmc.py's trans-
dimensional machinery (birth/death moves, per-source proposals, variable
leaf count): the branch here always has exactly one active leaf
(nleaves_min = nleaves_max = 1, no rj_moves), which turns eryn into a
plain fixed-dimension parallel-tempered ensemble sampler operating on the
SAME flat parameter vector (and the SAME log_likelihood_multi /
log_prior_multi functions, unmodified) that solver_emcee.py samples
without tempering.

A few eryn API specifics worth recording (found by direct testing, not
documented clearly in eryn itself):
  - With nleaves_min = nleaves_max = 1, eryn calls log_likelihood(x, *args)
    with x already a flat (ndim,) array -- NOT wrapped in a leaf axis like
    the (n_active, ndim) convention solver_rjmcmc.py's log_like_rj uses for
    a variable number of active leaves. So ExoPlanetLab.py's
    log_likelihood_multi/log_prior_multi can be passed straight through,
    with no adapter beyond the prior object wrapper below.
  - sampler.reset() forwards **kwargs to the backend without re-supplying
    branch_names, and raises if called bare -- avoided entirely here by
    running burn-in and production as one continuous stored chain and
    slicing the split afterward, rather than resetting between phases.
  - Temperature index 0 is the coldest (beta=1, full posterior); the last
    index is beta=0 (pure prior) when Tmax=inf.
  - EnsembleSampler.get_autocorr_time() unconditionally raises when
    ntemps > 1 ("not well-defined for number of temperatures > 1") -- tau
    is computed here by hand from the cold chain alone via
    emcee.autocorr.integrated_time, the same function plain emcee's own
    get_autocorr_time() calls internally.
  - eryn.prior.log_uniform(min, max) constructs
    scipy.stats.loguniform(min, max - min), which looks like a bug (second
    positional arg to scipy's loguniform is the upper bound, not a width) --
    not used here; a Jeffreys/log-uniform prior is unnecessary in this
    module since the *_CallablePrior wrapper below just reuses
    ExoPlanetLab.py's own log_prior_multi, which already implements it
    correctly (see K_MIN/JITTER_MIN comments there).

Public API mirrors solver_emcee.py's run() call signature and return
SimpleNamespace shape, so it is a drop-in swap at the ExoPlanetLab.py
joint-fit call site.
"""

import os
import numpy as np
if not hasattr(np, "in1d"):
    # eryn 1.2.6 (latest on PyPI) still calls np.in1d, removed in NumPy 2.0.
    np.in1d = np.isin
from types import SimpleNamespace

import emcee.autocorr
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import StretchMove
from eryn.moves.red_blue import RedBlueMove

from config import N_POOL, limit_blas_threads
from solver_emcee import (ADAPTIVE_STOP, ADAPTIVE_NTAU, ADAPTIVE_TAU_RTOL,   # noqa: F401
                          adaptive_stop_check, VecPool, VECTORIZE_WORKERS)
from solver_emcee import LogCoords, InPhysical, results_to_physical
from solver_emcee import print_diagnostics, check_initial_spread   # noqa: F401 -- re-exported; same
                                              # SimpleNamespace fields (tau,
                                              # tau_err, nsteps_prod), no need
                                              # to duplicate the convergence
                                              # warning logic

NTEMPS = int(os.environ.get('EXOPLANETLAB_PT_NTEMPS', '4'))
# Production steps between adaptive-stop checks (see run()); 0 = the default,
# nsteps_prod // 25. The tau estimate of slowly mixing chains keeps growing
# with their length, so frequent checks can stop a run while it is still low.
PT_ADAPTIVE_CHECK = int(os.environ.get('EXOPLANETLAB_PT_ADAPTIVE_CHECK', '0'))

# Evidence mode (run(..., evidence=True)): more temperatures, the ladder
# frozen at the end of burn-in, and ln Z from the production chains of all
# temperatures by stepping-stone sampling (thermodynamic integration printed as a
# cross-check; eryn's Backend.get_evidence_estimate). Both estimate
# ln[Z(beta=1) / Z(beta=0)], i.e. the evidence for the NORMALISED prior, so an
# unnormalised log_prior (log-uniform or isotropic factors without their
# constants) is fine. The ladder runs to beta = 0 (Tmax = inf).
NTEMPS_EVIDENCE = int(os.environ.get('EXOPLANETLAB_EVIDENCE_NTEMPS', '16'))
# ln Z is the bridge (Bennett acceptance ratio) estimate over the rungs, which
# uses the samples of both temperatures of a rung; the one-sided stepping-stone
# estimates are printed with it (forward: biased low, reverse: biased high, when
# a rung is under-sampled). An evidence run stops adaptively only when, besides
# the tau rule, ln Z of the two halves of the production chain agree within
# EXOPLANETLAB_EVIDENCE_TOL: the temperatures at a sharp transition (a strong
# signal switching on) equilibrate far more slowly than the cold chain, whose
# tau the rule looks at. A ladder frozen at the end of the burn-in may then
# have been adapted to chains that were not in equilibrium, and have a gap at
# the transition. So
# when the halves disagree at a check, the ladder is released to adapt again
# for another burn-in length and the evidence restarts from there; the cold
# chain, and with it the posterior, is unaffected (swaps preserve it) and keeps
# every production step. If the halves still differ at the end, ln Z is taken
# from the second half and the result is flagged (logz_stable = False). With
# EXOPLANETLAB_ADAPTIVE_STOP=0 the run has a fixed length and ladder.
EVIDENCE_TOL = float(os.environ.get('EXOPLANETLAB_EVIDENCE_TOL', '0.5'))
# EXOPLANETLAB_EVIDENCE_DUMP=<file.npz>: also save the production log-likelihoods
# of all temperatures (float32) and the ladder, to study the estimators offline.
EVIDENCE_DUMP = os.environ.get('EXOPLANETLAB_EVIDENCE_DUMP', '').strip()

# Stored-chain thinning. eryn stores every step of EVERY temperature (chain,
# log-likelihood, log-prior), ~8 (ndim + 3) bytes per walker-step: 58 GB for
# 32 temperatures x 64 walkers x 210 000 steps x 17 parameters, more than the
# memory of most machines. 'auto' (default) stores every
# T-th step, T the smallest that keeps the store under PT_STORE_MAX_GB; an
# integer sets T. All step counts (burn-in, production, the adaptive-stop
# checks, nsteps_prod) and tau stay in real steps; only the stored chain and
# log-likelihoods are thinned (the stepping-stone evidence uses every stored
# step, thinning by T << tau loses nothing). The ladder adapts per step either
# way (eryn's tempering counts proposals).
PT_THIN = os.environ.get('EXOPLANETLAB_PT_THIN', 'auto').strip().lower()

# Differential-evolution share of the moves (EXOPLANETLAB_PT_DE, 0-1; default
# 0.5: the same posteriors and ln Z as the stretch move alone, with tau 1.4-31x
# lower; 0 = stretch move only). DE proposes x + gamma (x_a - x_b) with x_a,
# x_b from the complementary half of the walkers at the same temperature
# (emcee's DEMove on eryn's red-blue split), gamma = 2.38 / sqrt(2 ndim), and
# gamma = 1 a fraction PT_DE_GAMMA1 of the time (jumps between modes that the
# ensemble already spans). It steps along the strongly correlated directions
# of the orbital posterior. Symmetric: no Hastings factor.
PT_DE = float(os.environ.get('EXOPLANETLAB_PT_DE', '0.5'))
PT_DE_GAMMA1 = float(os.environ.get('EXOPLANETLAB_PT_DE_GAMMA1', '0.1'))


class DEMove(RedBlueMove):
    """Differential-evolution move on eryn's red-blue split (see PT_DE)."""

    def __init__(self, p_gamma1=0.1, sigma=1.0e-5, **kwargs):
        RedBlueMove.__init__(self, **kwargs)
        self.p_gamma1 = float(p_gamma1)
        self.sigma = float(sigma)

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        newpos, ntemps, Ns = {}, None, None
        for name in s_all:
            s = np.asarray(s_all[name])
            c = np.concatenate([np.asarray(ci) for ci in c_all[name]], axis=1)
            ntemps, Ns, nleaves, ndim = s.shape
            Nc = c.shape[1]
            gamma0 = 2.38 / np.sqrt(2.0 * nleaves * ndim)
            ia = random.randint(Nc, size=(ntemps, Ns))
            ib = (ia + random.randint(1, Nc, size=(ntemps, Ns))) % Nc   # b != a
            t = np.arange(ntemps)[:, None]
            diff = c[t, ia] - c[t, ib]                                  # (ntemps, Ns, nleaves, ndim)
            g = gamma0 * (1.0 + self.sigma * random.randn(ntemps, Ns))
            g = np.where(random.rand(ntemps, Ns) < self.p_gamma1, 1.0, g)
            newpos[name] = s + g[:, :, None, None] * diff
        return newpos, np.zeros((ntemps, Ns))
PT_STORE_MAX_GB = float(os.environ.get('EXOPLANETLAB_PT_STORE_MAX_GB', '2'))


def _store_thin(nsteps_total, ntemps, nwalkers, ndim):
    if PT_THIN != 'auto':
        return max(1, int(PT_THIN))
    gb = nsteps_total * ntemps * nwalkers * (ndim + 3) * 8 / 1e9
    return max(1, int(np.ceil(gb / PT_STORE_MAX_GB)))


def evidence_estimates(betas, logls, block=50, repeats=200, seed=0):
    """ln Z from tempered chains with a fixed ladder.

    betas : (ntemps,) inverse temperatures, including 0 (the prior).
    logls : (nsteps, ntemps, nwalkers) log-likelihoods of the production chains.
    Returns (lnZ_ss, err_ss, lnZ_ti, err_ti):
      stepping-stone: sum_k ln < exp((beta_{k+1} - beta_k) ln L) >_{beta_k}, the
        mean over all samples of the lower temperature, with a block bootstrap
        over steps for the error;
      thermodynamic integration: trapezoid of <ln L>_beta over beta, error = full
        ladder minus every second rung.
    Written here rather than taken from eryn: its stepping-stone reshapes the
    (steps, temps, walkers) array without moving the walker axis, mixing
    temperatures, and its integration calls np.trapz, removed in NumPy 2."""
    from scipy.special import logsumexp
    o = np.argsort(betas)
    b = np.asarray(betas, float)[o]
    L = np.asarray(logls, float)[:, o, :]                 # (steps, temps, walkers)
    L = np.where(np.isfinite(L), L, -np.inf)
    db = np.diff(b)

    def ss(Lsub):
        # samples of temperature k (all steps and walkers) for each rung k -> k+1
        tot = 0.0
        for k in range(len(db)):
            x = db[k] * Lsub[:, k, :].ravel()
            tot += logsumexp(x) - np.log(x.size)
        return tot

    lnz_ss = ss(L)
    rng = np.random.default_rng(seed)
    n = L.shape[0]
    nb = max(1, n // block)
    reps = []
    for _ in range(repeats):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        idx = (starts[:, None] + np.arange(min(block, n))[None, :]).ravel()
        reps.append(ss(L[idx]))
    err_ss = float(np.std(reps))
    m = np.nanmean(np.where(np.isfinite(L), L, np.nan), axis=(0, 2))
    lnz_ti = float(np.trapezoid(m, b))
    err_ti = float(abs(lnz_ti - np.trapezoid(m[::2], b[::2]))) if len(b) > 2 else np.nan
    return float(lnz_ss), err_ss, lnz_ti, err_ti


def _rung_terms(betas, logls):
    """Sorted ladder: db (ntemps - 1,) and ln L as (steps, temps, walkers)."""
    o = np.argsort(betas)
    L = np.asarray(logls, float)[:, o, :]
    return np.diff(np.asarray(betas, float)[o]), np.where(np.isfinite(L), L, -np.inf), \
        np.asarray(betas, float)[o]


def _rung_estimates(db, L, k):
    """ln[Z(beta_k+1) / Z(beta_k)] of rung k: (forward, reverse, bridge).

    forward: ln < exp(db ln L) > over the samples of the lower beta (biased low
    when under-sampled); reverse: -ln < exp(-db ln L) > over those of the upper
    beta (biased high); bridge: the Bennett acceptance ratio, which uses both
    and is solved between them. Where the two agree, the bridge is their mean."""
    from scipy.optimize import brentq
    from scipy.special import expit, logsumexp
    u0 = db[k] * L[:, k, :].ravel()
    u1 = db[k] * L[:, k + 1, :].ravel()
    fwd = float(logsumexp(u0) - np.log(u0.size))
    rev = float(-(logsumexp(-u1) - np.log(u1.size)))
    if abs(rev - fwd) < 1e-3:
        return fwd, rev, 0.5 * (fwd + rev)
    m = np.log(u0.size / u1.size)

    def g(c):
        return np.sum(expit(u0 - c - m)) - np.sum(expit(c + m - u1))
    lo, hi = min(fwd, rev) - 1.0, max(fwd, rev) + 1.0
    while g(lo) < 0.0:
        lo -= 5.0
    while g(hi) > 0.0:
        hi += 5.0
    return fwd, rev, float(brentq(g, lo, hi, xtol=1e-4))


def evidence_bridge(betas, logls, nblocks=8):
    """ln Z by bridge sampling over the rungs (see _rung_estimates), and its
    error from the scatter of nblocks consecutive parts of the chain,
    sd / sqrt(nblocks) -- which also grows when the chains drift.
    Returns (lnZ, err, block values)."""
    db, L, _ = _rung_terms(betas, logls)

    def tot(Lsub):
        return float(sum(_rung_estimates(db, Lsub, k)[2] for k in range(len(db))))
    n = L.shape[0]
    edges = np.linspace(0, n, nblocks + 1).astype(int)
    blocks = [tot(L[a:e]) for a, e in zip(edges[:-1], edges[1:]) if e > a]
    err = float(np.std(blocks, ddof=1) / np.sqrt(len(blocks))) if len(blocks) > 1 else np.nan
    return tot(L), err, blocks


def evidence_halves(betas, logls):
    """Bridge ln Z of the first and of the second half of the chain."""
    db, L, _ = _rung_terms(betas, logls)
    h = L.shape[0] // 2
    return tuple(float(sum(_rung_estimates(db, part, k)[2] for k in range(len(db))))
                 for part in (L[:h], L[h:]))


def evidence_diagnostics(betas, logls):
    """The one-sided estimates behind a bridge ln Z: dict(forward, reverse,
    rungs = [(beta_lo, beta_hi, forward, reverse, bridge)])."""
    db, L, b = _rung_terms(betas, logls)
    rungs = [(float(b[k]), float(b[k + 1])) + _rung_estimates(db, L, k) for k in range(len(db))]
    return dict(forward=float(sum(r[2] for r in rungs)),
                reverse=float(sum(r[3] for r in rungs)), rungs=rungs)


class _CallablePrior:
    """Wraps a plain log_prior(params) -> float callable as an
    eryn-compatible prior object (needs .logpdf, .rvs, .key_order).

    Reuses whatever prior function is passed in (ExoPlanetLab.py's
    log_prior_multi) unchanged, rather than reconstructing an equivalent
    prior from eryn's own distribution classes -- one fewer place for the
    Jeffreys-on-P, Jeffreys-on-jitter, Gaussian-on-M_star/parallax, and
    bounds logic to drift out of sync with the plain (non-tempered) emcee
    path.
    """

    def __init__(self, log_prior, ndim, lo=None, hi=None, log_prior_vec=None):
        self._log_prior = log_prior
        self._log_prior_vec = log_prior_vec
        self.ndim = ndim
        self._lo = np.asarray(lo, dtype=float) if lo is not None else None
        self._hi = np.asarray(hi, dtype=float) if hi is not None else None
        self.key_order = list(range(ndim))

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(self._log_prior(x))
        if self._log_prior_vec is not None:
            return np.asarray(self._log_prior_vec(x), dtype=float)
        return np.array([self._log_prior(row) for row in x])

    def rvs(self, size=1):
        # Only exercised if eryn needs a fresh draw outside the initial
        # state supplied in run() below -- with no rj_moves and a fixed
        # leaf count, this shouldn't normally fire; a wide uniform
        # fallback keeps it from crashing if it ever does.
        if self._lo is None or self._hi is None:
            raise NotImplementedError(
                "_CallablePrior.rvs() needs param_lo/param_hi (see run()'s "
                "param_lo/param_hi arguments) -- not provided.")
        if isinstance(size, int):
            size = (size,)
        u = np.random.uniform(0.0, 1.0, size + (self.ndim,))
        return self._lo + u * (self._hi - self._lo)


def run(log_likelihood, log_prior, prior_transform, args, p0=None, ndim=None,
        nsteps_burn=4000, nsteps_prod=60000, n_pool=N_POOL,
        ntemps=None, param_lo=None, param_hi=None,
        log_likelihood_vec=None, log_prior_vec=None, log_idx=None, evidence=False):
    """Run a parallel-tempered, fixed-dimension eryn EnsembleSampler.

    Same call signature and return SimpleNamespace shape as
    solver_emcee.run -- see that docstring for field meanings. Differences:
    only the cold (beta=1) chain is returned in `chain`/`chain_burn`/
    `samples`/etc. (the hotter chains exist purely to improve the cold
    chain's mixing across modes); `logz`/`logzerr` stay None unless
    evidence=True (see NTEMPS_EVIDENCE), which also returns `logz_ti`.

    prior_transform is accepted but unused (kept for interface parity with
    solver_emcee.run/solver_nessai.run).

    param_lo, param_hi : optional (ndim,) arrays, only used as a fallback
        bound for the prior wrapper's rvs() -- see _CallablePrior. Safe to
        omit; rvs() just isn't available then, which is fine as long as
        it's never called (expected, given no rj_moves).

    log_idx : sequence of int or None -- columns sampled as their natural log
        (see solver_emcee.LogCoords). The log-Jacobian is added to the PRIOR,
        which is not tempered, so every temperature targets prior x
        likelihood^beta of the same physical model. p0 and everything returned
        are in physical units; only tau refers to the log coordinates.
    """
    nwalkers, _ndim = p0.shape
    check_initial_spread(p0)

    _coords = LogCoords(log_idx) if log_idx is not None and len(log_idx) else None
    if _coords is not None:
        p0 = _coords.to_samp(p0)
        log_prior = InPhysical(log_prior, _coords, add_jac=True)
        log_likelihood = InPhysical(log_likelihood, _coords, add_jac=False)
        if log_prior_vec is not None:
            log_prior_vec = InPhysical(log_prior_vec, _coords, add_jac=True)
        if log_likelihood_vec is not None:
            log_likelihood_vec = InPhysical(log_likelihood_vec, _coords, add_jac=False)
        if param_lo is not None and param_hi is not None:
            param_lo = np.array(param_lo, dtype=float)
            param_hi = np.array(param_hi, dtype=float)
            param_lo[_coords.idx] = np.log(np.maximum(param_lo[_coords.idx], 1e-300))
            param_hi[_coords.idx] = np.log(np.maximum(param_hi[_coords.idx], 1e-300))
        print(f"  Sampling {len(_coords.idx)} parameter(s) in log coordinates "
              f"(columns {_coords.idx.tolist()}).")
    if ndim is None:
        ndim = _ndim
    ntemps = ntemps or (NTEMPS_EVIDENCE if evidence else NTEMPS)

    # Vectorised path (log_likelihood_vec / log_prior_vec given): eryn passes
    # all walkers of all temperatures as one (k, ndim) array, evaluated in a
    # single process -- see ExoPlanetLab.py's VECTORIZE note.
    _vec = log_likelihood_vec is not None and log_prior_vec is not None
    pool = None
    if n_pool > 1 and not _vec:
        import multiprocessing as _mp
        if _mp.get_start_method(allow_none=True) != 'fork':
            _mp.set_start_method('fork', force=True)
        pool = _mp.Pool(n_pool, initializer=limit_blas_threads)

    _vp = VecPool(log_likelihood_vec, VECTORIZE_WORKERS) if _vec else None
    prior  = _CallablePrior(log_prior, ndim, param_lo, param_hi,
                            log_prior_vec=log_prior_vec if _vec else None)
    priors = {'model': prior}

    sampler = EnsembleSampler(
        nwalkers, {'model': ndim}, _vp if _vec else log_likelihood, priors,
        branch_names=['model'], nleaves_max={'model': 1}, nleaves_min={'model': 1},
        args=args, moves=([(StretchMove(), 1.0)] if PT_DE <= 0.0 else
                          [(StretchMove(), 1.0 - PT_DE), (DEMove(PT_DE_GAMMA1), PT_DE)]),
        tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf, adaptive=True,
                              **({'stop_adaptation': nsteps_burn} if evidence else {})),
        pool=pool, vectorize=_vec,
    )

    # Seed every temperature from the same p0 (built the same way
    # solver_emcee.make_p0 does), with a little extra jitter per
    # temperature so the ladder doesn't start bit-identical.
    coords = np.broadcast_to(p0, (ntemps, nwalkers, ndim)).copy()
    _spread = 1.0e-3 * (np.std(p0, axis=0) + 1.0e-12)
    coords += _spread * np.random.randn(ntemps, nwalkers, ndim)

    # p0 itself is already valid (the caller -- ExoPlanetLab.py --
    # clips every parameter to its prior's support before calling run()),
    # but the jitter just added is not re-clipped, and for a parameter
    # whose warm-start value sits near a hard prior edge (e.g. eccentricity
    # near its 0 floor, RV jitter near its Jeffreys floor, K/gamma near
    # their bounds, M_star/parallax near 0) that jitter alone can push a
    # coordinate outside the prior's support. eryn's sample() hard-fails
    # ("The initial log_prior was +/- infinite") on ANY -inf anywhere in
    # the initial ensemble, across every temperature and walker, so this
    # has to be checked exhaustively, not just on the cold chain -- any
    # offending (temperature, walker) just falls back to its own
    # unperturbed p0 row rather than the perturbed one.
    _lp0  = prior.logpdf(coords.reshape(-1, ndim)).reshape(ntemps, nwalkers)
    _bad  = ~np.isfinite(_lp0)
    if np.any(_bad):
        _bad_t, _bad_w = np.nonzero(_bad)
        coords[_bad_t, _bad_w, :] = p0[_bad_w]
        print(f"  Note: {_bad.sum()} of {ntemps * nwalkers} tempered initial "
              f"walkers had a perturbed-out-of-bounds coordinate (near a "
              f"hard prior edge); reset to the unperturbed seed.")

    # The unperturbed seed rows can themselves be outside the prior's support:
    # e.g. a warm start that ended at the prior's edge (an undetected weak
    # signal whose best fit slid to the a0 floor of the isotropic prior). Plain emcee tolerates such walkers (any finite proposal is
    # accepted), eryn refuses to start. Replace each by a copy of a valid walker,
    # nudged slightly (and re-checked); stop only if no walker is valid.
    _lp0 = prior.logpdf(coords.reshape(-1, ndim)).reshape(ntemps, nwalkers)
    _bad = ~np.isfinite(_lp0)
    if np.any(_bad):
        _flat = coords.reshape(-1, ndim)
        _good_rows = np.nonzero(np.isfinite(_lp0.reshape(-1)))[0]
        if _good_rows.size == 0:
            raise ValueError("solver_ptemcee: no initial walker lies inside the prior's "
                             "support -- check the starting point (p0).")
        _rng0 = np.random.default_rng(12345)
        _scale = 1.0e-6 * (np.std(_flat[_good_rows], axis=0) + 1.0e-12)
        for _r in np.nonzero(_bad.reshape(-1))[0]:
            _src = _flat[_rng0.choice(_good_rows)]
            _new = _src + _scale * _rng0.standard_normal(ndim)
            if not np.isfinite(prior.logpdf(_new[None, :])[0]):
                _new = _src.copy()
            _flat[_r] = _new
        coords = _flat.reshape(ntemps, nwalkers, ndim)
        print(f"  Note: {_bad.sum()} of {ntemps * nwalkers} initial walkers were outside "
              f"the prior's support even unperturbed (a start at a prior edge); replaced "
              f"by copies of valid walkers.")

    coords = coords[:, :, np.newaxis, :]              # add the nleaves=1 axis
    inds = {'model': np.ones((ntemps, nwalkers, 1), dtype=bool)}
    state = State({'model': coords}, inds=inds)

    # T: store every T-th step (_store_thin). Burn-in and production are
    # rounded up to multiples of T; _burn_s / _k are in stored steps.
    T = _store_thin(nsteps_burn + nsteps_prod, ntemps, nwalkers, ndim)
    _burn_s = -(-nsteps_burn // T)
    nsteps_burn = _burn_s * T
    nsteps_prod = -(-nsteps_prod // T) * T
    nsteps_total = nsteps_burn + nsteps_prod
    print(f"Running PT-MCMC: {nwalkers} walkers x {ntemps} temperatures, "
          f"{nsteps_burn} burn-in + {nsteps_prod} production steps, "
          + (f"vectorised over walkers, in up to {_vp.n} parallel blocks" if _vec else f"{n_pool} workers"))
    if PT_DE > 0.0:
        print(f"(moves: stretch {1.0 - PT_DE:.2f}, differential evolution {PT_DE:.2f}; "
              f"EXOPLANETLAB_PT_DE)")
    print("(burn-in and production are one continuous stored run here, "
          "split afterward -- see this module's reset() note)")
    if T > 1:
        print(f"(every {T}th step stored: all temperatures unthinned would take "
              f"{nsteps_total * ntemps * nwalkers * (ndim + 3) * 8 / 1e9:.0f} GB; "
              f"EXOPLANETLAB_PT_THIN / _PT_STORE_MAX_GB)")
    print("----------------------------------------------------------------------")

    _every = max(1, nsteps_total // 20)
    # Adaptive stopping of the production phase on the cold chain -- same rule
    # as solver_emcee (adaptive_stop_check), checked every PT_ADAPTIVE_CHECK
    # steps (default nsteps_prod // 25; see PT_ADAPTIVE_CHECK). The cold chain
    # is read as a view of eryn's storage (no copy of the hotter chains, which
    # get_chain() made: up to ~2 GB per check).
    _check = PT_ADAPTIVE_CHECK if PT_ADAPTIVE_CHECK > 0 else max(2000, nsteps_prod // 25)
    _check = max(T, _check // T * T)
    _tau_prev = None        # tau estimates in REAL steps
    _tau_hist = []          # (production steps, tau) of every computed check
    _n_done = 0
    _ev_start = nsteps_burn  # real step from which the ladder is final (evidence)
    _readapt = nsteps_burn   # length of the next re-adaptation; doubles each time,
                             # so a slow transition (HD 114762) gets a few long
                             # adaptations rather than many short ones
    for _k, _ in enumerate(sampler.sample(state, iterations=nsteps_total // T,
                                           thin_by=T, store=True), 1):
        _step = _k * T                       # real steps done
        _n_done = _step
        if _step == nsteps_burn:
            print("Burn-in done. Production...", flush=True)
        if _step // _every > (_step - T) // _every:
            _phase = 'burn-in ' if _step <= nsteps_burn else 'production'
            print(f"  {_phase} {100*_step//nsteps_total:3d}%  "
                  f"({_step}/{nsteps_total})", flush=True)
        _n_prod = _step - nsteps_burn
        if (ADAPTIVE_STOP and _n_prod > 0 and _n_prod % _check == 0
                and _step < nsteps_total):
            # Evidence mode: ln Z must be stable over the steps with the final
            # ladder (see EVIDENCE_TOL); if it is not, the ladder adapts again.
            # Checked at every check, before and independently of the tau rule:
            # a run whose cold chain never passes the tau rule
            # must still get its ladder re-adapted.
            _ev_ok = True
            if evidence:
                _ev_ok = False
                _bk = sampler.backend
                if _step - _ev_start >= _check:
                    try:
                        _h1, _h2 = evidence_halves(
                            sampler.temperature_control.betas,
                            _bk.log_like[_ev_start // T:_bk.iteration])
                        _ev_ok = abs(_h2 - _h1) < EVIDENCE_TOL
                        if not _ev_ok and nsteps_total - _step >= _readapt + 2 * _check:
                            _tc = sampler.temperature_control
                            # its clock in units of real steps (counts adaptations)
                            _tc.stop_adaptation = _tc.time + int(round(
                                _readapt * _tc.time / _step))
                            _ev_start = _step + _readapt
                            print(f"  ln Z not stable at {_n_prod} production steps (halves "
                                  f"{_h1:.2f}, {_h2:.2f}): the ladder adapts for another "
                                  f"{_readapt} steps, the evidence restarts.", flush=True)
                            _readapt *= 2
                        elif not _ev_ok:
                            print(f"  ln Z not stable at {_n_prod} production steps (halves "
                                  f"{_h1:.2f}, {_h2:.2f}); continuing.", flush=True)
                    except Exception as _ex:
                        _ev_ok = False
            _known = _tau_prev is not None and np.all(np.isfinite(_tau_prev))
            # Skip checks that cannot stop the run (chain still shorter than
            # 90% of NTAU x the last tau estimate); estimate tau from a chain
            # thinned by ~5% of the smallest previous tau (cost ~1/thin).
            if _known and _n_prod < 0.9 * ADAPTIVE_NTAU * np.max(_tau_prev):
                continue
            _thin = max(1, int(0.05 * np.min(_tau_prev) / T)) if _known else 1
            # Stability is judged against the estimate from a chain at most 80%
            # as long, not the previous check's: tau estimates grow with chain
            # length while the chain is too short, and with frequent checks two
            # consecutive estimates would agree even while still biased low
            # (a synthetic test stopped at 49 true tau that way; with this
            # reference it stops above 60 true tau).
            _ref = next((tv for (nv, tv) in reversed(_tau_hist) if nv <= 0.8 * _n_prod), None)
            try:
                _bk = sampler.backend
                _cold = _bk.chain['model'][_burn_s:_bk.iteration, 0, :, 0, :]
                # in stored steps inside adaptive_stop_check, real steps outside
                _stop, _tau_s = adaptive_stop_check(
                    _cold, _n_prod // T, None if _ref is None else np.asarray(_ref) / T,
                    thin=_thin)
                _tau_prev = T * np.asarray(_tau_s, dtype=float)
                _tau_hist.append((_n_prod, _tau_prev))
            except Exception as _ex:
                _stop = False
            if _stop and evidence and not _ev_ok:
                _stop = False
            if _stop:
                print(f"  Adaptive stop at {_n_prod} production steps: "
                      f"> {ADAPTIVE_NTAU:g} x max tau ({np.max(_tau_prev):.0f}), "
                      f"max tau stable to {100*ADAPTIVE_TAU_RTOL:g}% "
                      f"(limit was {nsteps_prod}).", flush=True)
                break
    nsteps_prod = max(0, _n_done - nsteps_burn)

    if pool is not None:
        pool.close()
        pool.join()
    if _vp is not None:
        _vp.close()

    # Cold chain only (temperature index 0, beta=1) -- see module docstring.
    # Copied out of eryn's store (get_chain() would copy every temperature).
    _bk = sampler.backend
    chain_full = np.array(_bk.chain['model'][:_bk.iteration, 0, :, 0, :])  # (stored, nwalkers, ndim)
    logl_full  = np.array(_bk.log_like[:_bk.iteration, 0, :])               # (stored, nwalkers)

    chain_burn   = chain_full[:_burn_s]
    logl_burn    = logl_full[:_burn_s]
    chain        = chain_full[_burn_s:]
    logl_prod    = logl_full[_burn_s:]

    logprior_burn = prior.logpdf(chain_burn.reshape(-1, ndim)).reshape(len(chain_burn), nwalkers)
    logprob_burn  = logl_burn + logprior_burn

    samples   = chain.reshape(-1, ndim)
    logl_flat = logl_prod.reshape(-1)
    logprior_flat = prior.logpdf(samples)
    log_probs = logl_flat + logprior_flat   # combined log-posterior, matching
                                             # solver_emcee's get_log_prob() semantics

    print(f"\nAcceptance fraction (cold chain): "
          f"{np.mean(sampler.acceptance_fraction[0]):.3f}")
    print("Acceptance fraction by temperature: "
          + ', '.join(f"T{t}={np.mean(sampler.acceptance_fraction[t]):.3f}"
                       for t in range(ntemps)))
    _betas = sampler.temperature_control.betas
    print("Temperature ladder (beta=1/T): "
          + ', '.join(f"{b:.4g}" for b in _betas))

    logz = logzerr = logz_ti = logzerr_ti = logz_blocks = None
    logz_forward = logz_reverse = logz_stable = None
    if evidence:
        try:
            # the steps with the final ladder: all production, or what follows
            # the last re-adaptation
            _bb = sampler.backend.get_betas(discard=_burn_s)
            _chg = np.nonzero(np.any(_bb != _bb[-1], axis=1))[0]
            _s0 = _burn_s + (int(_chg[-1]) + 1 if _chg.size else 0)
            _b = _bb[-1:]
            _ll = sampler.backend.get_log_like(discard=_s0).astype(float)
            if len(_ll) < 200:
                raise ValueError('too few steps with the final temperature ladder')
            if _s0 > _burn_s:
                print(f"Evidence from the last {len(_ll) * T} production steps (the ladder "
                      f"was re-adapted until step {(_s0 - _burn_s) * T}).")
            _, _, logz_ti, logzerr_ti = evidence_estimates(_b[0], _ll, repeats=1)
            _h1, _h2 = evidence_halves(_b[0], _ll)
            logz_stable = bool(abs(_h2 - _h1) < EVIDENCE_TOL)
            # stable: the whole production chain; not: its second half only
            _use = _ll if logz_stable else _ll[len(_ll) // 2:]
            logz, logzerr, logz_blocks = evidence_bridge(_b[0], _use)
            _dg = evidence_diagnostics(_b[0], _use)
            logz_forward, logz_reverse = _dg['forward'], _dg['reverse']
            print(f"Log evidence ({ntemps} temperatures, {nsteps_prod} production steps): "
                  f"ln Z = {logz:.2f} +/- {logzerr:.2f} (bridge; stepping-stone forward "
                  f"{logz_forward:.2f}, reverse {logz_reverse:.2f}; thermodynamic "
                  f"integration {logz_ti:.2f})", flush=True)
            print(f"  ln Z of the two halves of the production chain: {_h1:.2f}, {_h2:.2f}"
                  + ("" if logz_stable else
                     f"  -- NOT STABLE (differ by more than {EVIDENCE_TOL:g}): ln Z above is "
                     f"from the second half; run longer"))
            _worst = max(_dg['rungs'], key=lambda r: abs(r[3] - r[2]))
            if abs(_worst[3] - _worst[2]) > 0.1:
                print(f"  least overlapping rung: beta {_worst[0]:.4g} -> {_worst[1]:.4g}, "
                      f"forward {_worst[2]:.2f}, reverse {_worst[3]:.2f}, bridge "
                      f"{_worst[4]:.2f}", flush=True)
            if EVIDENCE_DUMP:
                np.savez_compressed(EVIDENCE_DUMP, betas=_b[0], logl=_ll.astype(np.float32),
                                    thin=T, nsteps_prod=nsteps_prod)
                print(f"  (production log-likelihoods of all temperatures saved to "
                      f"{EVIDENCE_DUMP})")
        except Exception as err:
            print(f"WARNING: evidence estimate failed: {err!r}", flush=True)

    try:
        # eryn's own get_autocorr_time() unconditionally raises for
        # ntemps > 1 -- compute it by hand on the cold chain only, the same
        # function plain emcee's own get_autocorr_time() calls internally.
        tau     = T * emcee.autocorr.integrated_time(chain, quiet=False)   # real steps
        tau_err = None
    except Exception as err:
        tau     = None
        tau_err = str(err)

    res = SimpleNamespace(
        samples=samples,
        log_probs=log_probs,
        best_params=samples[np.argmax(log_probs)],
        chain=chain,
        chain_burn=chain_burn,
        logprob_burn=logprob_burn,
        acceptance_fraction=float(np.mean(sampler.acceptance_fraction[0])),
        tau=tau,
        tau_err=tau_err,
        nsteps_prod=nsteps_prod,
        logz=logz,
        logzerr=logzerr,
        logz_ti=logz_ti,
        logzerr_ti=logzerr_ti,
        logz_blocks=logz_blocks,
        logz_forward=logz_forward,
        logz_reverse=logz_reverse,
        logz_stable=logz_stable,
    )
    return results_to_physical(res, _coords) if _coords is not None else res
