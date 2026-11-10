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
emcee EnsembleSampler backend for parameter estimation.

Public API
----------
make_p0(initial_guess, param_lo, param_hi, fit_idx, ...)
    Spread walkers around an initial guess and clip to prior bounds.

run(log_likelihood, log_prior, prior_transform, args, p0, ...) -> SimpleNamespace
    Run burn-in + production MCMC.

print_diagnostics(result, active_names)
    Print convergence summary to stdout.
"""

import os
import numpy as np
import emcee
import emcee.autocorr
from types import SimpleNamespace
from config import N_POOL, limit_blas_threads


class _LogPosterior:
    """Picklable log-posterior; local closures can't be sent through pool.map."""

    def __init__(self, log_prior, log_likelihood):
        self._lp = log_prior
        self._ll = log_likelihood

    def __call__(self, params, *args):
        lp = self._lp(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._ll(params, *args)

class LogCoords:
    """Sample some strictly positive parameters as their natural logarithm.

    The caller's parameter vector stays in physical units everywhere; only the
    sampler sees x[idx] = ln(p[idx]). A density in the new coordinates picks up
    the Jacobian |dp/dx| = prod(p[idx]), i.e. + sum(x[idx]) in log space, so a
    log-uniform (Jeffreys) prior 1/p becomes flat in x and the posterior itself
    is unchanged. Used for the RV and astrometric jitters, whose log-uniform
    priors span several decades: in linear units the single-point instruments'
    (almost pure-prior) jitter distributions are extremely skewed, which the
    affine-invariant stretch move handles badly (long autocorrelation times).
    """

    # exp() argument limit: coordinates outside the prior's support are
    # rejected by the prior anyway; this only avoids overflow warnings.
    _XMAX = 700.0

    def __init__(self, idx):
        self.idx = np.asarray(sorted(set(int(i) for i in idx)), dtype=int)

    def to_phys(self, x):
        p = np.array(x, dtype=float, copy=True)
        p[..., self.idx] = np.exp(np.clip(p[..., self.idx], -self._XMAX, self._XMAX))
        return p

    def to_samp(self, p):
        x = np.array(p, dtype=float, copy=True)
        if np.any(x[..., self.idx] <= 0.0):
            raise ValueError("LogCoords: a log-sampled parameter is <= 0 "
                             f"(columns {self.idx.tolist()}).")
        x[..., self.idx] = np.log(x[..., self.idx])
        return x

    def log_jac(self, x):
        """ln |dp/dx| = sum of the log-sampled coordinates."""
        return np.sum(np.asarray(x, dtype=float)[..., self.idx], axis=-1)


class InPhysical:
    """Picklable wrapper: evaluate f (defined on physical parameters) at
    sampler coordinates x, optionally adding the log-Jacobian (the Jacobian
    belongs with the prior: it must not be tempered in solver_ptemcee)."""

    def __init__(self, f, coords, add_jac):
        self.f = f
        self.coords = coords
        self.add_jac = add_jac

    def __call__(self, x, *args):
        v = self.f(self.coords.to_phys(x), *args)
        if self.add_jac:
            v = np.asarray(v, dtype=float) + self.coords.log_jac(x)
            if v.ndim == 0:
                v = float(v)
        return v


def results_to_physical(res, coords, log_prob_fields=('log_probs', 'logprob_burn'),
                        chain_fields=('samples', 'chain', 'chain_burn')):
    """Convert a solver result sampled in LogCoords back to physical units.

    Chains/samples are transformed; log-probabilities have the Jacobian removed
    so they are the physical-coordinate log posterior, as without LogCoords;
    best_params is re-taken as the maximum of that. The autocorrelation times
    are left as measured, i.e. in the sampler's (log) coordinates, which is
    what describes its mixing."""
    for f in log_prob_fields:
        lp = getattr(res, f, None)
        ch = getattr(res, 'samples' if f == 'log_probs' else 'chain_burn', None)
        if lp is None or ch is None:
            continue
        setattr(res, f, np.asarray(lp) - coords.log_jac(ch).reshape(np.shape(lp)))
    for f in chain_fields:
        arr = getattr(res, f, None)
        if arr is not None:
            setattr(res, f, coords.to_phys(arr))
    if getattr(res, 'samples', None) is not None and getattr(res, 'log_probs', None) is not None:
        res.best_params = res.samples[np.argmax(res.log_probs)]
    res.log_coords_idx = coords.idx
    return res


# Default MCMC tuning parameters
NWALKERS    = 64
EPSILON     = 1.0e-4   # Gaussian spread scale applied to initial guess
NSTEPS_BURN = 2000
NSTEPS_PROD = 25000

# Adaptive stopping of the production phase (also used by solver_ptemcee): stop
# once the chain is longer than ADAPTIVE_NTAU x the largest integrated
# autocorrelation time AND that largest tau has changed by less than
# ADAPTIVE_TAU_RTOL since the previous check (the emcee documentation's
# recipe: 100 tau, a wide margin above the 50-tau convergence criterion). The
# configured number of production steps remains the upper limit.
# EXOPLANETLAB_ADAPTIVE_STOP=0 gives fixed-length runs.
#
# Only the largest tau has to be stable, to 5%: it alone sets the required
# length (with many parameters some tau always moves by more).
ADAPTIVE_STOP      = os.environ.get('EXOPLANETLAB_ADAPTIVE_STOP', '1') == '1'
# Target length 100 tau (emcee's documentation): the tau
# estimates of the real RV chains keep growing with chain length (Gaia-4 + RV:
# 266 at 16 000 steps, 594 at 200 000), so a target close to the 50-tau
# criterion can be met while the estimate is still low.
ADAPTIVE_NTAU      = float(os.environ.get('EXOPLANETLAB_ADAPTIVE_NTAU', '100'))
ADAPTIVE_TAU_RTOL  = float(os.environ.get('EXOPLANETLAB_ADAPTIVE_TAU_RTOL', '0.05'))
ADAPTIVE_CHECK_MIN = 500     # steps between checks (at least)


# Vectorised likelihood evaluated in parallel blocks: the walkers are split
# into one block per worker, and each worker evaluates its block with the
# vectorised function -- efficient array arithmetic and only one
# inter-process message per worker per call (instead of one per walker).
# EXOPLANETLAB_VECTORIZE_WORKERS: number of workers (default N_POOL); 1 = a
# single process, the better choice when several sources run side by side.
VECTORIZE_WORKERS = int(os.environ.get('EXOPLANETLAB_VECTORIZE_WORKERS', str(N_POOL)))
# Split only if every block gets at least this many walkers: below that the
# inter-process overhead outweighs the gain (measured on Gaia BH3: emcee's 32-walker
# half-ensembles are fastest in one process, 2.3 ms/step against 3.5 in 9 blocks;
# the tempered fit's 128-walker calls are fastest in blocks, 5.2 against 7.6).
# EXOPLANETLAB_VEC_MIN_ROWS overrides.
VEC_MIN_ROWS_PER_BLOCK = int(os.environ.get('EXOPLANETLAB_VEC_MIN_ROWS', '12'))


def _vec_block(task):
    f, X, args = task
    return f(X, *args)


class VecPool:
    """Callable f(X, *args) -> (k,) that splits X over a worker pool.

    `f` must be picklable by reference (a module-level function; with the fork
    start method a function defined in the running script's __main__ works)."""

    def __init__(self, f, n_workers):
        import multiprocessing as _mp
        if _mp.get_start_method(allow_none=True) != 'fork':
            _mp.set_start_method('fork', force=True)
        self.f = f
        self.n = max(1, int(n_workers))
        # The pool is created lazily, on the first call large enough to be split.
        self.pool = None
        self._mp = _mp

    def __call__(self, X, *args):
        # as many blocks as have VEC_MIN_ROWS_PER_BLOCK rows each, up to n
        nb = min(self.n, X.shape[0] // max(1, VEC_MIN_ROWS_PER_BLOCK))
        if nb <= 1:
            return np.asarray(self.f(X, *args), dtype=float)
        if self.pool is None:
            self.pool = self._mp.Pool(self.n, initializer=limit_blas_threads)
        blocks = np.array_split(X, nb)
        return np.concatenate(self.pool.map(_vec_block, [(self.f, B, args) for B in blocks]))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def adaptive_stop_check(chain, n_steps, tau_prev, thin=1):
    """(stop?, tau) for a production chain of shape (n_steps, nwalkers, ndim).

    thin > 1 estimates tau from every thin-th step (tau in the thinned chain,
    times thin); valid for thin well below tau and much cheaper for long chains
    (solver_ptemcee uses thin ~ 5% of the smallest previous tau)."""
    try:
        thin = max(1, int(thin))
        tau = thin * emcee.autocorr.integrated_time(chain[::thin], tol=0)
    except Exception:
        return False, tau_prev
    if not np.all(np.isfinite(tau)):
        return False, tau
    stop = (tau_prev is not None and np.all(np.isfinite(tau_prev))
            and n_steps > ADAPTIVE_NTAU * np.max(tau)
            and abs(np.max(tau) - np.max(tau_prev)) < ADAPTIVE_TAU_RTOL * np.max(tau))
    return stop, tau


# Parallel likelihood evaluations across walkers.  Leave 1 core for the main
# thread.  Uses 'fork' start method so workers inherit module-level arrays
# (t_obs, data) without re-running the script.



def check_initial_spread(p0, names=None):
    """Raise if any walker dimension has zero spread in the initial ensemble.

    An ensemble stretch move only explores the span of the current ensemble,
    so a dimension in which every walker starts at the same value can never
    move (typically because a start value was clipped onto a prior bound).
    That parameter is then silently *not sampled* -- it sits at its start
    value, its autocorrelation time comes out nan, and every other parameter
    is fit conditional on it. Fail loudly instead.
    """
    p0 = np.asarray(p0, dtype=float)
    frozen = np.where(np.ptp(p0, axis=0) == 0.0)[0]
    if len(frozen):
        labels = [names[j] if names is not None and j < len(names) else f'#{j}'
                  for j in frozen]
        raise ValueError(
            "Initial walker ensemble has zero spread in dimension(s) "
            f"{labels} (all walkers identical, e.g. a start value clipped onto "
            "a prior bound) -- the stretch move can never move these, so they "
            "would silently not be sampled. Start them strictly inside their "
            "bounds.")


def make_p0(initial_guess, param_lo, param_hi, fit_idx,
            nwalkers=NWALKERS, epsilon=EPSILON):
    """Spread walkers around initial_guess and clip to prior bounds.

    Parameters
    ----------
    initial_guess : ndarray  shape (ndim,) — central starting point
    param_lo      : list     lower prior bound per global parameter index
    param_hi      : list     upper prior bound per global parameter index
    fit_idx       : list     global parameter indices of the active parameters
    nwalkers      : int      number of walkers; raised to max(2*ndim+2, nwalkers)
    epsilon       : float    Gaussian spread scale applied to initial_guess

    Returns
    -------
    p0 : ndarray  shape (nwalkers, ndim)
    """
    ndim     = len(initial_guess)
    nwalkers = max(2 * ndim + 2, nwalkers)
    p0       = initial_guess + epsilon * np.random.randn(nwalkers, ndim)
    for j, i in enumerate(fit_idx):
        p0[:, j] = np.clip(p0[:, j], param_lo[i], param_hi[i])
    return p0


def run(log_likelihood, log_prior, prior_transform, args, p0=None, ndim=None,
        nsteps_burn=NSTEPS_BURN, nsteps_prod=NSTEPS_PROD, n_pool=N_POOL,
        log_posterior_vec=None, log_idx=None):
    """Run emcee burn-in + production MCMC.

    Parameters
    ----------
    log_likelihood  : callable  signature log_likelihood(params, *args)
    log_prior       : callable  signature log_prior(params) -> float
    prior_transform : callable  not used; included for interface consistency
    args            : tuple     extra arguments forwarded to log_likelihood
    p0              : ndarray   shape (nwalkers, ndim) — initial positions
    nsteps_burn     : int       number of burn-in steps
    nsteps_prod     : int       number of production steps
    n_pool          : int       worker processes for parallel likelihood evaluation
    log_posterior_vec : callable or None  signature f(X, *args) -> (k,) log
                      posterior for a (k, ndim) array of walkers. If given, it
                      is used instead of log_prior + log_likelihood, with
                      emcee's vectorize=True in a single process (n_pool is
                      then not used).
    log_idx         : sequence of int or None  columns (strictly positive
                      parameters) to sample as their natural log -- see
                      LogCoords. p0, the functions and everything returned stay
                      in physical units; only tau refers to the log coordinates.

    Returns
    -------
    SimpleNamespace with fields:
        samples              (nsteps_prod * nwalkers, ndim) — flat chain
        log_probs            (nsteps_prod * nwalkers,)
        best_params          ndarray — maximum a posteriori sample
        chain                (nsteps_prod, nwalkers, ndim)
        chain_burn           (nsteps_burn, nwalkers, ndim)
        logprob_burn         (nsteps_burn, nwalkers)
        acceptance_fraction  float — mean over walkers
        tau                  ndarray or None — autocorrelation times
        tau_err              str or None — error message if tau failed
        nsteps_prod          int
        logz                 None
        logzerr              None
    """
    import multiprocessing as _mp

    nwalkers, ndim = p0.shape
    check_initial_spread(p0)

    _coords = LogCoords(log_idx) if log_idx is not None and len(log_idx) else None
    if _coords is not None:
        p0 = _coords.to_samp(p0)
        log_prior = InPhysical(log_prior, _coords, add_jac=True)
        log_likelihood = InPhysical(log_likelihood, _coords, add_jac=False)
        if log_posterior_vec is not None:
            log_posterior_vec = InPhysical(log_posterior_vec, _coords, add_jac=True)
        print(f"  Sampling {len(_coords.idx)} parameter(s) in log coordinates "
              f"(columns {_coords.idx.tolist()}).")

    _log_posterior = _LogPosterior(log_prior, log_likelihood)

    pool = None
    _vp = None
    if log_posterior_vec is not None:
        _vp = VecPool(log_posterior_vec, VECTORIZE_WORKERS)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _vp, args=args, vectorize=True)
        _how = "vectorised over walkers"
    else:
        if n_pool > 1:
            if _mp.get_start_method(allow_none=True) != 'fork':
                _mp.set_start_method('fork', force=True)
            pool = _mp.Pool(n_pool, initializer=limit_blas_threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_posterior,
                                         args=args, pool=pool)
        _how = f"{n_pool} workers"

    print(f"Running MCMC: {nwalkers} walkers, "
          f"{nsteps_burn} burn-in + {nsteps_prod} production steps, {_how}")
    print("----------------------------------------------------------------------")

    print("Burn-in...")
    state = sampler.run_mcmc(p0, nsteps_burn, progress=True)
    print(flush=True)   # newline after tqdm bar so "Production..." starts cleanly

    chain_burn   = sampler.get_chain()      # (nsteps_burn, nwalkers, ndim)
    logprob_burn = sampler.get_log_prob()   # (nsteps_burn, nwalkers)

    # Carry the final burn-in state directly into production.
    # Re-collapsing walkers around the best position kills StretchMove acceptance.
    p1 = state.coords
    sampler.reset()

    print("Production...", flush=True)
    _every = max(1, nsteps_prod // 20)
    _check = max(ADAPTIVE_CHECK_MIN, nsteps_prod // 50)
    _tau_prev = None
    nsteps_done = 0
    for _step, _ in enumerate(sampler.sample(p1, iterations=nsteps_prod), 1):
        nsteps_done = _step
        if _step % _every == 0:
            print(f"  {100*_step//nsteps_prod:3d}%  ({_step}/{nsteps_prod})",
                  flush=True)
        if ADAPTIVE_STOP and _step % _check == 0 and _step < nsteps_prod:
            _stop, _tau_prev = adaptive_stop_check(
                sampler.get_chain(), _step, _tau_prev)
            if _stop:
                print(f"  Adaptive stop at {_step} production steps: "
                      f"> {ADAPTIVE_NTAU:g} x max tau ({np.max(_tau_prev):.0f}), "
                      f"max tau stable to {100*ADAPTIVE_TAU_RTOL:g}% "
                      f"(limit was {nsteps_prod}).", flush=True)
                break
    nsteps_prod = nsteps_done

    if pool is not None:
        pool.close()
        pool.join()
    if _vp is not None:
        _vp.close()

    samples   = sampler.get_chain(flat=True)    # (nsteps_prod * nwalkers, ndim)
    log_probs = sampler.get_log_prob(flat=True)
    chain     = sampler.get_chain()              # (nsteps_prod, nwalkers, ndim)

    print(f"\nAcceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    try:
        tau     = sampler.get_autocorr_time()
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
        acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        tau=tau,
        tau_err=tau_err,
        nsteps_prod=nsteps_prod,
        logz=None,
        logzerr=None,
    )
    return results_to_physical(res, _coords) if _coords is not None else res


def print_diagnostics(result, active_names):
    """Print autocorrelation convergence summary to stdout."""
    _acc = getattr(result, 'acceptance_fraction', None)
    if _acc is not None and float(_acc) < 0.05:
        print("\n  " + "!" * 60)
        print(f"  FROZEN CHAIN: cold-chain acceptance fraction is {float(_acc):.3f}.")
        print("  Walkers are barely moving, so tau and the posterior widths")
        print("  below describe the starting point, not the posterior (a")
        print("  frozen chain also produces nan/huge tau and implausibly")
        print("  tight credible intervals). Usual cause: a pathological")
        print("  starting point (e.g. eccentricity pinned near its upper")
        print("  bound with enormous Thiele-Innes constants) -- check the")
        print("  warm-start best-fit printed above.")
        print("  " + "!" * 60)
    print("\n--- Autocorrelation times ---")
    if result.tau is None:
        print("  " + "!" * 60)
        print("  NOT CONVERGED -- emcee could not even reliably estimate tau")
        print("  (the chain is too short relative to it). The raw estimates")
        print("  below are UNRELIABLE and likely optimistic -- true tau may")
        print("  be longer. Treat the corner/burn-in/trace plots from this")
        print("  run with real suspicion: apparent bimodality, traces that")
        print("  don't settle, or posteriors that look non-Gaussian are the")
        print("  expected symptom, not necessarily a separate problem.")
        print("  Fix: run longer (more production steps), or if the corner")
        print("  plot shows multiple distinct, well-separated modes, that")
        print("  may be a real degeneracy in the data that more steps alone")
        print("  won't resolve (e.g. RV epochs many orbital cycles away from")
        print("  the astrometry baseline) -- consider parallel tempering.")
        print("  " + "!" * 60)
        print(f"  Raw (unreliable) estimate: {result.tau_err}")
        return

    tau         = result.tau
    nsteps_prod = result.nsteps_prod
    print(f"  {'Parameter':12s}  {'tau (steps)':>12s}  {'Need >':>10s}  Status")
    print("  " + "-" * 50)
    for name, t in zip(active_names, tau):
        need   = 50 * t
        status = "OK" if nsteps_prod > need else "NOT CONVERGED"
        print(f"  {name:12s}  {t:12.1f}  {need:10.0f}  {status}")
    # np.max propagates nan when a parameter's tau estimate failed. Such a tau
    # is unknown, not small, and may be far longer than every other one, so the
    # run is reported as NOT CONVERGED rather than judged on the finite entries.
    if not np.all(np.isfinite(tau)):
        print(f"\n  Max tau = nan,  production = {nsteps_prod} steps")
        print("  Overall: NOT CONVERGED -- at least one parameter's "
              "autocorrelation time could not be estimated (nan above). "
              "Its true tau is unknown and may be far longer than the "
              "others' -- treat this run's corner/burn-in/trace plots with "
              "the same suspicion as the 'tau is None' case.")
        return
    tau_max = np.max(tau)
    print(f"\n  Max tau = {tau_max:.1f},  production = {nsteps_prod} steps")
    if nsteps_prod > 50 * tau_max:
        print("  Overall: CONVERGED")
    else:
        print(f"  Overall: NOT CONVERGED — need ~{int(50*tau_max)} steps")
