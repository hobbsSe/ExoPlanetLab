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
Single-job dynesty nested sampling for exoplanet Thiele-Innes orbit fitting.

Runs one dynesty nested-sampling job over the full N_PLANETS*ndim_per(+RV)
parameter model.  ExoPlanetLab.py hands it the prior transform to sample --
in practice a flat per-parameter box (~10 sigma wide) centred on an unbiased
emcee joint fit, which reduces the prior volume enough for a single nested run
to converge while staying wide enough not to narrow the posterior.

Parallelism note
----------------
pool.map() requires every callable it sends to workers to be picklable.
Local closures (e.g. the prior-transform lambda) are not.  We store the
transform in a module-level _PT_CACHE dict before forking the pool; forked
workers inherit it, and the picklable _CachedPT(key) proxy looks it up by key.

Public API
----------
run(log_likelihood, log_prior, prior_transform, args, ...) -> SimpleNamespace
    Run one nested-sampling job over prior_transform.

print_diagnostics(result, active_names)
    Print the log-evidence and posterior sample count.
"""

import os
import numpy as np
import dynesty
import dynesty.utils as dyutils
from types import SimpleNamespace

from config import N_POOL

NLIVE   = int(os.environ.get('EXOPLANETLAB_DYNESTY_NLIVE', '500'))
SAMPLE  = 'rslice'   # handles curved ridges better than uniform sampling
# rslice decorrelation steps per proposal. None (default) = 3 x ndim: dynesty's
# fixed 5 is far too few in ~17 dimensions (the RV jitters barely move once the
# live points have contracted onto the orbital posterior, and ln Z comes out
# low). EXOPLANETLAB_DYNESTY_SLICES sets a fixed number.
_SLICES_ENV = os.environ.get('EXOPLANETLAB_DYNESTY_SLICES', '').strip()
SLICES  = int(_SLICES_ENV) if _SLICES_ENV else None

# ── Pickling helpers ───────────────────────────────────────────────────────────
# prior_transform closures can't be pickled by pool.map.  We store them in
# this module-level dict before forking the pool; workers inherit the dict
# via fork and _CachedPT looks up the function by key.

_PT_CACHE = {}


class _CachedPT:
    """Picklable prior-transform proxy; looks up the real closure from _PT_CACHE."""
    def __init__(self, key):
        self.key = key
    def __call__(self, u):
        return _PT_CACHE[self.key](u)


class _FullLogl:
    """Picklable full log-likelihood for dynesty pool.map."""
    def __init__(self, log_likelihood, args):
        self._logl = log_likelihood
        self._args = args
    def __call__(self, p):
        return float(self._logl(p, *self._args))


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _make_pool(n_pool, pt, cache_key):
    """Store pt in _PT_CACHE, then fork the pool so workers inherit it.

    Must be called immediately before the sampler is built so that the cache
    entry exists at fork time.  Returns (pool, queue_size).
    """
    import multiprocessing as _mp
    _PT_CACHE[cache_key] = pt
    if n_pool > 1:
        if _mp.get_start_method(allow_none=True) != 'fork':
            _mp.set_start_method('fork', force=True)
        return _mp.Pool(n_pool), n_pool
    return None, None


def _close_pool(pool, cache_key):
    if pool is not None:
        pool.close()
        pool.join()
    _PT_CACHE.pop(cache_key, None)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run(log_likelihood, log_prior, prior_transform, args,
        p0=None, ndim=None, nlive=NLIVE, sample=SAMPLE, n_pool=N_POOL,
        dlogz=None, slices=SLICES, seed=None, dynamic=False, pfrac=1.0):
    """Run one dynesty nested-sampling job over the supplied prior transform.

    Parameters
    ----------
    log_likelihood  : callable  (params, *args) -> float
    log_prior       : callable  unused (dynesty samples via prior_transform)
    prior_transform : callable  (u_cube) -> params.  Either the full
                      multi-planet prior transform or a peel-derived tight box,
                      chosen by ExoPlanetLab.py.
    args            : tuple     forwarded to log_likelihood
    p0              : ignored (interface consistency with the MCMC solvers)
    ndim            : int       number of parameters
    nlive           : int       live points
    sample          : str       dynesty sampling method ('rslice' recommended)
    n_pool          : int       worker processes for likelihood evaluation
    dlogz           : float or None.  Evidence stopping tolerance; None -> the
                      dynesty default (~0.5 at nlive=500).
    slices          : int or None.  rslice decorrelation steps per proposal;
                      None = 3 x ndim (see SLICES).
    seed            : int or None.  Fixed RNG seed for a reproducible run.
    dynamic         : bool.  If True use DynamicNestedSampler (better posteriors
                      per CPU-second, less accurate evidence).
    pfrac           : float     dynamic posterior/evidence weighting (1.0 = PE).

    Returns
    -------
    SimpleNamespace with the standard solver fields.
    """
    if ndim is None:
        ndim = p0.shape[1]
    if slices is None:
        slices = max(5, 3 * int(ndim))      # see SLICES

    logl     = _FullLogl(log_likelihood, args)
    pool, qs = _make_pool(n_pool, prior_transform, 'main')

    _kw = dict(sample=sample, pool=pool, queue_size=qs)
    if sample in ('rslice', 'slice', 'hslice'):
        _kw['slices'] = slices
    if seed is None:      # from NumPy's global generator (EXOPLANETLAB_SEED seeds it)
        seed = int(np.random.randint(2 ** 31 - 1))
    if seed is not None:
        _kw['rstate'] = np.random.default_rng(int(seed))

    mode = 'dynamic' if dynamic else 'static'
    print(f"\n{'='*66}")
    print(f"  dynesty ({mode}) — single full-dimensional nested job")
    print(f"  {ndim} params | {nlive} live points | sampler='{sample}'"
          f" (slices={slices}) | dlogz={'default' if dlogz is None else dlogz}"
          f"{'' if seed is None else f' | seed={int(seed)}'}"
          f"{f' | pfrac={pfrac}' if dynamic else ''}")
    print(f"{'='*66}")

    try:
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(logl, _CachedPT('main'), ndim, **_kw)
            _run_kw = dict(nlive_init=nlive, nlive_batch=nlive,
                           wt_kwargs={'pfrac': pfrac}, print_progress=True)
            if dlogz is not None:
                _run_kw['dlogz_init'] = dlogz
            sampler.run_nested(**_run_kw)
        else:
            sampler = dynesty.NestedSampler(logl, _CachedPT('main'), ndim,
                                            nlive=nlive, **_kw)
            _run_kw = dict(print_progress=True)
            if dlogz is not None:
                _run_kw['dlogz'] = dlogz
            sampler.run_nested(**_run_kw)
    finally:
        _close_pool(pool, 'main')

    dres    = sampler.results
    weights = np.exp(dres.logwt - dres.logz[-1])
    samples = dyutils.resample_equal(dres.samples, weights)
    best    = dres.samples[np.argmax(dres.logl)]
    logz    = float(dres.logz[-1])
    logzerr = float(dres.logzerr[-1])

    print(f"\n  ln Z = {logz:.2f} ± {logzerr:.2f}   ({len(samples)} posterior samples)")

    return SimpleNamespace(
        samples=samples,
        log_probs=None,
        best_params=best,
        chain=None,
        chain_burn=None,
        logprob_burn=None,
        acceptance_fraction=None,
        tau=None,
        tau_err=None,
        nsteps_prod=None,
        logz=logz,
        logzerr=logzerr,
    )


def print_diagnostics(result, active_names):
    """Print the log-evidence and posterior sample count."""
    print("\n--- dynesty diagnostics ---")
    print(f"  ln Z = {result.logz:10.2f} ± {result.logzerr:.2f}")
    print(f"  Posterior samples: {len(result.samples)}")
