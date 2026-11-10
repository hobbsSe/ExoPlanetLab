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
Nessai (normalizing-flow nested sampling) backend for parameter estimation.

Nessai trains a normalizing flow to learn the shape of the posterior and
generates proposals from the trained network, making it far more efficient
than standard nested sampling for narrow, curved posteriors.  The flow
adapts to the geometry during the run, so no staged warm-up is needed.

Unlike dynesty, nessai writes auxiliary files (flow weights, checkpoints,
result HDF5) to an output directory.  Pass output=os.path.join(outdir,
'nessai_flow') via the _extra_joint dict in ExoPlanetLab.py so these land
alongside the other results.

Public API
----------
run(log_likelihood, log_prior, prior_transform, args, ...) -> SimpleNamespace
    Run the FlowSampler.

print_diagnostics(result, active_names)
    Print evidence and sample count to stdout.
"""

import logging
import os
import sys
import numpy as np
from types import SimpleNamespace

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from config import N_MIN, N_MAX, N_POOL

# Route nessai INFO messages to stdout so progress appears in the terminal
# and is captured by the _Tee log.  Neural-network internals (glasflow, torch)
# stay suppressed at WARNING to avoid excessive output.
class _StdoutProxy:
    """Looks up sys.stdout on every write instead of binding to one object.

    logging.StreamHandler(sys.stdout) would instead snapshot whatever
    _Tee instance is current at import/reload time. ExoPlanetLab.py
    imports this module before installing *its own* run's _Tee, so under
    autoreload the handler always ends up pinned to the *previous* run's
    (now-closed) log file, spamming "Logging error: I/O operation on
    closed file" for the entire run. Proxying through sys.stdout avoids
    binding to any particular _Tee instance.
    """
    def write(self, data):
        sys.stdout.write(data)

    def flush(self):
        sys.stdout.flush()


_nessai_logger = logging.getLogger('nessai')
_nessai_logger.setLevel(logging.INFO)
_nessai_logger.handlers.clear()
_nessai_logger.propagate = False
_handler = logging.StreamHandler(_StdoutProxy())
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter('[nessai] %(message)s'))
_nessai_logger.addHandler(_handler)

logging.getLogger('glasflow').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

NLIVE    = 1000   # winner refinement pass (accurate posteriors)
NLIVE_TD = 500    # survey pass baseline (just needs to rank models)

# Two independent parallelism axes that operate in non-overlapping phases:
#   N_PYTORCH_THREADS — threads for normalising-flow training (PyTorch phase)
#   N_POOL            — worker processes for likelihood evaluation (sampling phase)
# The two phases alternate, so both can claim N_POOL cores without conflict.
N_PYTORCH_THREADS = N_POOL


class _Model(Model):
    """Adapter between log_likelihood / log_prior callables and nessai's Model API.

    nessai works in the physical parameter space directly (not the unit
    hypercube), so we supply physical bounds derived from the prior transform
    and let the flow learn the posterior geometry.
    """

    def __init__(self, log_likelihood_fn, log_prior_fn, prior_transform_fn,
                 ndim, bounds_lo, bounds_hi, args):
        # nessai uses plain string names for its internal structured arrays
        self.names  = [f'p{k}' for k in range(ndim)]
        self.bounds = {n: [float(lo), float(hi)]
                       for n, lo, hi in zip(self.names, bounds_lo, bounds_hi)}
        self._logl = log_likelihood_fn
        self._logp = log_prior_fn
        self._pt   = prior_transform_fn
        self._args = args

    def new_point(self, N=1):
        """Draw N points from the tight prior via prior_transform.

        nessai's default implementation samples uniformly within bounds,
        which spreads initial live points across the full prior volume and
        causes very slow convergence for tight warm-start posteriors.

        We call super() to get a correctly-typed structured array (which
        includes nessai's internal logP, logL, etc. fields), then overwrite
        the parameter values with tight-Gaussian samples from prior_transform
        and recompute logP so nessai's internal bookkeeping stays consistent.
        """
        pts = super().new_point(N=N)
        for i in range(N):
            u = np.random.uniform(0, 1, len(self.names))
            params = self._pt(u)
            for j, name in enumerate(self.names):
                pts[name][i] = float(params[j])
        if 'logP' in pts.dtype.names:
            pts['logP'] = np.array([
                self._logp(np.array([float(pts[n][i]) for n in self.names]))
                for i in range(N)
            ])
        return pts

    def new_point_log_prob(self, x):
        """Log probability of a point under the new_point distribution.

        Required by nessai when new_point is overridden.  Since new_point
        draws from the tight prior via prior_transform, the log prob is the
        tight log_prior density.
        """
        return self.log_prior(x)

    def log_likelihood(self, x):
        if np.ndim(x[self.names[0]]) == 0:      # single point
            params = np.array([float(x[n]) for n in self.names])
            return np.float64(self._logl(params, *self._args))
        return np.array([                         # batch of N points
            self._logl(np.array([float(x[n][i]) for n in self.names]), *self._args)
            for i in range(len(x[self.names[0]]))])

    def log_prior(self, x):
        if np.ndim(x[self.names[0]]) == 0:      # single point
            params = np.array([float(x[n]) for n in self.names])
            return np.float64(self._logp(params))
        return np.array([                         # batch of N points
            self._logp(np.array([float(x[n][i]) for n in self.names]))
            for i in range(len(x[self.names[0]]))])


def run(log_likelihood, log_prior, prior_transform, args,
        p0=None, ndim=None, nlive=NLIVE, output='nessai_flow', n_pool=N_POOL,
        stopping_criterion=0.1, max_iteration=None,
        reparameterisations=None, seed=None):
    """Run nessai FlowSampler.

    Parameters
    ----------
    log_likelihood      : callable  (params, *args) → float
    log_prior           : callable  (params) → float
    prior_transform     : callable  (u) → params  used to derive parameter bounds
    args                : tuple     forwarded to log_likelihood
    p0                  : ndarray   ignored (nessai draws from the prior itself)
    ndim                : int       number of active parameters
    nlive               : int       number of live points
    output              : str       directory for flow weights, checkpoints, results
    reparameterisations : dict or None.  Forwarded to nessai's FlowProposal
                          (e.g. {'p2': {'reparameterisation': 'angle-2pi'}} to
                          treat a wide periodic parameter as periodic so the
                          flow does not have to learn the wrap).
    seed                : int or None.  Fixed RNG seed for reproducible runs.
    stopping_criterion  : float     dlogZ threshold; run stops when estimated
                          remaining log-evidence < this value.  Default 0.1 nat
                          (nessai default).  For model-selection survey runs
                          where only the evidence ranking matters, 1.0–2.0 is
                          sufficient and considerably faster.
    max_iteration       : int or None.  Hard cap on nested-sampling iterations,
                          forwarded to nessai's NestedSampler.  Use this (not a
                          wall-clock/subprocess timeout) to bound a run that
                          may be searching for a signal that isn't really
                          there: it's nessai's own native mechanism, so it
                          can't conflict with nessai's internal SIGALRM/pool
                          handling or macOS's fork-safety issues the way a
                          subprocess-based external timeout can. A run that
                          hits this cap still returns a valid (if less
                          converged) result rather than erroring out --
                          confirmed empirically.

    Returns
    -------
    SimpleNamespace with standard solver fields:
        samples, best_params, logz, logzerr
        chain / chain_burn / tau / etc. are all None
    """
    if ndim is None:
        ndim = p0.shape[1]

    # Derive physical parameter bounds from the edges of the prior transform
    eps       = 1e-9
    bounds_lo = prior_transform(np.full(ndim, eps))
    bounds_hi = prior_transform(np.full(ndim, 1.0 - eps))
    # Guard against zero-width dimensions (nessai's ScaleAndShift divides by
    # hi - lo, which causes NaN for a parameter pinned to a single value).
    _MIN_WIDTH = 1e-8
    for _i in range(ndim):
        if bounds_hi[_i] - bounds_lo[_i] < _MIN_WIDTH:
            _mid = 0.5 * (bounds_lo[_i] + bounds_hi[_i])
            bounds_lo[_i] = _mid - 0.5 * _MIN_WIDTH
            bounds_hi[_i] = _mid + 0.5 * _MIN_WIDTH

    model = _Model(log_likelihood, log_prior, prior_transform,
                     ndim, bounds_lo, bounds_hi, args)

    # Fork so pool workers inherit the non-picklable model closures.
    if n_pool > 1:
        import multiprocessing as _mp
        if _mp.get_start_method(allow_none=True) != 'fork':
            _mp.set_start_method('fork', force=True)

    os.makedirs(output, exist_ok=True)
    _extra_kwargs = {}
    if max_iteration is not None:
        _extra_kwargs['max_iteration'] = max_iteration
    if reparameterisations is not None:
        _extra_kwargs['reparameterisations'] = reparameterisations
    if seed is None:      # from NumPy's global generator (EXOPLANETLAB_SEED seeds it)
        seed = int(np.random.randint(2 ** 31 - 1))
    if seed is not None:
        _extra_kwargs['seed'] = int(seed)
    sampler = FlowSampler(model, output=output, nlive=nlive, resume=False,
                          pytorch_threads=N_PYTORCH_THREADS,
                          n_pool=n_pool if n_pool > 1 else None,
                          parallelise_prior=True if n_pool > 1 else None,
                          stopping=stopping_criterion, **_extra_kwargs)

    print(f"Running nessai: {nlive} live points, {ndim} parameters, "
          f"{N_PYTORCH_THREADS} PyTorch threads, {n_pool} likelihood+prior workers, "
          f"dlogZ<{stopping_criterion}"
          f"{'' if seed is None else f', seed={int(seed)}'}"
          f"{'' if not reparameterisations else f', reparam={list(reparameterisations)}'}")
    print("----------------------------------------------------------------------")
    sampler.run(plot=False)

    # Nessai registers SIGINT/SIGTERM/SIGALRM handlers during FlowSampler.__init__
    # and never restores them.  If another sampler runs after the peel (e.g. Eryn
    # in rjflow), those handlers intercept Ctrl+C and hang trying to close an
    # already-closed pool.  Restore defaults here so the caller gets normal signals.
    #
    # SIGALRM specifically: nessai uses it for periodic checkpointing, armed via
    # signal.alarm(). If that alarm is still pending when this run() call
    # returns (e.g. the checkpoint interval hadn't elapsed yet), it fires LATER
    # -- possibly during a subsequent solver_nessai.run() call in the same
    # script execution, or even a later script run in the same persistent
    # process (e.g. re-running the whole script in Spyder without restarting
    # the kernel) -- invoking a stale handler that still closes over THIS run's
    # sampler/output-path state. Cancelling any pending alarm (not just
    # resetting the handler) closes that gap.
    import signal as _signal
    try:
        _signal.alarm(0)
    except (OSError, ValueError):
        pass
    for _sig in (_signal.SIGINT, _signal.SIGTERM, _signal.SIGALRM):
        try:
            _signal.signal(_sig, _signal.SIG_DFL)
        except (OSError, ValueError):
            pass

    # Convert structured posterior array → plain (N_post, ndim) float array
    ps      = sampler.posterior_samples
    samples = np.column_stack([ps[n] for n in model.names])

    # MAP: sample with the highest log-likelihood
    if 'logL' in ps.dtype.names:
        best_params = samples[np.argmax(ps['logL'])]
    else:
        best_params = samples[np.argmax(ps['logP'])]

    logz    = float(sampler.logZ)
    logzerr = float(sampler.logZ_error)

    print(f"\nLog evidence: {logz:.2f} ± {logzerr:.2f}")
    print(f"Posterior samples: {len(samples)}")

    # nessai's equal-weight posterior_samples is capped at the effective sample
    # size (often only ~1-3k for a tight posterior -> lumpy corner contours).
    # Also expose the FULL nested samples with their normalised posterior
    # weights: a weighted corner uses every dead point (down-weighted in the
    # tail) instead of the ~ESS equal-weight subset, giving a smooth density
    # without changing the posterior it represents.
    weighted_samples, weights = None, None
    try:
        from nessai.posterior import compute_weights as _compute_weights
        _ns   = sampler.nested_samples
        _, _lw = _compute_weights(_ns['logL'], nlive)
        _w    = np.exp(_lw - _lw.max())
        _w   /= _w.sum()
        weighted_samples = np.column_stack([_ns[n] for n in model.names])
        weights          = _w
    except Exception as _e:
        print(f"  [weighted nested samples unavailable: {_e}]")

    return SimpleNamespace(
        samples             = samples,
        log_probs           = None,
        best_params         = best_params,
        chain               = None,
        chain_burn          = None,
        logprob_burn        = None,
        acceptance_fraction = None,
        tau                 = None,
        tau_err             = None,
        nsteps_prod         = None,
        logz                = logz,
        logzerr             = logzerr,
        weighted_samples    = weighted_samples,
        weights             = weights,
    )


def run_model_selection(logl_factory, pt_factory, args, ndim_per,
                        n_min=N_MIN, n_max=N_MAX, nlive=NLIVE,
                        nlive_survey=NLIVE_TD, nlive_per_dim=100,
                        output='nessai_td', log_prior_factory=None,
                        n_pool=N_POOL, stopping_criterion=1.0,
                        rerun_winner=True, stopping_criterion_winner=0.1):
    """Run Nessai for N = n_min..n_max and select N by log-evidence.

    Uses a two-pass strategy when rerun_winner=True (default):

      Pass 1 — survey: run all N with the (loose) stopping_criterion.
        Evidences gaps between models are typically >100 nats, so a loose
        threshold (1.0) is sufficient to identify the winning N reliably and
        runs ~10× faster than the tight nessai default of 0.1.

      Pass 2 — refinement: re-run the winning N only with the tight
        stopping_criterion_winner (default 0.1, the nessai default).
        This gives accurate posterior samples and a precise log-evidence
        for the most probable model without paying for tight convergence
        on every disfavoured N.

    Total cost: (n_max - n_min + 1) loose runs + 1 tight run.
    This is always cheaper than (n_max - n_min + 1) tight runs and never
    worse in model-order accuracy.

    Live-point counts are set separately for the two passes:

      Survey  — flat nlive_survey for every N (default 500).  Evidence gaps
        are typically >100 nats so a rough estimate is all that is needed to
        rank models.  No per-dimension scaling is applied; keeping this flat
        avoids the flow-training overhead growing with N.

      Winner  — nlive_w = max(nlive, nlive_per_dim * ndim_n) ensures the
        refinement run is never starved.  With nlive_per_dim=100 and
        ndim_per=7: N=1→1000, N=2→1400, N=3→2100.

    Parameters
    ----------
    logl_factory               : callable  n -> log_likelihood(params, *args)
    pt_factory                 : callable  n -> prior_transform(unit_cube) -> params
    args                       : tuple  forwarded to each log_likelihood.
    ndim_per                   : int    parameters per source.
    n_min, n_max               : int    range of N values to test.
    nlive                      : int    baseline live-point count for the
                                 winner refinement pass.  Actual count is
                                 max(nlive, nlive_per_dim * ndim_n).
    nlive_survey               : int    flat live-point count for all survey
                                 runs (default NLIVE_TD=500).  Not scaled
                                 with ndim — survey only needs to rank models.
    nlive_per_dim              : int    minimum live points per parameter for
                                 the winner pass (default 100).
    output                     : str    parent directory; N{n}/ subdirs per N,
                                 N{n}_final/ for the winner refinement run.
    log_prior_factory          : callable or None
                                 n -> log_prior(params) -> float.
    n_pool                     : int    likelihood worker processes.
    stopping_criterion         : float or callable  (survey pass)
                                 dlogZ threshold for all per-N survey runs.
                                 Float or callable n -> float.
    rerun_winner               : bool   if True (default) re-run the winning
                                 N with stopping_criterion_winner for accurate
                                 posteriors.
    stopping_criterion_winner  : float  dlogZ threshold for the winner
                                 refinement run (default 0.1, nessai default).

    Returns
    -------
    SimpleNamespace with standard ExoPlanetLab fields plus model-selection extras:
        most_probable_n, n_logz, n_logzerr, all_results, n_distribution, n_chain.
    """
    import time

    n_logz     = np.full(n_max + 1, -np.inf)
    n_logzerr  = np.zeros(n_max + 1)
    all_results = [None] * (n_max + 1)

    _sc_fn = stopping_criterion if callable(stopping_criterion) \
             else (lambda n, _sc=stopping_criterion: _sc)

    # ── Pass 1: survey all N with loose stopping criterion ────────────────────
    t_total_start = time.time()
    for n in range(n_min, n_max + 1):
        outdir_n = os.path.join(output, f'N{n}')
        ndim_n   = n * ndim_per
        logl_n   = logl_factory(n)
        pt_n     = pt_factory(n)
        sc_n     = _sc_fn(n)

        if log_prior_factory is not None:
            lp_n = log_prior_factory(n)
        else:
            eps  = 1e-9
            _lo  = pt_n(np.full(ndim_n, eps))
            _hi  = pt_n(np.full(ndim_n, 1.0 - eps))
            def lp_n(params, _lo=_lo, _hi=_hi):
                if np.any(params < _lo) or np.any(params > _hi):
                    return -np.inf
                return 0.0

        print(f"\n{'='*66}")
        print(f"  Nessai survey (pass 1): N = {n}  "
              f"({ndim_n} parameters, nlive={nlive_survey}, dlogZ<{sc_n})")
        print(f"  Start: {time.strftime('%H:%M:%S')}")
        print(f"{'='*66}", flush=True)
        t_n_start = time.time()

        res_n = run(logl_n, lp_n, pt_n, args,
                    ndim=ndim_n, nlive=nlive_survey, output=outdir_n,
                    n_pool=n_pool, stopping_criterion=sc_n)

        elapsed_n = time.time() - t_n_start
        n_logz[n]      = res_n.logz
        n_logzerr[n]   = res_n.logzerr
        all_results[n] = res_n
        print(f"  N = {n}: ln Z = {res_n.logz:.2f} ± {res_n.logzerr:.2f}"
              f"  (elapsed: {elapsed_n/60:.1f} min)", flush=True)

    valid = [n for n in range(n_min, n_max + 1) if np.isfinite(n_logz[n])]
    most_probable_n = int(max(valid, key=lambda n: n_logz[n]))
    print(f"\n  Survey complete. Winner: N = {most_probable_n} "
          f"(ln Z = {n_logz[most_probable_n]:.2f})", flush=True)

    # ── Pass 2: refine winner with tight stopping criterion ───────────────────
    if rerun_winner and stopping_criterion_winner < _sc_fn(most_probable_n):
        n       = most_probable_n
        ndim_n  = n * ndim_per
        nlive_w = max(nlive, nlive_per_dim * ndim_n)
        logl_n  = logl_factory(n)
        pt_n    = pt_factory(n)

        if log_prior_factory is not None:
            lp_n = log_prior_factory(n)
        else:
            eps = 1e-9
            _lo = pt_n(np.full(ndim_n, eps))
            _hi = pt_n(np.full(ndim_n, 1.0 - eps))
            def lp_n(params, _lo=_lo, _hi=_hi):
                if np.any(params < _lo) or np.any(params > _hi):
                    return -np.inf
                return 0.0

        outdir_winner = os.path.join(output, f'N{n}_final')
        print(f"\n{'='*66}")
        print(f"  Nessai refinement (pass 2): N = {n}  "
              f"({ndim_n} parameters, nlive={nlive_w}, "
              f"dlogZ<{stopping_criterion_winner})")
        print(f"  Start: {time.strftime('%H:%M:%S')}")
        print(f"{'='*66}", flush=True)
        t_winner_start = time.time()

        res_winner = run(logl_n, lp_n, pt_n, args,
                         ndim=ndim_n, nlive=nlive_w, output=outdir_winner,
                         n_pool=n_pool,
                         stopping_criterion=stopping_criterion_winner)

        elapsed_winner = time.time() - t_winner_start
        n_logz[n]      = res_winner.logz
        n_logzerr[n]   = res_winner.logzerr
        all_results[n] = res_winner
        print(f"  N = {n} (final): ln Z = {res_winner.logz:.2f} "
              f"± {res_winner.logzerr:.2f}  "
              f"(elapsed: {elapsed_winner/60:.1f} min)", flush=True)

    t_total = time.time() - t_total_start
    print(f"\n  Total model-selection time: {t_total/60:.1f} min", flush=True)

    # Recompute N-distribution with (possibly updated) winner logZ.
    logz_max = n_logz[most_probable_n]
    n_dist   = np.zeros(n_max + 1)
    for n in range(n_min, n_max + 1):
        if np.isfinite(n_logz[n]):
            n_dist[n] = np.exp(n_logz[n] - logz_max)
    n_dist /= n_dist.sum()

    best = all_results[most_probable_n]
    return SimpleNamespace(
        samples             = best.samples,
        log_probs           = best.log_probs,
        best_params         = best.best_params,
        chain               = None,
        chain_burn          = None,
        logprob_burn        = None,
        acceptance_fraction = None,
        tau                 = None,
        tau_err             = None,
        nsteps_prod         = None,
        logz                = best.logz,
        logzerr             = best.logzerr,
        most_probable_n     = most_probable_n,
        n_logz              = n_logz,
        n_logzerr           = n_logzerr,
        all_results         = all_results,
        n_distribution      = n_dist,
        n_chain             = None,
    )


def print_diagnostics(result, active_names):
    """Print nessai diagnostics to stdout."""
    print("\n--- Nessai (normalizing-flow nested sampling) diagnostics ---")
    if result.logz is not None:
        print(f"  Log evidence (ln Z): {result.logz:.2f} ± {result.logzerr:.2f}")
    print(f"  Posterior samples  : {len(result.samples)}")

    if hasattr(result, 'most_probable_n') and hasattr(result, 'n_logz'):
        print(f"\n  Model-selection evidence table:")
        logz_map = result.n_logz[result.most_probable_n]
        for n, (lz, lze) in enumerate(zip(result.n_logz, result.n_logzerr)):
            if np.isfinite(lz):
                marker = '  ← MAP' if n == result.most_probable_n else ''
                delta  = lz - logz_map
                print(f"    N = {n}: ln Z = {lz:10.2f} ± {lze:.2f}"
                      f"  (Δln Z = {delta:+.2f}){marker}")
        print(f"  Most probable N    : {result.most_probable_n}")
        probs = "  ".join(f"P(N={n})={p:.3f}"
                          for n, p in enumerate(result.n_distribution) if p > 0.001)
        print(f"  N-distribution     : {probs}")

    if result.logz is not None and result.logz < -1000:
        print("  WARNING: logZ is very negative — sampler may not have reached the peak.")
        print("  Try increasing nlive or tightening priors.")
