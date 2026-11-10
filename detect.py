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
Fast signal-count screening for ExoPlanetLab: a profile Keplerian
periodogram with bootstrap false-alarm-probability (FAP) significance
testing, used as a much cheaper alternative to full nested-sampling
(nessai) or trans-dimensional (rjflow) model selection when the only
question is "how many planets are in this data".

Background
----------
A naive Lomb-Scargle periodogram assumes the signal is a pure function of
time. Gaia along-scan astrometry is not: each epoch's signal is also
projected through a per-epoch scan angle (see orbit.orbital_model_AL), so
a plain time-domain periodogram aliases badly. This module instead
profiles the two genuinely nonlinear orbital parameters (eccentricity e
and periastron epoch T0) on a coarse grid at each trial period, and at
each (P, e, T0) node solves the four Thiele-Innes amplitudes (A, B, F, G)
in closed form via weighted linear least squares (they enter the model
linearly once P, e, T0 are fixed), with the basis projected by the same
single-star-fit operator (I - H) as every model compared with the data
(_PROJ; none is built for data simulated directly as orbit plus noise). The
periodogram value at each trial period is the best chi2 reduction over the
(e, T0) grid.

The significance of a peak is a permutation bootstrap: the residuals are
shuffled among the epochs, keeping the times and scan angles, which destroys
any orbital signal but keeps the noise and the sampling, and the highest peak
of each shuffled periodogram gives its distribution for noise alone. The FAP
is the fraction of shuffles whose highest peak reaches the observed one,
(n_exceed + 1) / (n_bootstrap + 1); by default 200 permutations and a 1%
threshold (detect_signals), with early stopping (_bootstrap_fap).

Three robustness fixes came out of extensive empirical validation on a
2-planet test case, and are load-bearing -- do not "simplify" them away
without re-validating:

  1. Bounded local refinement with explicit failure detection.
     An unconstrained optimizer at a period near a search boundary can
     get stuck oscillating across the (invalid) edge, silently returning
     a garbage near-zero-amplitude fit. The refinement here is bounded
     and its result is checked against FIT_FAILURE_SENTINEL before being
     trusted.

  2. Boundary-pinning rejection.
     Even with a bounded optimizer, very long trial periods combined with
     high eccentricity give the model enough freedom (a periastron
     "spike" freely positionable via T0, at the slowest-varying available
     period) to fit noise convincingly -- repeatedly, regardless of where
     the period/eccentricity search bounds were set. Capping eccentricity
     alone does NOT fix this: the artifact simply re-pins at whatever cap
     is chosen. The one reliable tell (consistent across every failure
     mode observed) is that the *period* of a spurious fit sits at or
     within a few percent of the search boundary. Eccentricity proximity
     to its own cap is NOT used as a rejection criterion: genuine signals
     with moderately high eccentricity legitimately refine close to any
     reasonable e_max, so that check is confounded and is not made.
     Exception: a fit pinned at the UPPER period bound is kept when its FAP is
     below BOUNDARY_FAP_MAX (default 1e-20), since a strong orbit longer than
     the baseline also pins there. Lower-bound pins are always rejected.

  3. Cross-refinement after sequential peeling, with amplitude-gated
     period-duplicate/harmonic rejection.
     Sequential "fit one, subtract, screen for the next" peeling fits each
     signal against a residual that still fully contains every later
     signal's unmodeled contribution. That contamination biases the
     early fits (e.g. eccentricity pulled toward the search cap) just
     enough that subtracting them leaves real leftover structure behind
     -- which a later screen can then mistake for a spurious extra copy
     of an earlier signal. Confirmed empirically on a 2-planet case: one
     round of alternating re-fits (each signal refit against the residual
     with only the *other* signals subtracted) cut the joint residual
     chi2 by ~8x and dropped the leftover periodogram power that had
     triggered a spurious third "detection" from 0.54 (comparable to a
     real signal) to 0.22 (roughly noise level).

     Cross-refinement alone is NOT sufficient, though: a duplicate/
     harmonic artifact refines to a small-but-otherwise-valid fit (it
     doesn't fail, and isn't boundary-pinned), so a period-coincidence
     check is also needed. But period coincidence alone isn't safe
     either -- a genuine second planet can legitimately sit at a
     harmonic-like period ratio to the first (confirmed on a case with
     true periods in an exact 2:1 ratio, where a naive period-only check
     wrongly dropped the real second planet). Amplitude is what actually
     distinguishes them: the real, harmonic-period second planet retained
     ~42% of the dominant signal's amplitude, while genuine artifacts
     retained only ~2-5%. See _cross_refine_all and
     _dedupe_period_harmonics.

Public API
----------
detect_signals(t_obs, scan_angles, data, sigma, p_min, p_max, **kwargs)
    Sequentially detects and subtracts significant signals; returns a
    SimpleNamespace(n_detected, params, screens).
"""

import multiprocessing as _mp
from config import limit_blas_threads
import os
import numpy as np
from scipy.optimize import minimize
from types import SimpleNamespace

import orbit

try:
    from config import N_POOL as _N_POOL_DEFAULT
except Exception:
    _N_POOL_DEFAULT = 1

FIT_FAILURE_SENTINEL = 1e11  # -logL above this -> refinement treated as failed

# Single-star-fit projector (orbit.SingleStarProjector) applied to every model
# this module compares with the data, or None. The data handed to
# detect_signals are the residuals of a single-star fit, i.e. (I - H) d, so
# the model must be (I - H) m(theta) (see orbit.SingleStarProjector). Set by
# detect_signals() before any worker pool is created ('fork' workers inherit
# it); every array here is over the same data epochs the projector was built
# for.
_PROJ = None


def _proj(v):
    return v if _PROJ is None else _PROJ.apply(v)


def _ensure_fork():
    """Use the 'fork' start method so workers inherit module state without a
    re-import.  Matches the pattern in the solver_*.py backends."""
    if _mp.get_start_method(allow_none=True) != 'fork':
        try:
            _mp.set_start_method('fork', force=True)
        except (RuntimeError, ValueError):
            pass


def _split_counts(total, n_parts):
    """Split `total` items into `n_parts` chunks as evenly as possible."""
    base, rem = divmod(total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


# Robust screen. The periodogram is weighted least squares, the
# optimal statistic for Gaussian noise. The DR4 epoch data have heavy tails
# (Student-t with nu ~ 2-9): a few outliers then dominate chi2 and the screen
# finds 0-2 of 6 signals at S = 6-10 that the Student-t joint fit detects
#. With robust=True each screen first fits t_nu(0, (f sigma)^2) to its
# residual and replaces it by the score of that distribution,
#
#     y' = f sigma phi(y / (f sigma)),   phi(r) = (nu + 1) r / (nu + r^2),
#
# with errors f sigma: least squares on y' is the locally most powerful (score)
# test for a small linear signal in t noise -- outliers get a bounded, and for
# large r decreasing, influence. Everything else (periodogram, permutation
# bootstrap, thresholds) is unchanged, and the permutation test stays valid:
# it shuffles y' as it shuffled y. The local refinement of a candidate uses the
# residual clipped at +/- ROBUST_CLIP f sigma instead (phi would rescale its
# amplitude). For Gaussian data nu is large and y' = y.
ROBUST_CLIP = 3.0


N_TOP_PEAKS = 5        # distinct peaks of the first periodogram kept as alternative starts
TOP_PEAK_SEP = 0.15    # their minimum separation in ln P


def _t_fit(y, sigma):
    """Maximum-likelihood (nu, f) of y_i ~ t_nu(0, (f sigma_i)^2); nu in
    [1, 100], f in [0.1, 1e4] (f is free to grow when the residual is still
    dominated by a signal, so that the transform stays near-linear there)."""
    from scipy.special import gammaln
    r0 = np.asarray(y, float) / np.asarray(sigma, float)
    lo, hi = np.log([1.0, 0.1]), np.log([100.0, 1.0e4])

    def nll(x):
        if np.any(x < lo) or np.any(x > hi):
            return np.inf
        nu, f = np.exp(x)
        return -(len(r0) * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
                            - np.log(f))
                 - (nu + 1) / 2 * np.sum(np.log1p((r0 / f) ** 2 / nu)))
    f0 = max(0.2, min(1.0e3, 1.4826 * float(np.median(np.abs(r0)))))
    best = min((minimize(nll, np.log([nu0, f0]), method='Nelder-Mead',
                         options=dict(xatol=1e-3, fatol=1e-4)) for nu0 in (3.0, 30.0)),
               key=lambda r: r.fun)
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def _robustify(y, sigma):
    """(score-transformed residual, clipped residual, f sigma, nu, f)."""
    nu, f = _t_fit(y, sigma)
    s = f * np.asarray(sigma, float)
    r = np.asarray(y, float) / s
    return (s * (nu + 1.0) * r / (nu + r * r), np.clip(y, -ROBUST_CLIP * s, ROBUST_CLIP * s),
            s, nu, f)


def _make_period_grid(baseline, p_min, p_max, oversample):
    f_min = 1.0 / p_max
    f_max = 1.0 / p_min
    df = 1.0 / (oversample * baseline)
    freqs = np.arange(f_min, f_max, df)
    return 1.0 / freqs


def _chi2_reduction(y, w, X, Y, cp, sp):
    """Given fixed (P, e, T0) -> X, Y arrays, solve the linear (A, B, F, G)
    amplitudes and return (chi2_of_best_linear_fit, coeffs)."""
    # Basis columns for orbit.orbital_model_AL's mapping
    #   AL = (B X + G Y) sin(phi) + (A X + F Y) cos(phi)
    # (x = north, y = east), i.e. coefficients of [X sp, X cp, Y sp, Y cp]
    # are (B, A, G, F); returned reordered to (A, B, F, G).
    F = _proj(np.column_stack([X * sp, X * cp, Y * sp, Y * cp]))
    FtWF = (F * w[:, None]).T @ F
    FtWy = F.T @ (y * w)
    try:
        coeffs = np.linalg.solve(FtWF, FtWy)
    except np.linalg.LinAlgError:
        return float(np.sum(w * y ** 2)), np.zeros(4)
    model = F @ coeffs
    return float(np.sum(w * (y - model) ** 2)), coeffs[[1, 0, 3, 2]]


# Orbital-plane coordinates X(M) = cos E - e, Y(M) = sqrt(1-e^2) sin E on a fine
# mean-anomaly grid, one table per eccentricity of the (small, fixed) screening
# grid. Linear interpolation on 2^14 intervals is accurate to ~2e-7 for e <= 0.6
# (error ~ dM^2/8 * max|X''|), negligible for a screen whose candidates are then
# refined with the exact model, and it removes the per-period Kepler solve that
# dominated the screen's cost.
_M_GRID = np.linspace(0.0, 2.0 * np.pi, 2 ** 14 + 1)
_XY_TABLES = {}


def _xy_table(e):
    key = float(e)
    if key not in _XY_TABLES:
        E = orbit.kepler_eq(_M_GRID.copy(), key, tol=1e-14)
        _XY_TABLES[key] = (np.cos(E) - key, np.sqrt(1.0 - key ** 2) * np.sin(E))
    return _XY_TABLES[key]


_M_NSEG = len(_M_GRID) - 1
_XY_TABLES_D = {}


def _xy_table_d(e):
    """(X, Y, dX, dY) tables: values and per-interval differences, for direct
    linear interpolation on the uniform _M_GRID (see _projected_basis)."""
    key = float(e)
    if key not in _XY_TABLES_D:
        Xtab, Ytab = _xy_table(key)
        _XY_TABLES_D[key] = (Xtab, Ytab, np.diff(Xtab), np.diff(Ytab))
    return _XY_TABLES_D[key]


def _projected_basis(t, P, cp, sp, e_arr, n_phase):
    """Projected linear basis [X sp, X cp, Y sp, Y cp] at every (e, T0) node of
    the screening grid for trial period P; returns (K, n_t, 4), K = n_e*n_phase,
    nodes in e-major order with T0_k = t[0] - P k / n_phase. It depends only on
    the design (times, scan angles, grid), not on the data.

    X, Y are linearly interpolated from the per-e tables (_xy_table). The
    table grid is uniform, so the interval index and fraction are computed
    directly -- once per period, shared by every e and by X and Y -- instead
    of np.interp's binary search per point (which was ~40% of the screen's
    time); the result is the same linear interpolation. The basis is built
    in (n_t, K, 4) layout, which the projection needs, so no transposed copies
    are made."""
    n_e, n_t = len(e_arr), len(t)
    # T0_k = t[0] - P k / n_phase  ->  M = 2 pi (t - t[0]) / P + 2 pi k / n_phase
    k = np.arange(n_phase)
    M = np.mod((2.0 * np.pi * (t - t[0]) / P)[:, None]
               + (2.0 * np.pi * k / n_phase)[None, :], 2.0 * np.pi)   # (n_t, n_phase)
    u = M * (_M_NSEG / (2.0 * np.pi))
    idx = np.minimum(u.astype(np.intp), _M_NSEG - 1)
    frac = u - idx
    K = n_e * n_phase
    Fb = np.empty((n_t, K, 4))
    for i, e in enumerate(e_arr):
        Xtab, Ytab, dX, dY = _xy_table_d(e)
        X = Xtab[idx] + frac * dX[idx]                                # (n_t, n_phase)
        Y = Ytab[idx] + frac * dY[idx]
        sl = slice(i * n_phase, (i + 1) * n_phase)
        Fb[:, sl, 0] = X * sp[:, None]
        Fb[:, sl, 1] = X * cp[:, None]
        Fb[:, sl, 2] = Y * sp[:, None]
        Fb[:, sl, 3] = Y * cp[:, None]
    Fb = _proj(Fb.reshape(n_t, K * 4)).reshape(n_t, K, 4)
    return Fb.transpose(1, 0, 2)                                       # (K, n_t, 4) view


def _profile_power_at_period(t, y, sigma, P, cp, sp, e_grid, n_phase):
    """Profile over (e, T0) at fixed P; return (best_power, best_e, best_T0).

    Vectorised over the whole (e, T0) grid: X, Y interpolated from per-e
    tables (_xy_table) for all n_e * n_phase combinations, the projection
    applied to all basis columns at once, and batched 4x4 normal-equation
    solves, with
    chi2_min = y'Wy - b'c. This gives the same result as looping over the grid
    one node at a time (same grid, same first-maximum tie-breaking), but
    avoids ~n_e*n_phase Python-level iterations per period, which dominated
    the screen's run time (dozens of minutes per source with the bootstrap).
    """
    w = 1.0 / sigma ** 2
    wy = w * y
    chi2_null = float(np.sum(wy * y))
    e_arr = np.asarray(e_grid, dtype=float)
    Fb = _projected_basis(t, P, cp, sp, e_arr, n_phase)             # (K, n_t, 4)
    K = Fb.shape[0]
    FtW  = Fb.transpose(0, 2, 1) * w[None, None, :]                 # (K, 4, n_t)
    FtWF = np.matmul(FtW, Fb)                                       # (K, 4, 4)
    FtWy = Fb.transpose(0, 2, 1) @ wy                               # (K, 4)
    try:
        coeffs = np.linalg.solve(FtWF, FtWy[..., None])[..., 0]
        chi2_model = chi2_null - np.einsum('ki,ki->k', FtWy, coeffs)
    except np.linalg.LinAlgError:
        chi2_model = np.empty(K)
        for j in range(K):
            try:
                c = np.linalg.solve(FtWF[j], FtWy[j])
                chi2_model[j] = chi2_null - FtWy[j] @ c
            except np.linalg.LinAlgError:
                chi2_model[j] = chi2_null
    power = (chi2_null - chi2_model) / chi2_null
    j = int(np.argmax(power))                       # first maximum, e-major order
    return float(power[j]), float(e_arr[j // n_phase]), float(t[0] - P * ((j % n_phase) / n_phase))


def _max_power_over_grid(t, y, sigma, periods, cp, sp, e_grid, n_phase):
    best = -np.inf
    for P in periods:
        pw, _, _ = _profile_power_at_period(t, y, sigma, P, cp, sp, e_grid, n_phase)
        if pw > best:
            best = pw
    return best


def _profile_grid_chunk(args):
    """Worker: (best_power, best_e, best_T0) for each period in a sub-grid.
    Independent of every other chunk -- read-only inputs, no shared state."""
    t, y, sigma, periods, cp, sp, e_grid, n_phase = args
    out = np.empty((len(periods), 3))
    for k, P in enumerate(periods):
        out[k] = _profile_power_at_period(t, y, sigma, P, cp, sp, e_grid, n_phase)
    return out


def _profile_grid(t, y, sigma, periods, cp, sp, e_grid, n_phase, n_pool=1):
    """Profile periodogram over a full period grid, split across n_pool
    worker processes.  Returns (powers, e_seeds, t0_seeds) arrays aligned
    with `periods`."""
    n_pool = max(1, int(n_pool))
    if n_pool <= 1 or len(periods) < 2 * n_pool:
        out = _profile_grid_chunk((t, y, sigma, periods, cp, sp, e_grid, n_phase))
        return out[:, 0], out[:, 1], out[:, 2]

    edges = np.cumsum([0] + _split_counts(len(periods), n_pool))
    tasks = [(t, y, sigma, periods[a:b], cp, sp, e_grid, n_phase)
             for a, b in zip(edges[:-1], edges[1:]) if b > a]
    _ensure_fork()
    with _mp.Pool(min(n_pool, len(tasks)), initializer=limit_blas_threads) as pool:
        parts = pool.map(_profile_grid_chunk, tasks)
    out = np.concatenate(parts, axis=0)
    return out[:, 0], out[:, 1], out[:, 2]


def _max_power_multi(t, Ys, sigma, periods, cp, sp, e_grid, n_phase):
    """_max_power_over_grid for many data vectors at once: Ys is (n_y, n_t),
    returns (n_y,) maximum powers over the whole (P, e, T0) grid.

    The permutation bootstrap only shuffles the data; the design -- times, scan
    angles, grid, projected basis and its 4x4 normal matrices -- is the same
    for every permutation. So the basis and the inverse normal matrices are
    built once per trial period and every data vector is handled by one matrix
    product, instead of one full grid pass per permutation (which made the
    bootstrap ~85% of the screen's cost). Same statistic, same grid, same
    permutations: equal to the per-permutation loop up to rounding. A node
    whose normal matrix is singular gets power 0, as there."""
    Ys = np.atleast_2d(np.asarray(Ys, dtype=float))
    w = 1.0 / sigma ** 2
    chi2_null = np.sum(Ys * Ys * w[None, :], axis=1)                  # (n_y,)
    e_arr = np.asarray(e_grid, dtype=float)
    best = np.full(Ys.shape[0], -np.inf)
    WYt = (Ys * w[None, :]).T                                         # (n_t, n_y)
    for P in periods:
        Fb = _projected_basis(t, P, cp, sp, e_arr, n_phase)           # (K, n_t, 4)
        K = Fb.shape[0]
        FtWF = np.matmul(Fb.transpose(0, 2, 1) * w[None, None, :], Fb)   # (K, 4, 4)
        try:
            Finv = np.linalg.inv(FtWF)
            good = np.ones(K, dtype=bool)
        except np.linalg.LinAlgError:
            Finv = np.zeros_like(FtWF)
            good = np.zeros(K, dtype=bool)
            for j in range(K):
                try:
                    Finv[j] = np.linalg.inv(FtWF[j])
                    good[j] = True
                except np.linalg.LinAlgError:
                    pass
        B = (Fb.transpose(0, 2, 1).reshape(K * 4, -1) @ WYt).reshape(K, 4, -1)  # F'W y
        expl = np.einsum('kin,kin->kn', B, np.matmul(Finv, B))       # (K, n_y)
        expl[~good] = 0.0
        best = np.maximum(best, expl.max(axis=0) / chi2_null)
    return best


def _max_power_multi_chunk(args):
    return _max_power_multi(*args)


def _max_power_multi_pool(t, Ys, sigma, periods, cp, sp, e_grid, n_phase, pool=None):
    """_max_power_multi with the period grid split over a worker pool."""
    if pool is None:
        return _max_power_multi(t, Ys, sigma, periods, cp, sp, e_grid, n_phase)
    n = pool._processes
    edges = np.cumsum([0] + _split_counts(len(periods), n))
    tasks = [(t, Ys, sigma, periods[a:b], cp, sp, e_grid, n_phase)
             for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return np.max(np.vstack(pool.map(_max_power_multi_chunk, tasks)), axis=0)


def _boot_chunk(args):
    """Worker: run `n_iter` bootstrap permutations and return their null
    max-powers (array).  Each chunk gets its own independent RNG stream
    (SeedSequence.spawn), so results are order-independent."""
    (t, y, sigma, periods, cp, sp, e_grid, n_phase, n_iter, seedseq) = args
    rng = np.random.default_rng(seedseq)
    return np.array([_max_power_over_grid(t, rng.permutation(y), sigma, periods,
                                          cp, sp, e_grid, n_phase)
                     for _ in range(n_iter)])


# Sequential bootstrap (see _bootstrap_fap). FAP_GEV=False forces the full
# n_bootstrap permutations for every significant peak.
FAP_GEV          = os.environ.get('EXOPLANETLAB_FAP_GEV', '1') == '1'
FAP_GEV_MIN_BOOT = 45       # permutations before a Gumbel extrapolation is tried
FAP_GEV_MARGIN   = 100.0    # accept only if the extrapolated FAP < threshold / margin
# Evaluate the bootstrap permutations in batches that share one grid pass
# (_max_power_multi). EXOPLANETLAB_SCREEN_BATCH_BOOT=0 restores one grid pass
# per permutation.
SCREEN_BATCH_BOOT = os.environ.get('EXOPLANETLAB_SCREEN_BATCH_BOOT', '1') == '1'
# A refined fit pinned at the UPPER period bound is accepted as a detection
# when its FAP is below this (module docstring, fix 2): a strong orbit longer
# than the baseline. Such FAPs come from the Gumbel extrapolation, so with
# FAP_GEV=0 (FAP floor 1/(n_bootstrap+1)) upper-bound pins are never accepted.
# EXOPLANETLAB_SCREEN_BOUNDARY_FAP=0 restores the old rule (always reject).
BOUNDARY_FAP_MAX = float(os.environ.get('EXOPLANETLAB_SCREEN_BOUNDARY_FAP', '1e-20'))
BOOT_BATCH        = 50       # permutations per shared pass after the first batch


def _gumbel_fap(null_max, observed_power):
    """Tail probability of observed_power under a Gumbel distribution fitted by
    moments to the null max-powers (extreme-value model of periodogram peaks,
    e.g. Suveges 2014)."""
    s = float(np.std(null_max, ddof=1))
    if not np.isfinite(s) or s <= 0.0:
        return np.nan
    beta = s * np.sqrt(6.0) / np.pi
    mu = float(np.mean(null_max)) - 0.5772156649 * beta
    return float(-np.expm1(-np.exp(-(observed_power - mu) / beta)))


def _bootstrap_fap(t, y, sigma, periods, cp, sp, e_grid, n_phase,
                    observed_power, n_bootstrap, seed, n_pool=1,
                    fap_threshold=None):
    """Shuffle the residual across epochs (keeping t, scan angles fixed) to
    build the null max-power distribution. FAP floor is ~1/(n_bootstrap+1),
    so fap_threshold must stay comfortably above that or even the
    strongest real signal could never register as significant.

    Permutations run in batches of n_pool (one per worker) and stop early
    when the answer is settled (fap_threshold given):
      * NOT significant: once n_exceed reaches the count that makes
        (n_exceed+1)/(n_bootstrap+1) >= fap_threshold, no further permutation
        can change the decision -- stop (exact).
      * Clearly significant: after FAP_GEV_MIN_BOOT permutations with no
        exceedance, fit a Gumbel distribution to the null max-powers; if its
        tail probability at observed_power is below fap_threshold /
        FAP_GEV_MARGIN, stop and return that value (an approximation, used
        only far from the threshold; EXOPLANETLAB_FAP_GEV=0 disables it).
      * Otherwise all n_bootstrap permutations are run.
    Returns (fap, n_used, method) with method 'full', 'early-stop' or 'gumbel'.
    The FAP is a Monte-Carlo estimate; batching changes the random streams, so
    it is statistically equivalent to, not bit-identical with, earlier runs."""
    common = (t, y, sigma, periods, cp, sp, e_grid, n_phase)
    n_pool = max(1, int(n_pool))
    if SCREEN_BATCH_BOOT:
        return _bootstrap_fap_batched(t, y, sigma, periods, cp, sp, e_grid, n_phase,
                                      observed_power, n_bootstrap, seed, n_pool,
                                      fap_threshold)
    batch = n_pool
    n_batches = int(np.ceil(n_bootstrap / batch))
    seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)
    k_stop = (int(np.ceil(fap_threshold * (n_bootstrap + 1))) - 1
              if fap_threshold is not None else None)

    pool = None
    if n_pool > 1:
        _ensure_fork()
        pool = _mp.Pool(n_pool, initializer=limit_blas_threads)
    try:
        null_max = []
        for b in range(n_batches):
            idx = range(b * batch, min((b + 1) * batch, n_bootstrap))
            tasks = [(*common, 1, seeds[i]) for i in idx]
            parts = (pool.map(_boot_chunk, tasks) if pool is not None
                     else [_boot_chunk(tk) for tk in tasks])
            null_max.extend(float(x) for p in parts for x in p)
            n_done = len(null_max)
            n_exceed = int(np.sum(np.asarray(null_max) >= observed_power))
            if k_stop is not None and n_exceed >= k_stop and n_done < n_bootstrap:
                return (n_exceed + 1) / (n_done + 1), n_done, 'early-stop'
            if (k_stop is not None and FAP_GEV and n_exceed == 0
                    and n_done >= FAP_GEV_MIN_BOOT and n_done < n_bootstrap):
                g = _gumbel_fap(np.asarray(null_max), observed_power)
                if np.isfinite(g) and g < fap_threshold / FAP_GEV_MARGIN:
                    return g, n_done, 'gumbel'
        n_exceed = int(np.sum(np.asarray(null_max) >= observed_power))
        return (n_exceed + 1) / (n_bootstrap + 1), n_bootstrap, 'full'
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _bootstrap_fap_batched(t, y, sigma, periods, cp, sp, e_grid, n_phase,
                           observed_power, n_bootstrap, seed, n_pool, fap_threshold):
    """_bootstrap_fap with the permutations evaluated in shared grid passes.

    Permutation i uses the same random stream as in _bootstrap_fap (SeedSequence(seed)
    spawned per permutation), so the null max-powers are the same numbers up to
    rounding. The stopping rules are unchanged but checked after each batch:
    the first batch has FAP_GEV_MIN_BOOT permutations (the point at which the
    Gumbel shortcut may first apply), later ones BOOT_BATCH. A batch costs about
    one grid pass whatever its size, so a clear detection costs ~1 pass instead
    of 45, and an obvious non-detection ~1 instead of a few."""
    seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)
    k_stop = (int(np.ceil(fap_threshold * (n_bootstrap + 1))) - 1
              if fap_threshold is not None else None)
    pool = None
    if n_pool > 1:
        _ensure_fork()
        pool = _mp.Pool(n_pool, initializer=limit_blas_threads)
    try:
        null_max = np.empty(0)
        n_done = 0
        while n_done < n_bootstrap:
            size = (FAP_GEV_MIN_BOOT if n_done == 0 else BOOT_BATCH)
            idx = range(n_done, min(n_done + size, n_bootstrap))
            Ys = np.vstack([np.random.default_rng(seeds[i]).permutation(y) for i in idx])
            null_max = np.concatenate([null_max, _max_power_multi_pool(
                t, Ys, sigma, periods, cp, sp, e_grid, n_phase, pool=pool)])
            n_done = len(null_max)
            n_exceed = int(np.sum(null_max >= observed_power))
            if k_stop is not None and n_exceed >= k_stop and n_done < n_bootstrap:
                return (n_exceed + 1) / (n_done + 1), n_done, 'early-stop'
            if (k_stop is not None and FAP_GEV and n_exceed == 0
                    and n_done >= FAP_GEV_MIN_BOOT and n_done < n_bootstrap):
                g = _gumbel_fap(null_max, observed_power)
                if np.isfinite(g) and g < fap_threshold / FAP_GEV_MARGIN:
                    return g, n_done, 'gumbel'
        n_exceed = int(np.sum(null_max >= observed_power))
        return (n_exceed + 1) / (n_bootstrap + 1), n_bootstrap, 'full'
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _refine_local_fit(t, scan_angles, data, sigma, p_candidate, e_seed, T0_seed,
                       p_min, p_search_max, e_max_search, ti_bound, n_restarts, seed):
    """Multi-start LOCAL refinement (this is a refinement of an
    already-identified candidate, not a fresh global search), bounded so
    the optimizer can't wander into / oscillate across the period edges.
    Returns (params, logL, ok); ok=False means every restart failed and
    the caller must not treat the result as a real detection."""
    rng = np.random.default_rng(seed)
    p_lo = max(p_min, p_candidate * 0.8)
    p_hi = min(p_search_max, p_candidate * 1.2)
    bounds = [(p_lo, p_hi), (0.0, e_max_search), (None, None),
              (-ti_bound, ti_bound), (-ti_bound, ti_bound),
              (-ti_bound, ti_bound), (-ti_bound, ti_bound)]
    cp = np.cos(scan_angles)
    sp = np.sin(scan_angles)

    best = None
    for i in range(n_restarts):
        e0 = np.clip(e_seed + rng.normal(0, 0.1), 0.01, e_max_search) if i > 0 else e_seed
        T0_0 = T0_seed + rng.normal(0, 0.05 * p_candidate) if i > 0 else T0_seed
        w = 1.0 / sigma ** 2
        M = 2.0 * np.pi * (t - T0_0) / p_candidate
        E = orbit.kepler_eq(M, e0)
        X = np.cos(E) - e0
        Y = np.sqrt(1.0 - e0 ** 2) * np.sin(E)
        _, coeffs0 = _chi2_reduction(data, w, X, Y, cp, sp)
        x0 = np.array([p_candidate, e0, T0_0, *coeffs0])

        def negloglike(x):
            model = orbit.orbital_model_AL(t, x, scan_angles, projector=_PROJ)
            resid = (data - model) / sigma
            return 0.5 * np.sum(resid ** 2)

        res = minimize(negloglike, x0, method='Powell', bounds=bounds,
                        options=dict(maxiter=4000, xtol=1e-6, ftol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    ok = np.isfinite(best.fun) and best.fun < FIT_FAILURE_SENTINEL
    return best.x, -best.fun, ok


def _signal_amplitude(params):
    """Total Thiele-Innes amplitude a0 = sqrt(A^2+B^2+F^2+G^2)."""
    return float(np.hypot(np.hypot(params[3], params[4]),
                           np.hypot(params[5], params[6])))


def _dedupe_period_harmonics(params, period_tol_frac=0.05, amp_ratio_max=0.2,
                              harmonics=(1.0, 2.0, 3.0, 0.5, 1.0 / 3.0)):
    """Drop the WEAKER of two signals whose periods coincide (or are a
    low-order harmonic/subharmonic of each other) -- but ONLY if it is
    also much weaker in amplitude. Period coincidence alone is NOT
    sufficient: a genuine second planet can legitimately sit at a
    harmonic-like period ratio to the first (e.g. an exact 2:1 case), and
    period ratio alone can't distinguish that from a duplicate/harmonic
    artifact of imperfect subtraction. Amplitude is what distinguishes
    them -- confirmed empirically on a case with a real 2:1-period second
    planet: it retained ~42% of the dominant signal's amplitude, while
    genuine artifacts retained only ~2-5%. `amp_ratio_max` is the cutoff
    below which a period-coincident weaker signal is treated as spurious.
    Returns a keep-mask (list of bool)."""
    amps = [_signal_amplitude(p) for p in params]
    order = sorted(range(len(params)), key=lambda i: -amps[i])  # strongest first
    keep = [True] * len(params)
    for pos, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[pos + 1:]:
            if not keep[j]:
                continue
            ratio = params[j][0] / params[i][0]
            period_matches = any(abs(ratio - h) <= period_tol_frac * h for h in harmonics)
            weak_enough = (amps[i] > 0) and (amps[j] / amps[i] < amp_ratio_max)
            if period_matches and weak_enough:
                keep[j] = False
    return keep


def _cross_refine_all(t, scan_angles, data, sigma, detected, p_min, p_max,
                       e_max_search, ti_bound, is_boundary_pinned_fn,
                       n_rounds=5, n_restarts=4, period_tol_frac=0.05,
                       amp_ratio_max=0.2, seed=0, verbose=True,
                       boundary_ok=None):
    """Alternate re-fitting each detected signal against the residual with
    only the OTHER currently-kept signals subtracted (a simple Gauss-Seidel
    -style joint deconvolution), instead of trusting each one's single-pass
    sequential fit. See module docstring (fix 3) for why this matters and
    the empirical evidence. Signals that fail to refine, or are still
    boundary-pinned once every other signal has been properly accounted
    for, are dropped during the rounds; afterwards, any signal whose period
    coincides with (or is a low-order harmonic of) a stronger signal's is
    also dropped -- confirmed empirically to be necessary: a duplicate/
    harmonic artifact of an imperfectly-subtracted signal can refine to a
    small-amplitude but otherwise "valid" (non-failing, non-boundary) fit,
    so period coincidence is the only reliable tell for that case.

    boundary_ok : list of bool or None
        Per signal: True if the screen accepted it although pinned at the
        upper period bound (a strong orbit longer than the baseline, see
        BOUNDARY_FAP_MAX); such a signal is exempt from the pinning check.

    Returns (kept_params, dropped) where dropped is a list of
    {index, params} for anything removed.
    """
    params = [p.copy() for p in detected]
    if len(params) < 2:
        return params, []

    keep = [True] * len(params)
    for r in range(n_rounds):
        is_last_round = (r == n_rounds - 1)
        for i in range(len(params)):
            if not keep[i]:
                continue
            others_model = np.zeros(len(t))
            for j, pj in enumerate(params):
                if j != i and keep[j]:
                    others_model += orbit.orbital_model_AL(t, pj, scan_angles, projector=_PROJ)
            resid_i = data - others_model

            refined, logl, ok = _refine_local_fit(
                t, scan_angles, resid_i, sigma,
                params[i][0], params[i][1], params[i][2],
                p_min, p_max, e_max_search, ti_bound, n_restarts,
                seed + r * 1000 + i)

            pinned = (is_last_round and is_boundary_pinned_fn(refined)
                      and not (boundary_ok is not None and boundary_ok[i]))
            if not ok or pinned:
                keep[i] = False
                if verbose:
                    reason = 'refinement failed' if not ok else 'boundary-pinned'
                    print(f"    Cross-refinement: signal {i+1} no longer "
                          f"holds up ({reason}) once the others are "
                          f"properly subtracted -- dropping (likely a "
                          f"sequential-peeling artifact, not a real signal).")
                continue
            params[i] = refined

    active_idx = [i for i, k in enumerate(keep) if k]
    active_params = [params[i] for i in active_idx]
    dedupe_keep = _dedupe_period_harmonics(active_params, period_tol_frac, amp_ratio_max)
    for local_i, keep_it in zip(active_idx, dedupe_keep):
        if not keep_it:
            keep[local_i] = False
            if verbose:
                print(f"    Cross-refinement: signal {local_i+1} "
                      f"(P={params[local_i][0]:.1f}d) coincides with a "
                      f"stronger signal's period (or its harmonic) -- "
                      f"dropping as a duplicate artifact.")

    kept_params = [p for p, k in zip(params, keep) if k]
    dropped = [dict(index=i, params=p) for i, (p, k)
               in enumerate(zip(detected, keep)) if not k]
    return kept_params, dropped


def detect_signals(t_obs, scan_angles, data, sigma, p_min, p_max, *,
                    e_max_search=0.6, n_phase=16, n_e_fine=6,
                    n_phase_coarse=8, n_e_coarse=3,
                    oversample_fine=4, oversample_coarse=2,
                    n_bootstrap=200, fap_threshold=1e-2, n_max_screen=4,
                    period_boundary_frac=0.05, ti_bound=500.0,
                    n_restarts=8, cross_refine=True, cross_refine_rounds=5,
                    cross_refine_restarts=4, period_tol_frac=0.05,
                    amp_ratio_max=0.2, seed=0, verbose=True,
                    n_pool=_N_POOL_DEFAULT, projector=None, robust=False):
    """Sequentially detect Keplerian signals via profile periodogram +
    bootstrap FAP, refining and subtracting each one before screening for
    the next, then (if more than one signal was found) cross-refine them
    all jointly to undo the contamination sequential peeling leaves
    behind (see module docstring, fix 3).

    Parameters
    ----------
    t_obs, scan_angles, data, sigma : (N,) arrays
        Observation times (days), Gaia scan angles (radians), along-scan
        residuals (mas), and per-epoch sigma_total (mas).
    p_min, p_max : float
        Period search bounds (days). p_max should match the same physical
        bound used elsewhere (e.g. 2 * MISSION_YR * 365.25 in
        ExoPlanetLab.py) so "near the boundary" means the same thing.
    e_max_search : float
        Eccentricity search cap (see module docstring for why ~0.6).
    fap_threshold : float
        Stop screening once a candidate's FAP is >= this. Must be well
        above 1/(n_bootstrap+1) (the floor) or nothing can register as
        significant.
    n_max_screen : int
        Safety cap on the number of screening iterations. Tie this to a
        single project-wide N_MAX (e.g. from config.py) rather than
        leaving it as an independent knob -- ExoPlanetLab.py does this.
    cross_refine : bool
        Whether to run the post-hoc joint cross-refinement pass when 2+
        signals are found (see module docstring, fix 3). Only disable
        this for debugging/comparison; it materially improves fit
        quality and rejects sequential-peeling duplicate artifacts.
    cross_refine_rounds, cross_refine_restarts : int
        Alternating rounds and per-signal restarts for cross-refinement.
    period_tol_frac : float
        Relative tolerance for treating two signals' periods as
        coincident/harmonically related during cross-refinement's
        de-duplication step (see _dedupe_period_harmonics).
    amp_ratio_max : float
        A period-coincident weaker signal is only treated as a spurious
        duplicate/harmonic if its amplitude is below this fraction of the
        stronger signal's -- period coincidence alone is not enough (a
        genuine second planet can legitimately sit at a harmonic-like
        period ratio to the first, e.g. an exact 2:1 case).
    n_pool : int
        Worker processes for the two hot loops -- the bootstrap-FAP null
        distribution and the fine-grid profile periodogram (both
        embarrassingly parallel).  Defaults to config.N_POOL; set to 1 for
        serial execution.  The FAP is a Monte-Carlo estimate, so parallel
        results are statistically equivalent but not bit-identical to serial
        at a given seed.

    robust : bool
        Screen the Student-t score of the residual instead of the residual
        (see ROBUST_CLIP above): for heavy-tailed noise. The returned orbit
        parameters, the peeling and the cross-refinement are unchanged.

    projector : orbit.SingleStarProjector or None
        If ``data`` are the residuals of a single-star fit (Gaia epoch
        astrometry), pass that fit's projector so every model -- the profile
        periodogram basis, the local refinements, the peeling subtraction --
        is projected the same way ((I - H) m), not compared bare.

    Returns
    -------
    SimpleNamespace(n_detected, params, screens, dropped)
        params       : list of [P, e, T0, A, B, F, G] arrays, one per
                       detected signal (ascending screening order).
        screens      : list of per-iteration diagnostic dicts (candidate
                       period/power/FAP and outcome) for logging/debugging.
        dropped      : signals removed during cross-refinement because
                       they didn't hold up once the others were properly
                       subtracted (see module docstring, fix 3).
        long_period  : list of bool, one per entry of params: True for a
                       signal accepted although pinned at the upper period
                       bound (an orbit longer than the baseline whose period
                       the screen cannot pin down; see BOUNDARY_FAP_MAX).
    """
    global _PROJ
    _PROJ = projector
    baseline = t_obs.max() - t_obs.min()
    cp = np.cos(scan_angles)
    sp = np.sin(scan_angles)

    e_grid_fine   = np.linspace(0.0, e_max_search, n_e_fine)
    e_grid_coarse = np.linspace(0.0, e_max_search, n_e_coarse)
    period_grid_fine   = _make_period_grid(baseline, p_min, p_max, oversample_fine)
    period_grid_coarse = _make_period_grid(baseline, p_min, p_max, oversample_coarse)

    def near_upper(params):
        return bool(params[0] >= p_max * (1 - period_boundary_frac))

    def near_lower(params):
        return bool(params[0] <= p_min * (1 + period_boundary_frac) + 1.0)

    def is_boundary_pinned(params):
        return near_upper(params) or near_lower(params)

    residual = data.copy()
    detected, screens, boundary_ok = [], [], []

    for i in range(n_max_screen):
        y_scr, y_fit, s_scr = residual, residual, sigma
        if robust:
            y_scr, y_fit, s_scr, _nu, _f = _robustify(residual, sigma)
            if verbose:
                print(f"    (robust screen: residual ~ t with nu = {_nu:.2f}, f = {_f:.2f})")
        powers, e_seeds, t0_seeds = _profile_grid(
            t_obs, y_scr, s_scr, period_grid_fine, cp, sp, e_grid_fine,
            n_phase, n_pool=n_pool)
        best_idx = np.argmax(powers)
        p_cand, peak_power = period_grid_fine[best_idx], powers[best_idx]
        e_seed, T0_seed = e_seeds[best_idx], t0_seeds[best_idx]

        obs_power_coarse = _max_power_over_grid(
            t_obs, y_scr, s_scr, period_grid_coarse, cp, sp, e_grid_coarse, n_phase_coarse)
        fap, n_boot_used, fap_method = _bootstrap_fap(
            t_obs, y_scr, s_scr, period_grid_coarse, cp, sp, e_grid_coarse,
            n_phase_coarse, obs_power_coarse, n_bootstrap, seed + i, n_pool=n_pool,
            fap_threshold=fap_threshold)

        info = dict(screen=i + 1, candidate_period=p_cand, power=peak_power, fap=fap,
                    n_bootstrap=n_boot_used, fap_method=fap_method)
        if i == 0:
            # the highest distinct peaks (more than TOP_PEAK_SEP apart in ln P) of
            # the first periodogram: alternative starting periods for the fit,
            # since an alias can be the highest peak
            top = []
            for j in np.argsort(-powers):
                if all(abs(np.log(period_grid_fine[j] / period_grid_fine[k])) > TOP_PEAK_SEP
                       for k in top):
                    top.append(j)
                if len(top) == N_TOP_PEAKS:
                    break
            info['top_peaks'] = [(float(period_grid_fine[j]), float(powers[j]),
                                  float(e_seeds[j]), float(t0_seeds[j])) for j in top]
        if verbose:
            print(f"--- Screen {i+1}: candidate P={p_cand:.2f}d, "
                  f"peak power={peak_power:.4f}, FAP={fap:.3e} "
                  f"({n_boot_used} permutations, {fap_method})")

        if fap >= fap_threshold:
            info['outcome'] = 'not_significant'
            screens.append(info)
            if verbose:
                print(f"    Not significant (FAP >= {fap_threshold:.0e}) -- stopping.")
            break

        best_params, logl, ok = _refine_local_fit(
            t_obs, scan_angles, y_fit, s_scr, p_cand, e_seed, T0_seed,
            p_min, p_max, e_max_search, ti_bound, n_restarts, seed + i)

        if not ok:
            info['outcome'] = 'refinement_failed'
            screens.append(info)
            if verbose:
                print(f"    Periodogram flagged this as significant, but local "
                      f"refinement FAILED (logL={logl:.2e}) -- treating as "
                      f"spurious, stopping rather than accepting a bogus fit.")
            break

        long_period = (near_upper(best_params) and not near_lower(best_params)
                       and fap < BOUNDARY_FAP_MAX)
        if is_boundary_pinned(best_params) and not long_period:
            info['outcome'] = 'boundary_pinned'
            screens.append(info)
            if verbose:
                print(f"    Periodogram flagged this as significant, but the "
                      f"refined fit is pinned at a search boundary "
                      f"(P={best_params[0]:.1f}d) -- the known spurious "
                      f"boundary-seeking pattern, not a real signal. Stopping.")
            break

        info['outcome'] = 'detected_long_period' if long_period else 'detected'
        info['params'] = best_params
        info['logl'] = logl
        screens.append(info)
        if verbose and long_period:
            print(f"    Refined fit pinned at the upper period bound "
                  f"(P={best_params[0]:.1f}d), but FAP={fap:.1e} < "
                  f"{BOUNDARY_FAP_MAX:.0e}: accepted as an orbit longer than "
                  f"the baseline (its period is set by the joint fit).")
        if verbose:
            print(f"    SIGNIFICANT. Refined fit: P={best_params[0]:.2f}d "
                  f"e={best_params[1]:.3f} T0={best_params[2]:.1f}d "
                  f"A={best_params[3]:.2f} B={best_params[4]:.2f} "
                  f"F={best_params[5]:.2f} G={best_params[6]:.2f} mas  logL={logl:.2f}")

        detected.append(best_params)
        boundary_ok.append(long_period)
        residual = residual - orbit.orbital_model_AL(t_obs, best_params, scan_angles, projector=_PROJ)

    dropped = []
    if cross_refine and len(detected) >= 2:
        if verbose:
            print(f"\n--- Cross-refining {len(detected)} candidates jointly "
                  f"(sequential peeling leaves contamination between "
                  f"signals -- see detect.py fix 3) ---")
        detected, dropped = _cross_refine_all(
            t_obs, scan_angles, data, sigma, detected, p_min, p_max,
            e_max_search, ti_bound, is_boundary_pinned,
            n_rounds=cross_refine_rounds, n_restarts=cross_refine_restarts,
            period_tol_frac=period_tol_frac, amp_ratio_max=amp_ratio_max,
            seed=seed, verbose=verbose, boundary_ok=boundary_ok)
        if verbose:
            print(f"    After cross-refinement: {len(detected)} signal(s) "
                  f"remain ({len(dropped)} dropped).")
            for i, p in enumerate(detected):
                print(f"    Signal {i+1}: P={p[0]:.2f}d e={p[1]:.3f} "
                      f"T0={p[2]:.1f}d A={p[3]:.2f} B={p[4]:.2f} "
                      f"F={p[5]:.2f} G={p[6]:.2f} mas")

    dropped_idx = {d['index'] for d in dropped}
    long_period = [f for i, f in enumerate(boundary_ok) if i not in dropped_idx]
    return SimpleNamespace(n_detected=len(detected), params=detected,
                            screens=screens, dropped=dropped,
                            long_period=long_period)
