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
Tight truncated-Gaussian prior utilities for the nessai warm start.

After the peel phase has located each signal, these helpers build narrow
priors centred on the peel posteriors.  They are used by both the
dynesty solver (between stages) and the nessai solver (for the
joint run after peeling).

Circular parameters
--------------------
The seven fit parameters here (P, e, T0, A, B, F, G — a Thiele-Innes
parameterization) are all linear/bounded quantities; the circular orbital
angles (argument of periastron ω, longitude of ascending node Ω) are only
ever derived from A/B/F/G for the forward model and are never sampled or
passed through these helpers, so `period_for` currently returns None for
every index here.  This module still supports `period=2*np.pi` (see
`_tight_stats`/`_tight_transform`/`_tight_log_prior_factory`) for a
periodic parameter, whose samples straddling the 0/2π wrap point would
otherwise look bimodal to a naive median/IQR and silently produce a tight
prior centred on the wrong value — if a direct ω/Ω parameterization is
ever added, register the relevant global indices in `CIRCULAR_PARAM_IDX`
below.

Public API
----------
period_for(global_idx)  -> float | None
_tight_stats(samples, lo_phys, hi_phys, period=None)  -> (mu, sig, lo, hi, a, b)
_tight_transform(samples, lo_phys, hi_phys, period=None)  -> callable u -> param
_tight_log_prior_factory(samples_list, lo_list, hi_list, period_list=None)  -> callable params -> float
"""

import numpy as np
from scipy.stats import truncnorm

N_SIGMA = 4   # half-width of tight prior in posterior-sigma units

# Global parameter indices of the fixed 7-parameter model (P, e, T0, A, B, F, G)
# that are 2π-periodic azimuthal angles. None of them are — see module docstring.
CIRCULAR_PARAM_IDX = frozenset()


def period_for(global_idx):
    """Return 2π for a circular (azimuthal-angle) parameter index, else None."""
    return 2.0 * np.pi if global_idx in CIRCULAR_PARAM_IDX else None


def _circular_center(samples, period):
    """Circular mean of samples on a period-length circle, plus samples
    re-wrapped into (mu - period/2, mu + period/2] so they are contiguous
    around the mean instead of straddling the 0/period cut."""
    ang = samples * (2.0 * np.pi / period)
    mu = float(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) * (period / (2.0 * np.pi))
    mu %= period
    d = np.mod(samples - mu + period / 2.0, period) - period / 2.0
    return mu, d


def _tight_stats(samples, lo_phys, hi_phys, period=None):
    """Compute (mu, sig, lo, hi, a, b) for a tight truncated Gaussian.

    Enforces a minimum sigma of 1e-4 × (hi_phys - lo_phys) so that
    even a perfectly converged peel parameter (near-zero IQR) still
    produces a non-degenerate distribution that nessai can sample from.

    If `period` is given, mu/sigma are computed with circular statistics
    (see module docstring) and the truncation bounds are centred on mu
    rather than clipped to [lo_phys, hi_phys], since the physical domain
    wraps around rather than having a hard edge there.
    """
    samples = np.asarray(samples, dtype=float)

    if period is not None:
        mu, d = _circular_center(samples, period)
        sig_iqr = float(0.5 * (np.percentile(d, 84) - np.percentile(d, 16)))
        sig_min = float(1e-4 * period)
        sig     = max(sig_iqr, sig_min, 1e-12)
        half_w  = min(N_SIGMA * sig, period / 2.0)
        lo, hi  = mu - half_w, mu + half_w
        a, b    = -half_w / sig, half_w / sig
        return mu, sig, lo, hi, a, b

    mu      = float(np.median(samples))
    sig_iqr = float(0.5 * (np.percentile(samples, 84) - np.percentile(samples, 16)))
    sig_min = float(1e-4 * (hi_phys - lo_phys))
    sig     = max(sig_iqr, sig_min, 1e-12)
    lo      = max(mu - N_SIGMA * sig, lo_phys)
    hi      = min(mu + N_SIGMA * sig, hi_phys)
    a, b    = (lo - mu) / sig, (hi - mu) / sig
    return mu, sig, lo, hi, a, b


def _tight_transform(samples, lo_phys, hi_phys, period=None):
    """Return a u→param transform using a truncated Gaussian fitted to samples."""
    mu, sig, lo, hi, a, b = _tight_stats(samples, lo_phys, hi_phys, period=period)

    if period is not None:
        def t(u):
            v = truncnorm.ppf(float(np.clip(u, 1e-10, 1 - 1e-10)), a, b, loc=mu, scale=sig)
            return float(np.mod(v - lo_phys, period) + lo_phys)
        return t

    def t(u):
        return float(truncnorm.ppf(float(np.clip(u, 1e-10, 1 - 1e-10)), a, b,
                                   loc=mu, scale=sig))
    return t


def _tight_log_prior_factory(samples_list, lo_phys_list, hi_phys_list, period_list=None):
    """Return a log_prior(params) callable using tight truncated Gaussians.

    Replaces the flat/Jeffreys log_prior for the nessai joint run so that
    nessai's initial live-point population and proposal generation are
    concentrated near the peel-phase posterior rather than spread uniformly
    across the full prior volume.

    Uses the same minimum-sigma floor as _tight_transform so that the
    log_prior support exactly matches the bounds nessai derives from the
    prior_transform, preventing the verify_model failure that arises when
    a perfectly-constrained peel parameter collapses to a near-delta function.

    For circular parameters (see `period_list`), the density is evaluated
    at the wrapped distance from the fitted circular mean so that a sample
    on the opposite side of the 0/period cut from mu isn't spuriously
    penalised.
    """
    if period_list is None:
        period_list = [None] * len(samples_list)

    records = []
    for samples, lo_phys, hi_phys, period in zip(samples_list, lo_phys_list, hi_phys_list, period_list):
        mu, sig, lo, hi, a, b = _tight_stats(samples, lo_phys, hi_phys, period=period)
        records.append((mu, sig, a, b, period))

    def log_prior(params):
        lp = 0.0
        for i, (mu, sig, a, b, period) in enumerate(records):
            val = float(params[i])
            if period is not None:
                val = np.mod(val - mu + period / 2.0, period) - period / 2.0
                v = float(truncnorm.logpdf(val, a, b, loc=0.0, scale=sig))
            else:
                v = float(truncnorm.logpdf(val, a, b, loc=mu, scale=sig))
            if not np.isfinite(v):
                return -np.inf
            lp += v
        return lp
    return log_prior
