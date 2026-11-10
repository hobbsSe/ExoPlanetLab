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
Priors of the orbital elements, as functions of their arguments only (no
run-time state), so that they can be tested on their own.
ExoPlanetLab.py calls them with its settings.

One orbit is (P, e, T0, A, B, F, G). Its prior density is

    Jeffreys in P on (p_min, p_max):          1 / P
    uniform in e on [e_min, 1) and in T0 on (t0_lo, t0_hi)
    orientation and size, in (A, B, F, G) with |A..G| < ti_max:
      'ti'         uniform -- implies p(a0, i) ~ a0^3 sin^2 i in Campbell
                   elements (dA dB dF dG = a0^3 sin^2 i da0 dcos(i) dOmega
                   domega, with two (Omega, omega) pairs per set of constants)
      'isotropic'  1 / (a0^4 sin^2 i): cos i, Omega, omega uniform and a0
                   log-uniform, for a0 >= a0_min; sin^2 i is floored at
                   sin2_floor, which keeps the density finite for face-on
                   orbits

For two or more orbits the periods are ordered and consecutive orbits are
separated (orbits_separated); the prior is the product of the single-orbit
priors truncated to that region.
"""

import numpy as np

import orbit


def orient_log_prior(A, B, F, G, orient_prior, a0_min, sin2_floor):
    """ln of the orientation density in (A, B, F, G), up to its normalisation
    (iso_log_norm): 0 for 'ti'; -4 ln a0 - ln max(sin^2 i, sin2_floor) for
    'isotropic', and -inf where a0 < a0_min. Element-wise on arrays."""
    if orient_prior == 'ti':
        return 0.0 if np.ndim(A) == 0 else np.zeros(np.shape(A))
    u  = A * A + B * B + F * F + G * G
    v  = A * G - B * F
    a2 = 0.5 * (u + np.sqrt(np.maximum(u * u - 4.0 * v * v, 0.0)))   # a0^2
    with np.errstate(divide='ignore', invalid='ignore'):
        ci = v / a2
        s2 = np.maximum(1.0 - ci * ci, sin2_floor)
        lp = -2.0 * np.log(a2) - np.log(s2)                          # -4 ln a0 - ln sin^2 i
    return np.where(a2 >= a0_min ** 2, lp, -np.inf) if np.ndim(A) else (
        float(lp) if a2 >= a0_min ** 2 else -np.inf)


def orbits_separated(Pa, ea, Pb, eb, mode, p_ratio_min):
    """True where orbit b (Pb > Pa) is far enough from orbit a.

    mode 'ratio': Pb / Pa >= p_ratio_min; 'noncrossing': the orbits do not
    cross, a_b (1 - e_b) > a_a (1 + e_a) with a ~ P^(2/3); 'none': always."""
    ratio = np.where((Pa > 0) & (Pb > 0), np.abs(Pb / np.where(Pa > 0, Pa, 1.0)), 0.0)
    if mode == 'ratio':
        return ratio >= p_ratio_min
    if mode == 'noncrossing':
        return ratio ** (2.0 / 3.0) * (1.0 - eb) > 1.0 + ea
    return np.ones(np.shape(Pa), dtype=bool)


def iso_log_norm(ti_max, a0_min, sin2_floor, n=2_000_000, seed=20261003):
    """ln of the integral of the isotropic orientation density 1 / (a0^4
    sin^2 i) (sin^2 i floored) over its support in (A, B, F, G): |A..G| <
    ti_max and a0 >= a0_min. By Monte Carlo in Campbell elements: with
    dA dB dF dG = a0^3 sin^2 i da0 dcos(i) dOmega domega and two (Omega, omega)
    per (A, B, F, G), the integral is 4 pi^2 ln(a_max / a0_min) E[g], g =
    sin^2 i / max(sin^2 i, floor) inside the box, 0 outside, for a0
    log-uniform to a_max = 2 ti_max (no larger a0 fits in the box)."""
    rng = np.random.default_rng(seed)
    lam = np.log(2.0 * ti_max / a0_min)
    a0 = a0_min * np.exp(rng.uniform(0.0, lam, n))
    c = rng.uniform(-1.0, 1.0, n)
    A, B, F, G = orbit.ti_from_campbell(a0, rng.uniform(0.0, 2 * np.pi, n),
                                         rng.uniform(0.0, 2 * np.pi, n), np.arccos(c))
    inside = (np.abs(A) < ti_max) & (np.abs(B) < ti_max) & (np.abs(F) < ti_max) & (np.abs(G) < ti_max)
    s2 = 1.0 - c * c
    return float(np.log(4.0 * np.pi ** 2 * lam * np.mean(inside * s2 / np.maximum(s2, sin2_floor))))


def sep_log_norm(n_planets, p_min, p_max, e_min, mode, p_ratio_min,
                 n=2_000_000, seed=20261004):
    """ln of the prior probability that n_planets orbits drawn independently
    from the single-orbit prior are in period order and separated: the
    normalisation of the truncated prior of two or more orbits. Monte Carlo."""
    if n_planets < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    P = p_min * (p_max / p_min) ** rng.uniform(0.0, 1.0, (n_planets, n))
    e = rng.uniform(e_min, 1.0, (n_planets, n))
    ok = np.ones(n, dtype=bool)
    for k in range(n_planets - 1):
        ok &= (P[k] < P[k + 1]) & orbits_separated(P[k], e[k], P[k + 1], e[k + 1],
                                                   mode, p_ratio_min)
    return float(np.log(np.mean(ok)))


def log_prior_orbit(p7, p_min, p_max, e_min, t0_lo, t0_hi, ti_max, orient_prior,
                    a0_min, sin2_floor, jeffreys_p=True):
    """ln prior density (unnormalised) of one orbit (P, e, T0, A, B, F, G);
    -inf outside the bounds."""
    P, e, T0, A, B, F, G = p7
    if not (p_min < P < p_max):
        return -np.inf
    if not (e_min <= e < 1.0):
        return -np.inf
    if not (t0_lo < T0 < t0_hi):
        return -np.inf
    for c in (A, B, F, G):
        if not (-ti_max < c < ti_max):
            return -np.inf
    lp = -np.log(P) if jeffreys_p else 0.0
    return lp + orient_log_prior(A, B, F, G, orient_prior, a0_min, sin2_floor)


def log_norm_orbit(p_min, p_max, e_min, t0_lo, t0_hi, ti_max, orient_prior,
                   a0_min, sin2_floor):
    """ln of the integral of exp(log_prior_orbit) over its support: subtract
    it to normalise the single-orbit prior."""
    return float(np.log(np.log(p_max / p_min)) + np.log(1.0 - e_min) + np.log(t0_hi - t0_lo)
                 + (iso_log_norm(ti_max, a0_min, sin2_floor) if orient_prior == 'isotropic'
                    else 4.0 * np.log(2.0 * ti_max)))
