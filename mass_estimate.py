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
Convert a fitted orbit (Thiele-Innes shape + period, plus optionally an RV
semi-amplitude) into a physical planet mass.

Neither the orbital shape nor the RV semi-amplitude alone fixes an
absolute mass — Kepler's third law only ever constrains a *ratio*
(M_planet / M_total for astrometry; M_planet·sin(i) tangled with M_total
for RV). An external mass — the host star's mass, from spectroscopy or
isochrone fitting — is required to anchor the scale, along with the
system's distance (parallax) to convert the fitted angular semi-major
axis to AU.

Two independent mass estimates are provided when both astrometry and RV
are available:
  1. Purely astrometric, from the photocenter wobble amplitude via
     Kepler's third law and the M_planet/M_total ratio. This one already
     carries the orbital inclination for free (no sin(i) ambiguity, unlike
     RV alone), since the shape of the fitted sky ellipse fixes i
     directly — see generate_sources.campbell_elements.
  2. From the RV mass function combined with that same astrometric
     inclination.
Agreement between the two is a useful consistency check on a candidate
signal.

Functions
---------
astrometric_planet_mass : Planet mass from the photocenter orbit + host star mass
rv_planet_mass          : Planet mass from the RV mass function + known inclination
estimate_planet_mass    : Convenience wrapper combining both when available
estimate_planet_mass_posterior : Vectorized version over posterior samples,
                          returning 16/50/84 percentiles for every derived
                          quantity instead of a single best-fit point value
"""

import numpy as np
from astropy import constants as const
from scipy.optimize import brentq

from generate_sources import campbell_elements

_G      = const.G.to('m3 / (kg s2)').value
_MSUN   = const.M_sun.to('kg').value
_MJUP   = const.M_jup.to('kg').value
_DAY_S  = 86400.0
_YR_DAY = 365.25


def _solve_mass_cubic(rhs, m_star_msun, sin_i=1.0):
    """Solve  (M_p sin_i)^3 / (M_star + M_p)^2 = rhs  for M_p (Msun), rhs >= 0.

    The left-hand side is monotonically increasing in M_p for M_p > 0, so
    the positive root is unique; found by bracketing and bisection
    (scipy.optimize.brentq) rather than the common M_p << M_star linearised
    approximation, so it stays valid for brown-dwarf/stellar-mass
    companions too.
    """
    if rhs <= 0:
        return 0.0

    def g(m_p):
        return (m_p * sin_i)**3 / (m_star_msun + m_p)**2 - rhs

    guess = (rhs * m_star_msun**2)**(1.0 / 3.0) / max(sin_i, 1e-6)
    hi = max(1e3 * guess, 1e3 * m_star_msun, 1.0)
    while g(hi) < 0:
        hi *= 10
    return brentq(g, 0.0, hi, xtol=1e-10, rtol=1e-12)


def astrometric_planet_mass(P_days, A, B, F, G, parallax_mas, m_star_msun):
    """Planet mass (Msun) from the astrometric photocenter orbit alone.

    Parameters
    ----------
    P_days       : orbital period (days)
    A, B, F, G   : Thiele-Innes constants of the fitted photocenter orbit (mas)
    parallax_mas : system parallax (mas)
    m_star_msun  : host star mass (Msun) — external input, not derivable
                   from the orbit fit itself

    Returns
    -------
    m_planet_msun : float
    a_mas         : angular semi-major axis of the photocenter orbit (mas)
    i_rad         : orbital inclination (radians)
    a_au          : linear semi-major axis of the photocenter orbit (AU)
    omega_rad     : argument of periastron (radians) -- the branch implied
                    by (A, B, F, G) alone; see estimate_planet_mass for the
                    K-sign correction when RV data picks the other branch
    Omega_rad     : longitude of ascending node (radians), in [0, pi): the
                    node consistent with omega_rad. When RV data pick the
                    other branch, both omega and Omega move by pi (see
                    estimate_planet_mass)
    """
    a_mas, omega_rad, Omega_rad, i_rad = campbell_elements(A, B, F, G)
    a_star_au    = a_mas / parallax_mas
    P_yr         = P_days / _YR_DAY

    rhs = a_star_au**3 / P_yr**2   # = M_p^3 / (M_star + M_p)^2  [Msun, AU, yr units]
    m_planet_msun = _solve_mass_cubic(rhs, m_star_msun, sin_i=1.0)
    return m_planet_msun, a_mas, i_rad, a_star_au, omega_rad, Omega_rad


def rv_planet_mass(P_days, K_ms, e, i_rad, m_star_msun):
    """Planet mass (Msun) from the RV mass function, using an independently
    known inclination (e.g. from astrometric_planet_mass) to break the
    M_p sin(i) degeneracy that RV data alone cannot resolve.

    Parameters
    ----------
    P_days      : orbital period (days)
    K_ms        : RV semi-amplitude (m/s)
    e           : eccentricity
    i_rad       : orbital inclination (radians), from astrometry
    m_star_msun : host star mass (Msun)

    Returns
    -------
    m_planet_msun : float
    """
    # K is a signed fit parameter (ExoPlanetLab.py's K_MIN is negative -- see
    # its comment: astrometry alone can't tell (Omega, omega) from
    # (Omega+pi, omega+pi), so a best-fit K<0 just means the orbit is on the
    # omega+pi branch). The RV mass function only cares about the amplitude.
    K_ms      = abs(K_ms)
    P_s       = P_days * _DAY_S
    f_mass_kg = P_s * K_ms**3 * (1.0 - e**2)**1.5 / (2.0 * np.pi * _G)
    f_mass_msun = f_mass_kg / _MSUN   # = (M_p sin i)^3 / (M_star + M_p)^2

    sin_i = np.sin(i_rad)
    return _solve_mass_cubic(f_mass_msun, m_star_msun, sin_i=sin_i)


def estimate_planet_mass(P_days, e, A, B, F, G, parallax_mas, m_star_msun,
                         K_ms=None, gamma_ms=None):
    """Combine astrometric (and, if available, RV) mass estimates for one planet.

    Parameters
    ----------
    P_days, e, A, B, F, G : best-fit orbital parameters (days, —, mas×4)
    parallax_mas          : system parallax (mas)
    m_star_msun           : host star mass (Msun) — external input
    K_ms                  : RV semi-amplitude (m/s), or None to skip the
                             RV-based estimate. May be negative -- see the
                             K_MIN comment in ExoPlanetLab.py: astrometry
                             alone can't distinguish the (Omega, omega) and
                             (Omega+pi, omega+pi) branches, so the fitter
                             lets K carry that sign. Reported here as
                             (|K|, omega+pi) when negative, matching the
                             literature convention of quoting K > 0.
    gamma_ms               : best-fit systemic velocity (m/s), pass-through
                             for display only, or None

    Returns
    -------
    dict with keys m_star_msun, parallax_mas, a_mas, a_au, P_yr, i_deg,
    Omega_deg, omega_deg, m_planet_astrometric_msun,
    m_planet_astrometric_mjup, and -- if K_ms is given -- K_ms (canonical,
    positive), rv_branch_flipped (bool), gamma_ms, m_planet_rv_msun,
    m_planet_rv_mjup.
    """
    m_ast_msun, a_mas, i_rad, a_au, omega_rad, Omega_rad = astrometric_planet_mass(
        P_days, A, B, F, G, parallax_mas, m_star_msun)

    out = dict(
        m_star_msun=m_star_msun,
        parallax_mas=parallax_mas,
        a_mas=a_mas,
        a_au=a_au,
        P_yr=P_days / _YR_DAY,
        i_deg=np.degrees(i_rad),
        Omega_deg=np.degrees(Omega_rad),
        omega_deg=np.degrees(omega_rad) % 360.0,
        m_planet_astrometric_msun=m_ast_msun,
        m_planet_astrometric_mjup=m_ast_msun * _MSUN / _MJUP,
    )

    if K_ms is not None:
        flipped = K_ms < 0.0
        out['K_ms']              = abs(K_ms)
        out['rv_branch_flipped'] = flipped
        if flipped:      # the (omega + pi, Omega + pi) branch: both angles move
            out['omega_deg'] = (out['omega_deg'] + 180.0) % 360.0
            out['Omega_deg'] = (out['Omega_deg'] + 180.0) % 360.0
        if gamma_ms is not None:
            out['gamma_ms'] = gamma_ms

        m_rv_msun = rv_planet_mass(P_days, K_ms, e, i_rad, m_star_msun)
        out['m_planet_rv_msun']  = m_rv_msun
        out['m_planet_rv_mjup']  = m_rv_msun * _MSUN / _MJUP

    return out


def estimate_planet_mass_posterior(P_days, e, A, B, F, G, parallax_mas, m_star_msun,
                                   K_ms=None):
    """Same physics as estimate_planet_mass, but vectorized over posterior
    draws: every argument is a 1-D array of the same length (one entry per
    posterior sample) instead of a single best-fit scalar.

    parallax_mas and m_star_msun may each be either a matching-length array
    (when they are themselves fit as free parameters with a Gaussian prior --
    see ExoPlanetLab.py's FIT_MSTAR_PARALLAX) or a plain scalar (fixed
    external input, broadcast to every draw; its uncertainty is then not
    propagated).

    campbell_elements is already elementwise-numpy, so (a_mas, omega_rad,
    Omega_rad, i_rad) come out vectorized for free; the mass cubic
    (brentq, scalar-only) is solved per draw in a plain Python loop -- a few
    hundred draws at microseconds each is negligible next to the MCMC itself.

    Returns
    -------
    dict of the same keys as estimate_planet_mass, but each value is a
    (16th, 50th, 84th) percentile tuple instead of a single float (booleans
    like rv_branch_flipped become the fraction of draws with that branch).
    """
    P_days, e, A, B, F, G = (np.asarray(x, dtype=float) for x in (P_days, e, A, B, F, G))
    n = len(P_days)
    parallax_arr = np.broadcast_to(np.asarray(parallax_mas, dtype=float), (n,))
    mstar_arr    = np.broadcast_to(np.asarray(m_star_msun, dtype=float), (n,))

    a_mas, omega_rad, Omega_rad, i_rad = campbell_elements(A, B, F, G)
    a_au = a_mas / parallax_arr
    P_yr = P_days / _YR_DAY

    m_ast_msun = np.empty(n)
    rhs = a_au**3 / P_yr**2
    for j in range(n):
        m_ast_msun[j] = _solve_mass_cubic(rhs[j], mstar_arr[j], sin_i=1.0)

    if K_ms is not None:
        K_ms   = np.asarray(K_ms, dtype=float)
        flipped = K_ms < 0.0
        K_abs   = np.abs(K_ms)
        omega_report = np.where(flipped, (omega_rad + np.pi) % (2.0 * np.pi),
                                 omega_rad % (2.0 * np.pi))
        Omega_rad = np.where(flipped, Omega_rad + np.pi, Omega_rad)   # the RV-selected node
        m_rv_msun = np.empty(n)
        for j in range(n):
            m_rv_msun[j] = rv_planet_mass(P_days[j], K_abs[j], e[j], i_rad[j], mstar_arr[j])
    else:
        omega_report = omega_rad % (2.0 * np.pi)

    def pct(x):
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        return (p16, p50, p84)

    out = dict(
        m_star_msun=pct(mstar_arr),
        parallax_mas=pct(parallax_arr),
        a_mas=pct(a_mas),
        a_au=pct(a_au),
        P_yr=pct(P_yr),
        i_deg=pct(np.degrees(i_rad)),
        Omega_deg=pct(np.degrees(Omega_rad)),
        omega_deg=pct(np.degrees(omega_report)),
        m_planet_astrometric_msun=pct(m_ast_msun),
        m_planet_astrometric_mjup=pct(m_ast_msun * _MSUN / _MJUP),
    )

    if K_ms is not None:
        out['K_ms']              = pct(K_abs)
        out['rv_branch_flipped'] = float(np.mean(flipped))   # fraction of draws
        out['m_planet_rv_msun']  = pct(m_rv_msun)
        out['m_planet_rv_mjup']  = pct(m_rv_msun * _MSUN / _MJUP)

    return out
