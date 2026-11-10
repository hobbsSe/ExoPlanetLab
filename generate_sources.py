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
generate_targets.py — planet and star definitions for ExoPlanetLab.

Two modes
---------
manual_targets()
    One or two hardcoded test planets with realistic Thiele-Innes parameters.
    Used as the default simulation ground truth (SOURCE_MODE = 'simulated').

query_gaia_source(source_id)
    Query Gaia DR3 for a star's astrometric parameters and NSS orbital solution.
    Returns (star_table, orbit_table). Not used by ExoPlanetLab.py (DR3 has
    no epoch astrometry, so there is no DR3 source mode).

Thiele-Innes parameterisation
------------------------------
An astrometric orbit is fully described by seven parameters:
  P   — orbital period (days)
  e   — eccentricity ∈ [0, 1)
  T0  — periastron epoch (days, same reference as t_obs)
  A, B, F, G — Thiele-Innes constants (mas)

The on-sky displacements are:
  Δα* = A X + F Y
  Δδ  = B X + G Y
where X = cos E − e,  Y = √(1−e²) sin E,  E = eccentric anomaly.

Campbell elements can be derived from A, B, F, G (see campbell_elements):
  a₀    = ((A+G)²+(B−F)²)^½/2 + ((A−G)²+(B+F)²)^½/2  [mas]
  cos i = (AG − BF) / a₀²
  ω, Ω  — from arctan combinations of A, B, F, G.
"""

import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────

def manual_targets():
    """Return hardcoded orbital parameters for test planets.

    Default: two synthetic planets around a nearby star.
    Planet 1 is a long-period giant (~ 4 yr) with moderate eccentricity.
    Planet 2 is a wider giant (~ 8 yr) with low eccentricity.

    To use a single planet, set N_PLANETS = 1 in ExoPlanetLab.py
    (rows are period-ordered; the first row is always used).

    Returns
    -------
    params : ndarray, shape (2, 7), rows sorted by ascending period
        Columns: [P (days), e, T0 (days), A (mas), B (mas), F (mas), G (mas)]
    """
    planet1 = np.array([
        1450.0,   # P    (days) ~ 3.97 yr
        0.54,     # e
        200.0,    # T0   (days from t_obs = 0)
        3.5,      # A    (mas)
        8.2,      # B    (mas)
       -4.1,      # F    (mas)
        2.8,      # G    (mas)
    ])

    planet2 = np.array([
        2900.0,   # P    (days) ~ 7.94 yr
        0.20,     # e
        800.0,    # T0   (days from t_obs = 0)
        1.8,      # A    (mas)
        3.5,      # B    (mas)
       -1.0,      # F    (mas)
        1.2,      # G    (mas)
    ])

    return np.array([planet1, planet2])


def query_gaia_source(source_id):
    """Query Gaia DR3 for a star's astrometric solution and NSS orbital parameters.

    Requires an active internet connection and astroquery.

    Parameters
    ----------
    source_id : int  Gaia DR3 source identifier

    Returns
    -------
    star  : astropy Table — Gaia gaia_source row (astrometry + correlations)
    orbit : astropy Table — Gaia nss_two_body_orbit row (P, e, T0, TI constants)
            or None if no NSS solution exists for this source.
    """
    from astroquery.gaia import Gaia

    star_job = Gaia.launch_job(f"""
        SELECT designation, ra, dec, pmra, pmdec, parallax,
               ra_error, dec_error, pmra_error, pmdec_error, parallax_error,
               ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
               dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
               parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
               phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
    """)
    star = star_job.get_results()

    orbit_job = Gaia.launch_job(f"""
        SELECT period, eccentricity, t_periastron,
               a_thiele_innes, b_thiele_innes,
               f_thiele_innes, g_thiele_innes,
               inclination, arg_periastron
        FROM gaiadr3.nss_two_body_orbit
        WHERE source_id = {source_id}
    """)
    orbit = orbit_job.get_results()
    if len(orbit) == 0:
        orbit = None

    return star, orbit


def orbit_to_params(orbit, t_ref_offset=0.0):
    """Convert a Gaia NSS orbit table row to a 7-element parameter vector.

    Parameters
    ----------
    orbit        : astropy Table from gaiadr3.nss_two_body_orbit
    t_ref_offset : float — subtract this value from T0 to shift to a local
                   time origin (e.g., set to t0_jd − 2451545.0 to shift from
                   JD to days from J2000.0, or 0 to keep Gaia TCB days).

    Returns
    -------
    params : ndarray, shape (7,)
        [P (days), e, T0 (days), A (mas), B (mas), F (mas), G (mas)]
    """
    return np.array([
        float(orbit['period'][0]),
        float(orbit['eccentricity'][0]),
        float(orbit['t_periastron'][0]) - t_ref_offset,
        float(orbit['a_thiele_innes'][0]),
        float(orbit['b_thiele_innes'][0]),
        float(orbit['f_thiele_innes'][0]),
        float(orbit['g_thiele_innes'][0]),
    ])


def campbell_elements(A, B, F, G):
    """Derive Campbell orbital elements from Thiele-Innes constants.

    Returns (a0, omega, Omega, inc) in (mas, rad, rad, rad), with Omega in
    [0, pi) and omega in [0, 2 pi): the pair that reproduces (A, B, F, G)
    through orbit.ti_from_campbell. Astrometry cannot tell it from
    (omega + pi, Omega + pi), which gives the same constants; radial
    velocities can (see the K sign in ExoPlanetLab.py and mass_estimate.py).

    omega and Omega are the half sum and half difference of omega + Omega and
    omega - Omega, each known modulo 2 pi, so the pair is first known up to
    adding pi to both; when Omega is reduced modulo pi, omega moves with it.
    """
    p = A + G;  s = B - F
    q = A - G;  r = B + F

    a_plus  = np.sqrt(p**2 + s**2) / 2.0
    a_minus = np.sqrt(q**2 + r**2) / 2.0

    a0    = a_plus + a_minus
    v     = A * G - B * F
    inc   = np.arccos(np.clip(v / a0**2, -1.0, 1.0))
    wpo, wmo = np.arctan2(s, p), np.arctan2(-r, q)      # omega + Omega, omega - Omega
    Omega0 = 0.5 * (wpo - wmo)                            # in (-pi, pi]
    shift  = np.where(Omega0 < 0.0, np.pi, 0.0)           # the same pi for both
    Omega  = (Omega0 + shift) % np.pi
    omega  = (0.5 * (wpo + wmo) + shift) % (2.0 * np.pi)

    return a0, omega, Omega, inc
