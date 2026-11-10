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
Exoplanet astrometric orbit library — Keplerian model in Thiele-Innes form.

Functions
---------
kepler_eq                : Solve Kepler's equation  M = E − e sin E
astrometric_orbit        : (ΔRA, ΔDec) displacement at given epochs
orbital_model_AL         : Along-scan signal from 7-parameter TI model
omega_from_ti            : Argument of periastron ω derived from Thiele-Innes constants
rv_model_single          : Keplerian radial-velocity model for one planet
build_gaia_covariance    : 5×5 astrometric covariance from Gaia catalogue columns
parallax_factors         : Per-epoch parallax factors from Earth barycentric position
astro_along_scan_variance: Per-epoch along-scan variance from astrometric uncertainty

References
----------
Thiele-Innes parameterisation:
  Green (1985), Spherical Astronomy, Cambridge University Press.
Gaia astrometric model:
  Lindegren et al. (2021), A&A 649, A2.
Variance propagation through the astrometric solution:
  Brandt (2021), ApJS 254, 42.
"""

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric


def kepler_eq(M, e, tol=1e-10):
    """Solve Kepler's equation M = E − e sin E (element-wise; e may broadcast).

    Halley (third-order) iterations from the second-order starter
    E0 = M + e sin M (1 + e cos M), with one sin/cos evaluation per step.
    Converges to |M - E + e sin E| ~ 1e-14 for every 0 <= e < 1 in 2-4 steps.
    (The previous Newton iteration from E0 = M diverged for e >~ 0.99 -- it
    returned E wrong by up to ~1e20 rad there, so the likelihood near e = 1
    was garbage -- and needed 5-8 steps at moderate e.)
    """
    M = np.asarray(M, dtype=float)
    E = M + e * np.sin(M) * (1.0 + e * np.cos(M))
    for _ in range(50):
        s, c = np.sin(E), np.cos(E)
        f   = E - e * s - M
        fp  = 1.0 - e * c
        dE  = -f / (fp - 0.5 * f * (e * s) / fp)
        E   = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


# Below this step size kepler_eq_sc updates sin E, cos E by a third-order
# rotation instead of recomputing them (error <= dE^4/24 ~ 4e-18).
_ROT_MAX = 1.0e-4


def kepler_eq_sc(M, e, tol=1e-10):
    """kepler_eq that also returns sin E and cos E: (E, sin E, cos E).

    Same Halley iteration and starter as kepler_eq, but sin E and cos E are
    carried along: once every correction is below _ROT_MAX they are updated
    by rotating through dE (third-order Taylor series of sin dE, cos dE)
    rather than recomputed, and the returned pair belongs to the final E, so
    callers need no further sin/cos (orbital_model_AL_vec). This removes about
    two of the five sin/cos evaluations per call -- the dominant cost of the
    vectorised likelihood. Agrees with kepler_eq to ~1e-15 rad.
    """
    M = np.asarray(M, dtype=float)
    E = M + e * np.sin(M) * (1.0 + e * np.cos(M))
    s, c = np.sin(E), np.cos(E)
    for _ in range(50):
        f   = E - e * s - M
        fp  = 1.0 - e * c
        dE  = -f / (fp - 0.5 * f * (e * s) / fp)
        E   = E + dE
        dmax = np.max(np.abs(dE))
        if dmax < _ROT_MAX:
            d2 = dE * dE
            sd = dE * (1.0 - d2 / 6.0)          # sin dE
            cd = 1.0 - 0.5 * d2                 # cos dE (next term d^4/24 < 5e-18)
            s, c = s * cd + c * sd, c * cd - s * sd
        else:
            s, c = np.sin(E), np.cos(E)
        if dmax < tol:
            break
    return E, s, c


def astrometric_orbit(t, P, e, T0, A, B, F, G):
    """Compute on-sky orbital displacement using Thiele-Innes constants.

    Parameters
    ----------
    t  : (N,) array — observation times in days
    P  : float  — orbital period (days)
    e  : float  — eccentricity ∈ [0, 1)
    T0 : float  — periastron epoch (days, same scale as t)
    A, B, F, G : float — Thiele-Innes constants (mas)

    Convention (standard Binnendijk / Gaia NSS): the orbit-plane x axis points
    NORTH and y points EAST, so
        delta_dec = A X + F Y          (north)
        delta_ra  = B X + G Y          (east; already includes cos(dec))
    with A = a0 (cos w cos W - sin w sin W cos i), B = a0 (cos w sin W +
    sin w cos W cos i), F = a0 (-sin w cos W - cos w sin W cos i),
    G = a0 (-sin w sin W + cos w cos W cos i), W = ascending-node longitude
    measured from north through east. This matches the published Gaia
    two-body-orbit constants and generate_sources.campbell_elements.

    Returns
    -------
    delta_ra  : (N,) along-RA displacement (mas)
    delta_dec : (N,) along-Dec displacement (mas)
    """
    M = 2.0 * np.pi * (t - T0) / P
    E = kepler_eq(M, e)
    X = np.cos(E) - e
    Y = np.sqrt(1.0 - e**2) * np.sin(E)
    delta_dec = A * X + F * Y
    delta_ra  = B * X + G * Y
    return delta_ra, delta_dec


class SingleStarProjector:
    """The operator (I - H) that a single-star astrometric fit applies to data.

    Gaia's DR4-style processing fits a linear single-star model (position,
    proper motion, parallax [+ pseudocolour]) to the raw along-scan centroids
    d and keeps the residuals r = d - X beta_hat = (I - H) d, where
        H = X (X^T W X + Lambda)^-1 X^T W
    (X: N x n_par design matrix, W: per-epoch weights, Lambda: prior precision
    on the colour parameter). If the truth is d = m(theta) + noise, then
    r = (I - H) m(theta) + (I - H) noise: the model for the *residuals* is the
    orbit with everything a single star could mimic (a linear drift, a
    parallax ellipse, ...) projected out. Comparing the bare orbit m(theta)
    with r is wrong whenever the orbit overlaps those terms -- above all for a
    period comparable to or longer than the baseline.

    Stored as X and the gain K = (X^T W X + Lambda)^-1 X^T W, so apply() costs
    two thin matrix products.
    """

    def __init__(self, design, gain):
        self.X = np.asarray(design, dtype=float)      # (N, n_par)
        self.K = np.asarray(gain, dtype=float)        # (n_par, N)
        self.n = self.X.shape[0]

    def apply(self, v):
        """(I - H) v for v of shape (N,) or (N, k)."""
        v = np.asarray(v, dtype=float)
        if v.shape[0] != self.n:
            raise ValueError(
                f"SingleStarProjector built for N={self.n} epochs, got a "
                f"vector of length {v.shape[0]} (must be the data epochs).")
        return v - self.X @ (self.K @ v)


def orbital_model_AL(t, params, scan_angles, projector=None):
    """Along-scan signal from the orbital wobble (7 Thiele-Innes parameters).

    Parameters
    ----------
    t           : (N,) observation times in days
    params      : sequence [P, e, T0, A, B, F, G]
    scan_angles : (N,) Gaia scan angles in radians
    projector   : SingleStarProjector or None. Pass it whenever the result is
                  compared with single-star-fit *residuals* (t must then be the
                  data epochs); None returns the bare physical signal.

    Returns
    -------
    along_scan : (N,) displacement in mas
    """
    P, e, T0, A, B, F, G = params
    dra, ddec = astrometric_orbit(t, P, e, T0, A, B, F, G)
    model = dra * np.sin(scan_angles) + ddec * np.cos(scan_angles)
    return model if projector is None else projector.apply(model)


def orbital_model_AL_vec(t, params, scan_angles):
    """Unprojected along-scan orbital signal for many parameter sets at once.

    params : (k, 7) array of [P, e, T0, A, B, F, G] rows (e.g. one per walker)
    Returns (k, N): row j equals orbital_model_AL(t, params[j], scan_angles).
    The single-star projection, if needed, is applied by the caller to the
    summed model of all planets (it is linear).
    """
    P, e, T0, A, B, F, G = (c[:, None] for c in np.asarray(params, dtype=float).T)
    _, sE, cE = kepler_eq_sc(2.0 * np.pi * (t[None, :] - T0) / P, e)
    X = cE - e
    Y = np.sqrt(1.0 - e ** 2) * sE
    return (B * X + G * Y) * np.sin(scan_angles) + (A * X + F * Y) * np.cos(scan_angles)


def astrometric_orbit_vec(t, params):
    """astrometric_orbit for many parameter sets: params (k, 7) rows of
    [P, e, T0, A, B, F, G]; returns (delta_ra, delta_dec), each (k, len(t))."""
    P, e, T0, A, B, F, G = (c[:, None] for c in np.asarray(params, dtype=float).T)
    _, sE, cE = kepler_eq_sc(2.0 * np.pi * (np.asarray(t, float)[None, :] - T0) / P, e)
    X = cE - e
    Y = np.sqrt(1.0 - e ** 2) * sE
    return B * X + G * Y, A * X + F * Y


def relative_orbit_vec(t, params, scale):
    """Position of a companion relative to its host star (mas; delta_ra* east,
    delta_dec north), (k, len(t)) each, for relative (imaging) astrometry.

    With mass fraction f_mass = M2 / (M1 + M2) and flux fraction
    f_flux = L2 / (L1 + L2), the host moves by -f_mass r and the unresolved
    photocentre by -(f_mass - f_flux) r about the barycentre, r being the
    companion's position relative to the host. The photocentre orbit of
    astrometric_orbit (a0) therefore gives
        r = -scale * photocentre,   scale = a_rel / a0 = 1 / (f_mass - f_flux) > 1:
    the same P, e, T0 and orientation, the opposite direction, larger by scale.
    """
    dra, ddec = astrometric_orbit_vec(t, params)
    s = np.asarray(scale, dtype=float).reshape(-1, 1)
    return -s * dra, -s * ddec


def relative_astrometry_lnlike(dx, dy, sx, sy, rho, jitter=0.0):
    """ln L of relative-astrometry residuals dx, dy (mas; (k, n) or (n,)) for
    points with errors sx, sy and correlation rho ((n,)): the sum over the
    points of two-dimensional Gaussians. A jitter ((k,) or scalar, mas) is
    added in quadrature to both coordinates of every point, leaving the
    covariance rho sx sy unchanged. Returns (k,) (a float for (n,) residuals).
    """
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    j2 = np.asarray(jitter, dtype=float) ** 2
    if j2.ndim:
        j2 = j2[:, None]
    vxx, vyy = np.asarray(sx, float) ** 2 + j2, np.asarray(sy, float) ** 2 + j2
    vxy = np.asarray(rho, float) * np.asarray(sx, float) * np.asarray(sy, float)
    det = vxx * vyy - vxy ** 2
    return np.sum(-np.log(2.0 * np.pi) - 0.5 * np.log(det)
                  - 0.5 * (vyy * dx * dx - 2.0 * vxy * dx * dy + vxx * dy * dy) / det, axis=-1)


def rv_model_vec(t, P, e, T0, K, omega):
    """rv_model_single for many parameter sets: P, e, T0, K, omega are (k,)
    arrays; returns (k, len(t))."""
    P, e, T0, K, omega = (np.asarray(v, dtype=float)[:, None] for v in (P, e, T0, K, omega))
    E  = kepler_eq(2.0 * np.pi * (t[None, :] - T0) / P, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E),
                           np.sqrt(1.0 - e) * np.cos(0.5 * E))
    return K * (np.cos(nu + omega) + e * np.cos(omega))


def implied_rv_scale(P, e, a0, inc):
    """c = K * parallax [m/s mas] of a star whose orbit is the photocentre
    orbit (a dark companion): K = 2 pi (a0 / parallax) sin i / (P sqrt(1 - e^2)),
    with P in days, a0 in mas and the inclination in radians."""
    au_km, day_s = 1.495978707e8, 86400.0
    return 2.0e3 * np.pi * au_km * a0 * np.sin(inc) / (P * day_s * np.sqrt(1.0 - e ** 2))


def coupled_k_lnterm(K, c, plx_hat, plx_sd, k_range):
    """ln of the factor that turns a uniform prior on K into the coupling of K
    to the astrometric orbit (ExoPlanetLab.py, EXOPLANETLAB_RV_COUPLED).

    Given the orbit, the parallax has the conditional distribution N(plx_hat,
    plx_sd^2) (the single-star fit with the orbit removed), and |K| = c /
    parallax (implied_rv_scale). K then has the density

        p(K) = 1/2 N(c / |K|; plx_hat, plx_sd^2) c / K^2,

    the 1/2 for the two nodes, which the sign of K selects. Sampling K with
    this density is the same as sampling the parallax, so the layout of the
    parameters does not change. Returned is ln [p(K) k_range], k_range being
    the width of the uniform prior that p(K) replaces; it integrates to
    k_range over K, so the evidence compares with that of the free-K model.
    -inf for K = 0."""
    K, c = np.abs(np.asarray(K, dtype=float)), np.asarray(c, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (c / K - plx_hat) / plx_sd
        out = (-0.5 * z * z - np.log(plx_sd * np.sqrt(2.0 * np.pi)) + np.log(c) - 2.0 * np.log(K)
               + np.log(k_range) - np.log(2.0))
    return np.where((K > 0.0) & np.isfinite(out), out, -np.inf)


def ti_from_campbell(a0, omega, Omega, inc):
    """Thiele-Innes constants (A, B, F, G) from Campbell elements: angular
    semi-major axis a0 (mas) and omega, Omega, inc (radians). The inverse of
    generate_sources.campbell_elements, in the convention of astrometric_orbit
    (x north, y east): A = a0 (cos w cos W - sin w sin W cos i), etc."""
    cw, sw, cW, sW, ci = np.cos(omega), np.sin(omega), np.cos(Omega), np.sin(Omega), np.cos(inc)
    A = a0 * (cw * cW - sw * sW * ci)
    B = a0 * (cw * sW + sw * cW * ci)
    F = a0 * (-sw * cW - cw * sW * ci)
    G = a0 * (-sw * sW + cw * cW * ci)
    return A, B, F, G


def omega_from_ti(A, B, F, G):
    """Argument of periastron ω (radians) derived from Thiele-Innes constants.

    Uses the identities
        tan(ω+Ω) = (B−F) / (A+G)
        tan(ω−Ω) = −(B+F) / (A−G)
    then  ω = [(ω+Ω) + (ω−Ω)] / 2, moved by π together with Ω so that Ω is in
    [0, π). Matches generate_sources.campbell_elements, which derives the same
    (ω, Ω, i, a0) from A, B, F, G.
    """
    wpo = np.arctan2(B - F, A + G)
    wmo = np.arctan2(-(B + F), A - G)
    shift = np.where(wpo - wmo < 0.0, np.pi, 0.0)
    return (0.5 * (wpo + wmo) + shift) % (2.0 * np.pi)


def rv_model_single(t, P, e, T0, K, omega):
    """Keplerian radial-velocity signal for one planet (m/s, systemic velocity excluded).

    V_r = K [cos(ν + ω) + e cos ω]

    Parameters
    ----------
    t     : (N,) observation times (days)
    P     : orbital period (days)
    e     : eccentricity ∈ [0, 1)
    T0    : periastron epoch (days)
    K     : RV semi-amplitude (m/s)
    omega : argument of periastron (radians), from omega_from_ti

    Returns
    -------
    rv : (N,) radial velocity contribution (m/s)
    """
    M  = 2.0 * np.pi * (t - T0) / P
    E  = kepler_eq(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E),
                           np.sqrt(1.0 - e) * np.cos(0.5 * E))
    return K * (np.cos(nu + omega) + e * np.cos(omega))


def build_gaia_covariance(star):
    """Build the 5×5 astrometric covariance matrix from Gaia catalogue columns.

    Parameter order: [ra, dec, pmra, pmdec, parallax]
    Units: mas (or mas/yr for proper motions).

    Parameters
    ----------
    star : astropy Table row (or compatible mapping) with Gaia gaia_source columns

    Returns
    -------
    C     : (5, 5) covariance matrix
    C_inv : (5, 5) precision matrix
    """
    sig = np.array([
        float(star['ra_error'][0]),
        float(star['dec_error'][0]),
        float(star['pmra_error'][0]),
        float(star['pmdec_error'][0]),
        float(star['parallax_error'][0]),
    ])

    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = float(star['ra_dec_corr'][0])
    corr[0, 2] = corr[2, 0] = float(star['ra_pmra_corr'][0])
    corr[0, 3] = corr[3, 0] = float(star['ra_pmdec_corr'][0])
    corr[0, 4] = corr[4, 0] = float(star['ra_parallax_corr'][0])
    corr[1, 2] = corr[2, 1] = float(star['dec_pmra_corr'][0])
    corr[1, 3] = corr[3, 1] = float(star['dec_pmdec_corr'][0])
    corr[1, 4] = corr[4, 1] = float(star['dec_parallax_corr'][0])
    corr[2, 3] = corr[3, 2] = float(star['pmra_pmdec_corr'][0])
    corr[2, 4] = corr[4, 2] = float(star['parallax_pmra_corr'][0])
    corr[3, 4] = corr[4, 3] = float(star['parallax_pmdec_corr'][0])

    C     = corr * np.outer(sig, sig)
    C_inv = np.linalg.inv(C)

    print("\nGaia astrometric formal uncertainties:")
    print(f"  sigma_ra     = {sig[0]:.4f} mas")
    print(f"  sigma_dec    = {sig[1]:.4f} mas")
    print(f"  sigma_pmra   = {sig[2]:.4f} mas/yr")
    print(f"  sigma_pmdec  = {sig[3]:.4f} mas/yr")
    print(f"  sigma_plx    = {sig[4]:.4f} mas")
    print(f"  det(C)       = {np.linalg.det(C):.3e}")

    return C, C_inv


def parallax_factors(t_obs, ra_deg, dec_deg, t0_jd):
    """Per-epoch parallax factors using Earth's barycentric position.

    Uses the Earth's barycentric position as a proxy for Gaia at L2
    (~1% error in the parallax factor — adequate for educational use).

    Parameters
    ----------
    t_obs   : (N,) observation times in days relative to t0_jd
    ra_deg  : float — target RA in degrees
    dec_deg : float — target Dec in degrees
    t0_jd   : float — JD corresponding to t_obs = 0

    Returns
    -------
    pf_ra  : (N,) parallax factor in RA direction
    pf_dec : (N,) parallax factor in Dec direction
    """
    ra_rad  = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    times = Time(t0_jd + t_obs, format='jd', scale='tcb')
    pos   = get_body_barycentric('earth', times)
    x = pos.x.to('AU').value
    y = pos.y.to('AU').value
    z = pos.z.to('AU').value

    pf_ra  = x * np.sin(ra_rad) - y * np.cos(ra_rad)
    pf_dec = (x * np.cos(ra_rad) * np.sin(dec_rad)
              + y * np.sin(ra_rad) * np.sin(dec_rad)
              - z * np.cos(dec_rad))

    return pf_ra, pf_dec


def astro_along_scan_variance(t_obs, scan_angles, C_astro, pf_ra, pf_dec):
    """Per-epoch along-scan variance from the 5-parameter astrometric uncertainty.

    σ²_astro,i = h_i^T C_astro h_i   where  h_i is the per-epoch design vector.

    See Lindegren et al. (2021), A&A 649, A2, Appendix A.

    Parameters
    ----------
    t_obs      : (N,) times in days
    scan_angles: (N,) scan angles in radians
    C_astro    : (5, 5) covariance matrix [mas², (mas/yr)²]
    pf_ra      : (N,) parallax factors in RA
    pf_dec     : (N,) parallax factors in Dec

    Returns
    -------
    sigma2_astro : (N,) per-epoch variance in mas²
    """
    dt_yr = (t_obs - t_obs[0]) / 365.25
    sin_a = np.sin(scan_angles)
    cos_a = np.cos(scan_angles)

    H = np.column_stack([
        sin_a,
        cos_a,
        dt_yr * sin_a,
        dt_yr * cos_a,
        pf_ra * sin_a + pf_dec * cos_a,
    ])
    return np.einsum('ni,ij,nj->n', H, C_astro, H)
