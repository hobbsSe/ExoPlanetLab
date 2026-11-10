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
gaia_dr4_epoch.py — load Gaia DR4 pre-release raw epoch astrometry and turn
it into the along-scan (AL) residual time series that orbit.py's Thiele-Innes
solvers consume.

Why this module exists
-----------------------
The DR4 pre-release VOTable (see data-release-4-tutorials/) gives the *raw*
per-CCD-transit along-scan centroid ``centroid_pos_al`` — not yet a residual.
To recover the orbital signal you first have to remove the single-star
(5p/6p) astrometric model (position, parallax, proper motion), the same
DR4-like fit the ``gaiasupdate`` package performs. This module drives that
fit via ``gaiasupdate`` and packages the matched per-epoch outputs
(time, AL residual, scan angle, uncertainty) that orbit.py expects.

Public API
----------
load_prerelease_table(path=None)
    Read the pre-release VOTable into a pandas DataFrame (all sources).
list_prerelease_sources(df=None, path=None)
    Return the distinct Gaia source_ids present in the table.
compute_al_residuals(df, source_id, model='6p_constrained_colour')
    Run the single-star fit for one source_id and return matched per-epoch
    arrays plus fit diagnostics.
load_source_epoch_data(source_id, path=None, model='6p_constrained_colour')
    Convenience: load the table and compute residuals for one source in a
    single call. This is what ExoPlanetLab.py's SOURCE_MODE =
    'gaia_dr4_prerelease' branch calls.
get_known_truth(source_id)
    Optional literature/NSS orbital-parameter lookup for comparison plots.
    Returns None for any source_id not explicitly added to KNOWN_TRUTH —
    which is the expected case for new, unpublished sources; a blind fit
    (no truth overlay) is the primary supported path, not a fallback.

Caveat
------
The residual-extraction path below calls a few of gaiasupdate's non-``_``-
prefixed but still fairly internal accessor methods (the same ones
``GaiaEpochAstrometryCu9.supdate`` itself uses) because the public
``supdate()`` convenience wrapper returns the fitted residuals but not the
per-epoch scan angle needed to hand them to orbit.py. It's pinned to
gaiasupdate==0.1.2's internal call sequence; re-verify against
``GaiaSourceEpochAstrometryCu9.compute_source_parameters`` if the package is
upgraded.

Note
----
``sigma_mas`` carries only the measurement (centroiding) uncertainty of each
CCD transit, not the uncertainty of the reference position propagated from the
covariance of the single-star solution, which the DR3 path
(``orbit.build_gaia_covariance``) adds. The orbit fits scale it, or add to it,
with their own noise terms.
"""

import os

import numpy as np
import pandas as pd
from astropy.table import Table


# ── Configuration ────────────────────────────────────────────────────────────

# Default location of the pre-release file; the path argument or
# EXOPLANETLAB_DR4_PRERELEASE_FILE override it.
_DEFAULT_RELATIVE_PATH = os.path.join(
    'data-release-4-tutorials', 'extracted_files',
    'GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml',
)


def _default_path():
    return os.environ.get(
        'EXOPLANETLAB_DR4_PRERELEASE_FILE',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _DEFAULT_RELATIVE_PATH),
    )


# Zero point of t_obs_day: J2017.5 TCB = JD 2457936.875, the DR4 reference
# epoch (gaiasupdate.constants.DR4_REFERENCE_EPOCH; Gaia Collaboration, Panuzzo
# et al. 2024, A&A 686, L2, Table 2 note a) -- not J2016.0, the reference
# epoch of DR3. Use it to convert a published periastron
# JD into a KNOWN_TRUTH T0.
DR4_EPOCH_REF_JD = 2457936.875   # J2017.5, TCB


# Published orbits, for comparison only (overlays and the "True" column of the
# summaries): source_id -> [P_days, e, T0_days, A, B, F, G (mas)], with
# T0_days = T_p (JD) - DR4_EPOCH_REF_JD. Never used to initialise, bound or fix
# a fit: every real-data fit is seeded from the data. Sources without an entry
# are fitted blind.
#   1457486023639239296 (Gaia-4): Stefansson et al. 2025, AJ 169, 107, Table 2,
#     "Gaia+RV" (adopted) column.
#   4318465066420528000 (Gaia BH3): Gaia Collaboration et al. 2024, A&A 686,
#     L2, Table 2, "Combined solution"; Campbell elements converted to
#     Thiele-Innes constants in the convention of orbit.astrometric_orbit.
#   3937211745905473024 (HD 114762): no entry -- no astrometric orbit has been
#     published (Kiefer et al. 2019 derive an inclination from the DR1 excess
#     noise).
KNOWN_TRUTH = {
    1457486023639239296: np.array(
        [571.3, 0.338, -550.775, 0.258, -0.103, -0.047, -0.116]),
    4318465066420528000: np.array(
        [4253.1, 0.7291, 240.515, 2.162385, 10.936403, 20.759550, -16.960855]),
}


def get_known_truth(source_id):
    """Return the literature/NSS truth vector for source_id, or None.

    Parameters
    ----------
    source_id : int

    Returns
    -------
    ndarray, shape (7,), or None
        [P (days), e, T0 (days), A, B, F, G (mas)] if present in
        KNOWN_TRUTH, else None (the default -- blind fit).
    """
    truth = KNOWN_TRUTH.get(int(source_id))
    return None if truth is None else np.asarray(truth, dtype=float)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_prerelease_table(path=None):
    """Read the DR4 pre-release epoch-astrometry VOTable into a DataFrame.

    Parameters
    ----------
    path : str, optional
        Path to the VOTable/XML export. Defaults to
        EXOPLANETLAB_DR4_PRERELEASE_FILE if set, else the file shipped
        under data-release-4-tutorials/extracted_files/.

    Returns
    -------
    pandas.DataFrame
        One row per CCD transit, all sources in the export.
    """
    path = path or _default_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Gaia DR4 pre-release epoch-astrometry file not found: {path!r}. "
            "Pass path=... explicitly or set EXOPLANETLAB_DR4_PRERELEASE_FILE."
        )
    return Table.read(path, format="votable").to_pandas()


def list_prerelease_sources(df=None, path=None):
    """Return the distinct Gaia source_ids present in the pre-release table.

    Parameters
    ----------
    df   : pandas.DataFrame, optional -- as returned by load_prerelease_table.
           Loaded from `path` if not given.
    path : str, optional -- forwarded to load_prerelease_table if df is None.

    Returns
    -------
    ndarray of int -- unique source_id values, ascending.

    Note
    ----
    Human-readable names/groups (e.g. "Gaia BH3", group "orbit") are only
    given in cosmos-datatable-export.pdf, not in the machine-readable
    export, so they aren't reproduced here.
    """
    if df is None:
        df = load_prerelease_table(path)
    return np.sort(df['source_id'].unique())


# ── Residual extraction ─────────────────────────────────────────────────────

def compute_al_residuals(df, source_id, model='6p_constrained_colour'):
    """Fit the DR4-like single-star model and return matched per-epoch arrays.

    Runs the same single-star (5p/6p) astrometric fit as the official DR4
    pipeline (via gaiasupdate) on this source's raw transits, then returns
    the post-fit along-scan residuals -- which, for an astrometric binary,
    contain the orbital wobble -- together with the scan angle and time of
    each retained epoch, matched row-for-row.

    Parameters
    ----------
    df        : pandas.DataFrame -- as returned by load_prerelease_table.
    source_id : int -- Gaia DR4 source identifier.
    model     : str -- '6p_constrained_colour' (default, DR4-like) or
                '5p_single_source'.

    Returns
    -------
    dict with keys
        t_obs_day       : (N,) ndarray -- time relative to the source's
                           reference epoch (days)
        along_scan_mas   : (N,) ndarray -- AL residual after the single-star
                           fit (mas); the orbital signal to fit
        scan_angle_rad   : (N,) ndarray -- Gaia scan position angle (rad)
        sigma_mas        : (N,) ndarray -- per-epoch centroiding uncertainty
                           (mas); deliberately the *measurement* variance,
                           not gaiasupdate's total_variance -- the latter
                           folds in the source's AGIS excess-noise estimate,
                           which for a genuine astrometric binary already
                           absorbs much of the orbital signal itself as
                           "noise" and would bias the orbit fit's error
                           budget.
        ra0_deg, dec0_deg, g_mag : float -- reference position and magnitude
        n_used, n_outliers : int -- epochs retained / AGIS-downweighted
        excess_noise_mas, significance : float -- single-star fit
                           diagnostics from gaiasupdate (large values are
                           expected for a real astrometric binary -- it
                           means the single-star model doesn't fit well,
                           which is exactly the signature you want to see
                           before handing the residuals to the orbit
                           solvers)
        resid_rms_mas    : float -- RMS of along_scan_mas (quick-look only)
        proj_design, proj_gain : (N, n_par) design matrix X and (n_par, N)
                           gain (X'WX+L)^-1 X'W of the single-star fit that
                           produced along_scan_mas: along_scan_mas =
                           (I - H) d with H = X @ gain. An orbit model must be
                           projected with the same (I - H) before it is
                           compared with along_scan_mas (orbit.
                           SingleStarProjector). Verified against the
                           residuals to proj_check_mas.

    Raises
    ------
    ValueError if source_id is not present in df.
    """
    from gaiasupdate.epoch_astrometry import (
        GaiaEpochAstrometryArchive, GaiaEpochAstrometryCu9,
        GaiaSourceEpochAstrometryCu9, GaiaCentroidDataFrame,
    )
    from gaiasupdate.solver import DesignEquation

    source_id = int(source_id)
    src_rows = df[df['source_id'] == source_id]
    if len(src_rows) == 0:
        raise ValueError(
            f"source_id={source_id} not found in this epoch-astrometry table. "
            "Use list_prerelease_sources() to see what's available."
        )
    ra0_deg  = float(src_rows['ra0'].iloc[0])
    dec0_deg = float(src_rows['dec0'].iloc[0])
    g_mag    = float(src_rows['g_mag'].iloc[0]) if 'g_mag' in src_rows else float('nan')

    # -- Convert to gaiasupdate's internal (CU9) representation and select
    #    this source, mirroring GaiaEpochAstrometryCu9.supdate()'s own
    #    setup so the filtering/fit below is identical to the public path.
    cu9_raw   = GaiaEpochAstrometryArchive.archive_to_cu9(df)
    is_exploded = GaiaCentroidDataFrame.check_if_is_exploded(
        GaiaEpochAstrometryCu9(cu9_raw), cu9_raw)
    cu9_filtered = GaiaEpochAstrometryCu9.filter_columns_for_computation(cu9_raw)
    epoch_all = GaiaEpochAstrometryCu9(cu9_filtered, is_exploded=is_exploded).epoch_data
    source_id_field = GaiaEpochAstrometryCu9(cu9_filtered, is_exploded=True)._source_id_field
    epoch_src = epoch_all[epoch_all[source_id_field] == source_id].copy()
    epoch_src.loc[:, 'colourFactorAl'] *= -1e3

    ea = GaiaSourceEpochAstrometryCu9.from_dataframe(epoch_src, source_id, is_exploded=True)

    # -- Same filtering + relative-time setup as
    #    GaiaSourceEpochAstrometryCu9.compute_source_parameters(), done here
    #    manually (rather than via that method, which only returns the
    #    solved-parameter dict) so epoch_data -- with scan angle intact --
    #    is still available afterwards to match against the residuals.
    epochastro = ea.epoch_data.epochastrometrycu9
    ea.epoch_data = ea.epoch_data.epochastrometrycu9.filter_on_used_by_agis()
    ea.epoch_data = ea.epoch_data.gaiacentroid.filter_null_from_column(
        epochastro._time_barycentric_correction_column)
    ea.epoch_data = ea.epoch_data.gaiacentroid.filter_null_from_column(
        epochastro._scan_angle_column)
    ea.epoch_data.epochastrometrycu9.set_relative_time()

    design_parameters = ea.epoch_data.epochastrometrycu9.get_design_equation_parameters(model=model)
    n_linear = design_parameters['design_matrix_coefficients'].shape[1]
    if model == '6p_constrained_colour':
        if n_linear != 6:
            raise RuntimeError(f"Expected 6 linear parameters for {model!r}, got {n_linear}.")
        design_parameters['gaussian_priors'] = np.array([None] * (n_linear - 1) + [0.085e-3])
    elif model == '5p_single_source':
        if n_linear != 5:
            raise RuntimeError(f"Expected 5 linear parameters for {model!r}, got {n_linear}.")
    else:
        raise ValueError(f"Unknown model={model!r}.")

    design_equation = DesignEquation(design_parameters)
    excess_noise_mas = float(ea.epoch_data.iloc[0]['agisSourceExcessNoise'])
    total_variance = design_equation.observation_variances + excess_noise_mas**2
    results = design_equation.solve(solver='agis', total_variance=total_variance)

    index_keep = results['index_keep']
    matched = ea.epoch_data.iloc[index_keep]

    t_obs_day     = matched['relative_time_day'].to_numpy(dtype=float)
    scan_angle_rad = np.radians(matched['scanPosAngle'].to_numpy(dtype=float))
    along_scan_mas = np.asarray(results['residuals'], dtype=float)
    # Measurement precision only (see the module docstring).
    sigma_mas = np.sqrt(np.asarray(results['measurement_variance'], dtype=float))

    # Experimental: EXOPLANETLAB_ASTRO_GATE_F="0:1.81,10:0.86" scales
    # the errors of the measurements taken with the listed gates (0 = no
    # gate). The gate is not among the columns kept above, so it is looked up
    # in the raw table by the observation time.
    gate_f = os.environ.get('EXOPLANETLAB_ASTRO_GATE_F', '').strip()
    if gate_f:
        factor = {int(k): float(v) for k, v in (kv.split(':') for kv in gate_f.split(','))}

        def _flat(col):
            return np.concatenate([np.ma.filled(np.ma.asarray(v, dtype=float), np.nan)
                                   for v in src_rows[col]])

        gate_of = dict(zip(_flat('obs_time_tcb'), _flat('gates')))
        gate = np.array([gate_of[v] for v in matched['obsTimeTcb'].to_numpy(dtype=float)])
        scale = np.array([factor.get(int(v), 1.0) for v in gate])
        sigma_mas = sigma_mas * scale
        print(f"  EXOPLANETLAB_ASTRO_GATE_F: errors of {int(np.sum(scale != 1.0))} of "
              f"{len(scale)} measurements scaled ({gate_f})")

    # -- The residuals are (I - H) d for the weighted single-star fit just
    #    solved (see orbit.SingleStarProjector). Rebuild the ingredients of H so
    #    that the orbit model can be projected the same way, and verify that
    #    the rebuild reproduces these residuals.
    X_all   = np.asarray(design_parameters['design_matrix_coefficients'], dtype=float)
    d_all   = np.asarray(design_parameters['dependent_variable'], dtype=float)
    n_total = X_all.shape[0]

    def _kept(a):
        a = np.asarray(a, dtype=float)
        return a[index_keep] if a.shape[0] == n_total else a

    proj_X   = X_all[index_keep]
    proj_d   = d_all[index_keep]
    tot_var  = _kept(results['total_variance'])
    down_w   = _kept(results['weights'])
    priors   = design_parameters.get('gaussian_priors')
    lam      = np.zeros(proj_X.shape[1])
    if priors is not None:
        for j, pv in enumerate(priors):
            if pv is not None:
                lam[j] = 1.0 / float(pv) ** 2

    proj_gain, proj_err, proj_scheme, proj_wgt = None, np.inf, None, None
    for scheme, wgt in (('weights/total_variance', down_w / tot_var),
                        ('1/total_variance', 1.0 / tot_var)):
        normal = proj_X.T @ (wgt[:, None] * proj_X) + np.diag(lam)
        gain   = np.linalg.solve(normal, proj_X.T * wgt[None, :])
        err    = float(np.max(np.abs(proj_d - proj_X @ (gain @ proj_d) - along_scan_mas)))
        if err < proj_err:
            proj_gain, proj_err, proj_scheme, proj_wgt = gain, err, scheme, wgt
        if err < 1e-6:
            break
    if proj_err > 1e-6:
        raise RuntimeError(
            f"Could not reproduce gaiasupdate's residuals from the single-"
            f"star design matrix (best max |diff| = {proj_err:.3e} mas with "
            f"weighting {proj_scheme!r}); the orbit model cannot be projected "
            f"consistently with them for source_id={source_id}.")

    n = len(t_obs_day)
    if not (len(along_scan_mas) == n and len(scan_angle_rad) == n and len(sigma_mas) == n):
        raise RuntimeError(
            "Internal mismatch between gaiasupdate residuals and matched "
            f"epoch metadata for source_id={source_id} "
            f"(t={len(t_obs_day)}, resid={len(along_scan_mas)}, "
            f"scan={len(scan_angle_rad)}, sigma={len(sigma_mas)})."
        )

    return {
        't_obs_day':        t_obs_day,
        'along_scan_mas':   along_scan_mas,
        'scan_angle_rad':   scan_angle_rad,
        'sigma_mas':        sigma_mas,
        'ra0_deg':          ra0_deg,
        'dec0_deg':         dec0_deg,
        'g_mag':            g_mag,
        'n_used':           n,
        'n_outliers':       int(results['n_outliers']),
        'excess_noise_mas': excess_noise_mas,
        'significance':     results.get('significance'),
        'resid_rms_mas':    float(np.std(along_scan_mas)),
        'proj_design':      proj_X,      # (N, n_par) single-star design matrix
        'proj_gain':        proj_gain,   # (n_par, N) (X'WX+L)^-1 X'W
        'proj_check_mas':   proj_err,    # max |rebuilt - gaiasupdate residual|
        # Ingredients to solve for the single-star parameters beta given an
        # orbit model s:  beta_hat = N^-1 D^T W (d - s),  cov = N^-1,  with
        # N = D^T W D + diag(Lambda). Columns of D (and elements of beta):
        # proj_param_names, in mas and mas/yr (colour: pseudo-colour units).
        'proj_data':        proj_d,      # (N,) raw AL centroid offsets d [mas]
        'proj_weights':     proj_wgt,    # (N,) the fit's weights W
        'proj_prior_prec':  lam,         # (n_par,) Lambda (colour prior)
        'proj_param_names': (['ra_offset', 'dec_offset', 'parallax', 'pmra', 'pmdec',
                              'pseudocolour'][:proj_X.shape[1]]),
        'source_id':        source_id,
        'model':            model,
    }


def load_source_epoch_data(source_id, path=None, model='6p_constrained_colour'):
    """Load the pre-release table and compute AL residuals for one source.

    Convenience wrapper combining load_prerelease_table() and
    compute_al_residuals() -- this is what ExoPlanetLab.py's SOURCE_MODE =
    'gaia_dr4_prerelease' branch calls.

    Parameters
    ----------
    source_id : int
    path      : str, optional -- forwarded to load_prerelease_table.
    model     : str -- forwarded to compute_al_residuals.

    Returns
    -------
    dict -- see compute_al_residuals.
    """
    df = load_prerelease_table(path)
    return compute_al_residuals(df, source_id, model=model)
