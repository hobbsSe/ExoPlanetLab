# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Hipparcos intermediate astrometric data (IAD) for a joint Hipparcos + Gaia DR4
epoch-astrometry fit. OPT-IN (EXOPLANETLAB_HIPPARCOS=1), see the
warning below.

Data
----
The 2007 re-reduction (van Leeuwen 2007), as exported from the Hipparcos 2
Java tool: ESA's ResRec_JavaTool_2014.zip (van Leeuwen & Michalik 2021; 350 MB,
one ASCII file per star; https://www.cosmos.esa.int/web/hipparcos/hipparcos-2),
read directly from the zip. Each record is one field transit: epoch (Julian
years from J1991.25), along-scan parallax factor PARF, scan direction (CPSI,
SPSI), abscissa residual RES against the catalogue solution, and its error
SRES. Following Octofitter / htof:
  - records with SRES <= 0 were rejected by the reduction and are dropped;
  - the errors are de-normalised (Nielsen et al. 2020, AJ 159, 71, Eq. 10),
    since the catalogue rescaled them to its own goodness of fit;
  - the Java-tool export sometimes repeats trailing rows; these are detected
    and the affected orbits removed as in G. M. Brandt et al. 2021 (AJ 162,
    186), Appendix A (htof.parse.find_epochs_to_reject_java).
The catalogue solution (5, 7 or 9 parameters; other types are refused) is added
back to the residuals, giving the measured abscissa relative to the catalogue
position at J1991.25:
    d_i = RES_i + PARF_i plx + CPSI_i x(t_i) + SPSI_i y(t_i),
    x(t) = pmra t + dpmra t^2/2 + ddpmra t^3/6  (and y likewise), t in years.

Model
-----
    d_i = CPSI_i [dx + pmra (t_i - T) + e_x + w_x t_i]
        + SPSI_i [dy + pmdec (t_i - T) + e_y + w_y t_i] + PARF_i plx + orbit_i

with (dx, dy, plx, pmra, pmdec) the SAME single-star terms as the Gaia DR4 fit
(position offsets from the Gaia reference position at J2017.5, absolute
parallax and proper motion), T = 26.25 yr from J1991.25 to J2017.5, and the
data shifted by the constant position of the Gaia reference path at J1991.25
relative to the Hipparcos catalogue position (rigorous propagation with the
radial velocity, i.e. including perspective acceleration; hipparcos.
reference_offset). (e_x, e_y, w_x, w_y) are the Hipparcos-to-Gaia frame
offset at J1991.25 and spin, per star, with zero-mean Gaussian priors of
0.6 mas and 0.25 mas/yr per axis: the stated accuracy of the Hipparcos link to
the ICRS (Kovalevsky et al. 1997), a conservative choice that blurs rather
than biases. All of these are linear and join the Gaia single-star terms in
robust.StudentTSingleStar's refit at every likelihood evaluation. The orbit
term is the same photocentre orbit as for Gaia, projected on the Hipparcos
scan direction: CPSI dra* + SPSI ddec.

WARNING -- systematics
----------------------
The Hipparcos IAD have known, only partly understood problems: the error
model (the reduction's errors need de-normalisation and are still not
reliable: this module gives Hipparcos its own Student-t nu and f), corrupted
exported records (repaired heuristically, above), the frame (only as good as
the priors above), and the band: the Hp photocentre orbit equals the G-band
one only for a dark companion; a luminous companion makes the two differ.
Results depend on all of these. Always compare with the Gaia-only fit.
"""

import itertools
import os
import zipfile

import numpy as np

HIP_EPOCH_JD = 2448349.0625      # J1991.25 (TT)
GAIA_DR4_EPOCH_JD = 2457936.875  # J2017.5 (TCB); gaia_dr4_epoch.DR4_EPOCH_REF_JD
T_HIP_TO_GAIA_YR = (GAIA_DR4_EPOCH_JD - HIP_EPOCH_JD) / 365.25     # 26.25
FRAME_POS_SIGMA = 0.6            # mas, per axis, at J1991.25
FRAME_SPIN_SIGMA = 0.25          # mas/yr, per axis
DEFAULT_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'HIPdata', 'ResRec_JavaTool_2014.zip')
ZIP_URL = ('https://www.cosmos.esa.int/documents/532822/6470227/ResRec_JavaTool_2014.zip/'
           'a58ad12e-cffb-f959-0ed5-2ae26899f61a?t=1631109433177&download=true')
FRAME_NAMES = ['hip_frame_ra', 'hip_frame_dec', 'hip_spin_ra', 'hip_spin_dec']


def _num(s):
    return np.nan if s.strip() in ('---', '') else float(s)


def read_iad(hip_id, zip_path=DEFAULT_ZIP):
    """Header and records of one star, as in the file (nothing corrected)."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"{zip_path} not found. Download ESA's Hipparcos 2 IAD archive (350 MB) with\n"
            f"  mkdir -p HIPdata && curl -L -o HIPdata/ResRec_JavaTool_2014.zip '{ZIP_URL}'\n"
            f"or set EXOPLANETLAB_HIP_IAD_ZIP.")
    name = f"ResRec_JavaTool_2014/H{hip_id:06d}"[:len('ResRec_JavaTool_2014/') + 4]
    name = f"{name}/H{hip_id:06d}.d"
    with zipfile.ZipFile(zip_path) as z:
        try:
            text = z.read(name).decode()
        except KeyError:
            raise KeyError(f"HIP {hip_id}: no file {name} in {zip_path}") from None
    lines = text.splitlines()
    h1 = lines[6].lstrip('#').split()
    h2 = lines[8].lstrip('#').split()
    h3 = lines[10].lstrip('#').split()
    keys3 = ['radeg', 'dedeg', 'plx', 'pmra', 'pmde', 'e_ra', 'e_de', 'e_plx', 'e_pmra',
             'e_pmde', 'dpmra', 'dpmde', 'e_dpmra', 'e_dpmde', 'ddpmra', 'ddpmde',
             'e_ddpmra', 'e_ddpmde', 'upsra', 'upsde', 'e_upsra', 'e_upsde', 'var']
    hdr = dict(hip=int(h1[0]), mce=int(h1[1]), nres=int(h1[2]), nc=int(h1[3]),
               isol_n=int(h1[4]), sce=int(h1[5]), f2=float(h1[6]), f1=float(h1[7]),
               hp=float(h2[0]), b_v=float(h2[1]), varann=int(h2[2]), nob=int(h2[3]),
               nr=int(h2[4]))
    hdr.update({k: _num(v) for k, v in zip(keys3, h3)})
    rec = np.array([[float(x) for x in ln.split()] for ln in lines[12:]
                    if ln.strip() and not ln.startswith('#')])
    cols = dict(iorb=rec[:, 0].astype(int), t=rec[:, 1], parf=rec[:, 2], cpsi=rec[:, 3],
                spsi=rec[:, 4], res=rec[:, 5], sres=rec[:, 6])
    return hdr, cols


def _corrupted_tail(c):
    """Number of repeated trailing rows (Brandt et al. 2021, App. A): the last
    four rows from one orbit whose errors read as a duplicated pair."""
    if len(c['sres']) < 4:
        return 0
    o, s = c['iorb'][-4:], c['sres'][-4:]
    if np.all(o == o[0]) and abs(s[0] - s[3]) < 1e-4 and abs(s[1] - s[2]) < 0.1:
        return 3
    return 0


def _repair_corruption(c, n_bad):
    """Drop the n_bad repeated residuals from the tail and the n_bad orbits'
    geometry that makes the catalogue solution most nearly stationary (the chi^2
    gradient closest to zero), then re-pair the rest (htof's procedure)."""
    n = len(c['t'])
    keep_res = np.ones(n, bool)
    keep_res[n - n_bad:] = False
    rf = (c['res'] / c['sres'] ** 2)[keep_res]
    geo = np.column_stack([c['parf'], c['cpsi'], c['spsi'],
                           c['t'] * c['cpsi'], c['t'] * c['spsi']])
    best, best_val = None, np.inf
    for drop in itertools.combinations(range(n), n_bad):
        keep = np.ones(n, bool)
        keep[list(drop)] = False
        val = np.linalg.norm(2.0 * rf @ geo[keep])
        if val < best_val:
            best, best_val = drop, val
    keep_geo = np.ones(n, bool)
    keep_geo[list(best)] = False
    out = {k: c[k][keep_geo] for k in ('iorb', 't', 'parf', 'cpsi', 'spsi')}
    out.update(res=c['res'][keep_res], sres=c['sres'][keep_res])
    return out, best


def load_iad(hip_id, zip_path=DEFAULT_ZIP):
    """Cleaned records with the measured abscissa d (mas) relative to the
    catalogue position at J1991.25. Returns (hdr, data, notes)."""
    hdr, c = read_iad(hip_id, zip_path)
    notes = []
    if hdr['isol_n'] not in (5, 7, 9):
        raise ValueError(f"HIP {hip_id}: catalogue solution type {hdr['isol_n']} "
                         f"(only 5, 7 and 9 are supported: the residuals of other "
                         f"types are relative to a model that cannot be rebuilt here)")
    rej = c['sres'] <= 0
    n_bad = _corrupted_tail({k: v for k, v in c.items()})
    if n_bad:
        c, dropped = _repair_corruption(c, n_bad)
        notes.append(f"corrupted trailing records repaired (Brandt et al. 2021 App. A): "
                     f"{n_bad} repeated residuals and orbits {list(dropped)} removed")
        rej = c['sres'] <= 0
    if rej.any():
        notes.append(f"{int(rej.sum())} records rejected by the reduction dropped")
        c = {k: v[~rej] for k, v in c.items()}
    # Nielsen et al. (2020) Eq. 10: undo the catalogue's error renormalisation
    D = hdr['nres'] - hdr['isol_n']
    fac = (hdr['f2'] * np.sqrt(2.0 / (9 * D)) + 1.0 - 2.0 / (9 * D)) ** 1.5
    c['sigma'] = c['sres'] * fac
    notes.append(f"errors de-normalised (Nielsen et al. 2020): x {fac:.3f} (F2 = {hdr['f2']:+.2f})")
    t = c['t']
    acc = lambda a, b: (np.nan_to_num(a) * t ** 2 / 2 + np.nan_to_num(b) * t ** 3 / 6)
    x = hdr['pmra'] * t + (acc(hdr['dpmra'], hdr['ddpmra']) if hdr['isol_n'] > 5 else 0.0)
    y = hdr['pmde'] * t + (acc(hdr['dpmde'], hdr['ddpmde']) if hdr['isol_n'] > 5 else 0.0)
    c['d'] = c['res'] + c['parf'] * hdr['plx'] + c['cpsi'] * x + c['spsi'] * y
    c['t_jd'] = HIP_EPOCH_JD + 365.25 * t
    return hdr, c, notes


def reference_offset(hdr, ra0_deg, dec0_deg, plx_mas, pmra, pmdec, rv_kms):
    """Position (dx, dy) [mas] at J1991.25 of the Gaia reference path -- the
    source at (ra0, dec0) at J2017.5 moving with (pmra, pmdec, rv) at the
    distance 1/plx -- relative to the Hipparcos catalogue position, in the
    tangent plane there. Rigorous (astropy), so it includes the perspective
    acceleration that a linear back-propagation over 26 yr would miss.
    Returns dx, dy, the path's proper motion (pmra*, pmdec) at J1991.25, and
    the size [mas] of the perspective-acceleration term (rv versus rv = 0)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    c0 = SkyCoord(ra=ra0_deg * u.deg, dec=dec0_deg * u.deg,
                  distance=(1000.0 / max(plx_mas, 1e-3)) * u.pc,
                  pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                  radial_velocity=rv_kms * u.km / u.s,
                  obstime=Time(GAIA_DR4_EPOCH_JD, format='jd', scale='tcb'))
    c1 = c0.apply_space_motion(new_obstime=Time(HIP_EPOCH_JD, format='jd', scale='tt'))
    xi, eta = _tangent(c1.ra.deg, c1.dec.deg, hdr['radeg'], hdr['dedeg'])
    # the same with rv = 0: the size of the perspective-acceleration term
    c0n = SkyCoord(ra=ra0_deg * u.deg, dec=dec0_deg * u.deg,
                   distance=(1000.0 / max(plx_mas, 1e-3)) * u.pc,
                   pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                   radial_velocity=0.0 * u.km / u.s,
                   obstime=Time(GAIA_DR4_EPOCH_JD, format='jd', scale='tcb'))
    c1n = c0n.apply_space_motion(new_obstime=Time(HIP_EPOCH_JD, format='jd', scale='tt'))
    persp = float(c1.separation(c1n).to(u.mas).value)
    # proper motion of the reference path at J1991.25 (changed by the
    # perspective acceleration too)
    pm1 = (float(c1.pm_ra_cosdec.to(u.mas / u.yr).value), float(c1.pm_dec.to(u.mas / u.yr).value))
    return float(xi), float(eta), pm1, persp


def _tangent(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic (xi, eta) [mas] of (ra, dec) about the tangent point (ra0, dec0)."""
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    a0, d0 = np.radians(ra0_deg), np.radians(dec0_deg)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    mas = np.degrees(1.0) * 3.6e6
    return xi * mas, eta * mas


def sky_path(ra_deg, dec_deg, epoch_jd, epoch_scale, plx_mas, pmra, pmdec, rv_kms,
             t_jd, t_scale, tan_ra_deg, tan_dec_deg):
    """Barycentric path (no parallax) of a star at (ra, dec) at epoch_jd moving
    with (pmra*, pmdec, rv) at the distance 1/plx, at the times t_jd, as
    tangent-plane (xi, eta) [mas] about (tan_ra, tan_dec). Rigorous (astropy),
    for synthetic single-star motion in injections (EXOPLANETLAB_INJECT_SS)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    c0 = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                  distance=(1000.0 / max(plx_mas, 1e-3)) * u.pc,
                  pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                  radial_velocity=rv_kms * u.km / u.s,
                  obstime=Time(epoch_jd, format='jd', scale=epoch_scale))
    c1 = c0.apply_space_motion(new_obstime=Time(np.asarray(t_jd, float), format='jd',
                                                scale=t_scale))
    return _tangent(c1.ra.deg, c1.dec.deg, tan_ra_deg, tan_dec_deg)


def design_rows(c, n_gaia_par, dx0, dy0, pmra_ref, pmdec_ref, pm_hip=None):
    """Hipparcos rows of the joint single-star design matrix and their data.

    Columns: the Gaia terms (ra_offset, dec_offset, parallax, pmra, pmdec
    [, pseudocolour = 0]) followed by FRAME_NAMES. The data are the measured
    abscissae minus the reference path's known part. The true path is the
    reference path R plus the Gaia offsets plus (pm - pm_ref)(t - T), and near
    J1991.25 R(t) = (dx0, dy0) + pm_hip t (pm_hip: the reference path's proper
    motion there, pm_ref changed by perspective acceleration), so
        path = [(dx0, dy0) + pm_hip t - pm_ref (t - T)] + offsets + pm (t - T):
    linear in the Gaia terms, the bracket subtracted from the data (the error
    of linearising about pm_ref is < 1e-4 mas)."""
    t, cp, sp, pf = c['t'], c['cpsi'], c['spsi'], c['parf']
    T = T_HIP_TO_GAIA_YR
    n = len(t)
    Xg = np.zeros((n, n_gaia_par))
    Xg[:, 0], Xg[:, 1], Xg[:, 2] = cp, sp, pf
    Xg[:, 3], Xg[:, 4] = cp * (t - T), sp * (t - T)
    Xf = np.column_stack([cp, sp, cp * t, sp * t])
    ph = (pmra_ref, pmdec_ref) if pm_hip is None else pm_hip
    d = (c['d'] - cp * (dx0 + ph[0] * t - pmra_ref * (t - T))
         - sp * (dy0 + ph[1] * t - pmdec_ref * (t - T)))
    return Xg, Xf, d


def frame_prior_precision():
    return np.array([1 / FRAME_POS_SIGMA ** 2] * 2 + [1 / FRAME_SPIN_SIGMA ** 2] * 2)


def orbit_along_scan(c, dra, ddec):
    """Photocentre orbit (dra*, ddec) [mas] projected on the Hipparcos scan."""
    return dra * c['cpsi'] + ddec * c['spsi']


def gaia_crossmatch(source_id):
    """(HIP id or None, radial velocity [km/s] or None, its error) from the Gaia
    DR3 archive: gaiadr3.hipparcos2_best_neighbour and gaia_source. The
    cross-match misses some high-proper-motion stars (HD 114762 = HIP 64426);
    set EXOPLANETLAB_HIP_ID then."""
    from astroquery.gaia import Gaia
    q = (f"SELECT g.radial_velocity, g.radial_velocity_error, h.original_ext_source_id "
         f"FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN gaiadr3.hipparcos2_best_neighbour AS h "
         f"ON h.source_id = g.source_id WHERE g.source_id = {int(source_id)}")
    r = Gaia.launch_job(q).get_results()
    if len(r) == 0:
        return None, None, None
    val = lambda k: (None if np.ma.is_masked(r[k][0]) else float(r[k][0]))
    hip = val('original_ext_source_id')
    return (None if hip is None else int(hip)), val('radial_velocity'), val('radial_velocity_error')
