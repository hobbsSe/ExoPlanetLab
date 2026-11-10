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
Exoplanet astrometric orbit parameter estimation — single and multi-planet joint inference.

Strategy
--------
1. Peel phase  : for N > 1, fit each planet individually in a short emcee run
                 on the progressively cleaned residual, giving good starting
                 positions for the joint run.  Skipped for N = 1.
2. Joint phase : run the full N×7-parameter inference.  A period-ordering
                 prior (P_0 < P_1 < …) breaks the label-switching degeneracy.

Model
-----
  Along-scan signal = sum of Keplerian orbital_model_AL for each planet.
  7 Thiele-Innes parameters per planet: [P, e, T0, A, B, F, G].
  Data: Gaia epoch along-scan astrometric residuals (mas).
  Likelihood: Gaussian with per-epoch sigma_total = sqrt(sigma_inst² + s_ast²), or Student-t.

Configuration
-------------
  SOLVER     — 'emcee', 'dynesty', 'nessai', 'rjmcmc', 'rjflow'
  SOURCE_MODE— 'simulated', 'gaia_dr4', or 'gaia_dr4_prerelease' (default)
  FIT_FLAGS  — which of the 7 parameters to fit (same for every planet)
  true_params_all — one row per planet; N_PLANETS = len(true_params_all).
                    For SOURCE_MODE='simulated' these are the injected ground truth.
                    For SOURCE_MODE='gaia_dr4_prerelease' these are an
                    optional literature/NSS truth (gaia_dr4_epoch.KNOWN_TRUTH)
                    if one has been added for SOURCE_ID, else an inert
                    placeholder -- see HAVE_TRUTH below and
                    gaia_dr4_epoch.py's module docstring. New, unpublished
                    sources are fit blind by default; this is the expected
                    case, not a fallback.
                    Rows must be P-ordered (ascending period).

Output
------
N > 0  →  Results/{N}plt_{SOLVER}_{YYYYMMDD_HHMMSS}/   (one directory per run)
    along_scan.png     — observed and true along-scan signal
    fit_residuals.png  — joint best-fit residual
    run.log            — copy of stdout
    planet_k/  (N > 1) — corner, orbit_sky, residuals, summary.txt
"""

import datetime
import importlib
import os
import sys
import numpy as np
from scipy.stats import norm as _norm_dist
if not hasattr(np, "in1d"):
    # eryn 1.2.6 (latest on PyPI) still calls np.in1d, removed in NumPy 2.0.
    np.in1d = np.isin
from types import SimpleNamespace

# ── Stage timing ──────────────────────────────────────────────────────────────
# _stage(name) marks the start of a pipeline stage: it prints the wall-clock
# time and the time since start-up, and the table printed at the end of the run
# (and appended to every summary.txt) gives each stage's duration and share.
_T0_WALL = datetime.datetime.now()
_STAGES  = []   # (name, start time)


def _fmt_td(seconds):
    s = max(0.0, float(seconds))
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}"


def _stage(name, at=None):
    now = at or datetime.datetime.now()
    _STAGES.append((name, now))
    print(f"\n[{now:%H:%M:%S} | +{_fmt_td((now - _T0_WALL).total_seconds())}] "
          f"=== Stage: {name}", flush=True)


def _stage_table(end=None):
    end = end or datetime.datetime.now()
    total = max((end - _T0_WALL).total_seconds(), 1e-9)
    lines = ['--- Stage timing (wall clock) ---',
             f"{'Stage':56s} {'start':>8s} {'duration':>10s} {'share':>6s}"]
    for i, (name, t) in enumerate(_STAGES):
        t_next = _STAGES[i + 1][1] if i + 1 < len(_STAGES) else end
        d = (t_next - t).total_seconds()
        lines.append(f"{name[:56]:56s} {t:%H:%M:%S} {_fmt_td(d):>10s} {100 * d / total:5.1f}%")
    lines.append(f"{'Total':56s} {'':8s} {_fmt_td(total):>10s}")
    return lines

import orbit
import priors
import mass_estimate
import plot_orbit
import solver_emcee   # always used for the peel phase
import solver_ptemcee  # parallel-tempered joint fit, used when RV_ENABLED (see below)
from generate_sources import manual_targets
import gaia_dr4_epoch


class _Tee:
    """Write to both the original stdout and a log file simultaneously.

    State is stashed on the `sys` module rather than tracked via
    isinstance-chain-walking of `sys.stdout`. Re-running this script inside
    one persistent session (Spyder/IPython autoreload) redefines the _Tee
    class fresh every run, so `isinstance(old_instance, _Tee)` is False
    against the *new* class object even though it's "the same" class by
    name -- a chain-walk keyed off that check silently fails to find/close
    the previous run's open log file. That leaves the old run.log file
    descriptor live and still on the write path, so every later run's
    print() calls cascade into every earlier run's run.log too. `sys` is a
    true singleton across those re-runs, so anchoring state there survives
    the class redefinition.

    The `sys` attribute names are deliberately generic (NOT
    project-prefixed): only one stdout redirection is ever live per
    process, regardless of which script set it up, so the previous-tee
    close-check must recognise a handle left by any other script using the
    same _Tee. Script-prefixed names previously let two such scripts miss
    each other's open tee when both were run in one persistent session,
    silently cascading one script's prints into the other's run.log.
    """
    def __init__(self, path):
        prev_file = getattr(sys, '_lab_tee_file', None)
        if prev_file is not None:
            try:
                prev_file.close()
            except Exception:
                pass
        self._stdout = getattr(sys, '_lab_tee_root', sys.stdout)
        self._file   = open(path, 'w')
        sys._lab_tee_file = self._file
        sys._lab_tee_root = self._stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ============================================================
# Solver selection  ← change SOLVER here
# ============================================================
#
# SOLVER            PEEL_SOLVER  Description
# ────────────────  ───────────  ──────────────────────────────────────────
# 'emcee'    emcee        Affine-invariant ensemble MCMC.  Fixed N.
# 'nessai'   nessai       Normalising-flow nested sampling.  Fixed N; ln Z.
# 'dynesty'  emcee        Single dynesty nested job over a flat +/-10 sigma box
#                         built from an emcee joint fit.  Fixed N; ln Z.
#                         See solver_dynesty.py's module docstring.
# 'rjmcmc'   emcee        Reversible-Jump MCMC (Eryn).  Variable N.
# 'rjflow'   nessai       RJMCMC + per-source RealNVP flows.  Variable N.
#
# EXOPLANETLAB_SOLVER overrides the default below (every setting has such a
# variable; user guide, "Environment-Variable Overrides").
SOLVER = os.environ.get('EXOPLANETLAB_SOLVER', 'emcee')

_DEFAULT_PEEL = {
    'emcee'   : 'emcee',
    'nessai'  : 'nessai',
    'dynesty' : 'emcee',
    'rjmcmc'  : 'emcee',
    'rjflow'  : 'nessai',
}
_VALID_PEEL = {
    'emcee'   : ('emcee',),
    'nessai'  : ('emcee', 'nessai'),
    'dynesty' : ('emcee', 'nessai'),
    'rjmcmc'  : ('emcee', 'nessai'),
    'rjflow'  : ('nessai',),
}

if SOLVER not in _DEFAULT_PEEL:
    raise ValueError(f"Unknown SOLVER={SOLVER!r}. "
                     f"Valid choices: {list(_DEFAULT_PEEL)}")

_PEEL_SOLVER_OVERRIDE = None   # set to 'emcee' or 'nessai' to override the default

PEEL_SOLVER = (_PEEL_SOLVER_OVERRIDE
               if _PEEL_SOLVER_OVERRIDE is not None
               else _DEFAULT_PEEL[SOLVER])

if PEEL_SOLVER not in _VALID_PEEL[SOLVER]:
    raise ValueError(
        f"PEEL_SOLVER={PEEL_SOLVER!r} is not valid for SOLVER={SOLVER!r}. "
        f"Valid choices: {_VALID_PEEL[SOLVER]}"
    )

# SOLVER='nessai': use emcee for the joint stage and nessai for the peels only.
# The flow proposal is slow on the correlated joint posterior of commensurate
# periods and no more accurate than the emcee joint fit. False restores nessai
# for both, which is needed for ln Z from the joint stage.
NESSAI_JOINT_USE_EMCEE = os.environ.get(
    'EXOPLANETLAB_NESSAI_JOINT_USE_EMCEE', '1') == '1'

# Number of live points for nessai peel runs (ignored when PEEL_SOLVER='emcee').
NLIVE_PEEL = 200

# Peels only locate the mode for a warm start, so a loose tolerance is enough.
STOPPING_PEEL = 1.0

# Nessai peel prior box. The peels only provide a warm start, so each samples a
# box centred on the matching fast-screen candidate instead of the full prior
# (which would repeat the screen's search at great cost):
#   P      log-uniform within [P_c/(1+f), P_c*(1+f)], f = PEEL_BOX_P_FRAC
#   T0     one period wide, [T0_c - P_c/2, T0_c + P_c/2] -- T0 is degenerate
#          mod P, so this only picks one of the equivalent copies
#   A, B, F, G  each within +-(PEEL_BOX_TI_SCALE * max(|A_c|, |B_c|, |F_c|, |G_c|)
#          + PEEL_BOX_TI_FLOOR) mas of the candidate's value (TI = Thiele-Innes)
#   e      unchanged (full range)
# all clipped to the global prior. The box is data-dependent, so the peel ln Z
# is not a full-prior evidence (it is not used downstream). Survey-probe slots,
# and slots with no screen candidate, keep the full prior.
# EXOPLANETLAB_NESSAI_PEEL_BOX=0 restores the full prior everywhere.
NESSAI_PEEL_BOX   = os.environ.get('EXOPLANETLAB_NESSAI_PEEL_BOX', '1') == '1'
PEEL_BOX_P_FRAC   = 0.20
PEEL_BOX_TI_SCALE = 3.0
PEEL_BOX_TI_FLOOR = 1.0   # mas

# "Survey probe" peels search the residual for a signal beyond the confirmed
# count and may not converge. They are capped with nessai's own max_iteration
# (which returns a valid, if unconverged, result), well above what a real peel
# needs; the slot is kept only if its SNR reaches SURVEY_PROBE_SNR_MIN.
SURVEY_PROBE_MAX_ITERATION = 6000
SURVEY_PROBE_SNR_MIN = 5.0

# When a survey-probe peel finds SNR < SURVEY_PROBE_SNR_MIN, discard the slot.
# EXOPLANETLAB_PROBE_CAP_NMAX=1 also caps this run's N_MAX to the number of
# planets actually kept (no dormant leaf).  Default (0) keeps N_MAX so the RJ
# solvers run with a dormant leaf -- a real trans-dimensional test -- and the
# discarded probe fit is kept as an extra RJ birth-proposal component.
PROBE_CAP_NMAX = os.environ.get('EXOPLANETLAB_PROBE_CAP_NMAX', '0') == '1'

# Joint nessai stage: a looser stopping tolerance than nessai's default, since
# the flow proposal stalls on thin, correlated posteriors as the tolerance
# tightens; still tighter than the peels'. JOINT_MAX_ITERATION is a hard cap
# through nessai's own max_iteration.
STOPPING_JOINT      = 0.1
JOINT_MAX_ITERATION = 20000

# Peel run length — rjmcmc/rjflow need longer chains for a reliable covariance.
if SOLVER in ('rjmcmc', 'rjflow') and PEEL_SOLVER == 'emcee':
    NSTEPS_PEEL_BURN = 2000
    NSTEPS_PEEL_PROD = 20000
else:
    NSTEPS_PEEL_BURN = 500
    NSTEPS_PEEL_PROD = 5000

# Skip the expensive joint RJ run when N is already known (only for non-RJ solvers).
RJ_SKIP_IF_N_KNOWN = SOLVER not in ('rjmcmc', 'rjflow')
# Source of the reported parameters of the reversible-jump solvers: 'chain',
# 'warmup' or 'auto' (the chain when its in-model acceptance is at least
# RJ_REPORT_MIN_ACC, else the warm-up).
RJ_REPORT = os.environ.get('EXOPLANETLAB_RJ_REPORT', 'auto').strip().lower()
RJ_REPORT_MIN_ACC = 0.05

# RJ solvers: refine the peel results with a full emcee joint fit before handing
# them over. The RJ in-model steps do not escape a peel start biased by
# cross-talk between planets, so the warm start, the proposal covariance
# (rjmcmc), the flow training data (rjflow) and the birth proposal are taken
# from the joint posterior. The RJ solver still determines N.
RJ_WARM_START_USE_EMCEE = os.environ.get(
    'EXOPLANETLAB_RJ_WARM_START_USE_EMCEE', '1') == '1'

# Cap on the samples per planet handed to the RJ solver after that refinement:
# the training time of rjflow's flows scales with the sample count.
_RJ_WARM_START_MAX_SAMPLES = 2000

solver = importlib.import_module(f'solver_{SOLVER}')
if PEEL_SOLVER == 'nessai':
    import solver_nessai


# ============================================================
# Source planets
# ============================================================
# SOURCE_MODE controls which planet population is used:
#
#   'simulated' — synthetic Gaia-like epoch astrometry generated from manual_targets().
#                 The true_params_all are the injected ground truth.
#
#   'gaia_dr4'  — real Gaia DR4 epoch astrometry (gaia_dr4_epoch.py) for
#                 SOURCE_ID, read from a DR4 epoch-astrometry file (DR4_FILE,
#                 required) in the format of the DR4 pre-release export.
#
#   'gaia_dr4_prerelease' — the same, from the DR4 pre-release export
#                 (DR4_PRERELEASE_FILE; default: the file shipped under
#                 data-release-4-tutorials/extracted_files/).
#
#   Both DR4 modes fit blind (no truth overlay) unless SOURCE_ID is in
#   gaia_dr4_epoch.KNOWN_TRUTH -- see HAVE_TRUTH below. (Gaia DR3 published
#   orbital solutions but no epoch astrometry, so there is no DR3 mode.)

# EXOPLANETLAB_SOURCE_MODE overrides the default below.
SOURCE_MODE = os.environ.get('EXOPLANETLAB_SOURCE_MODE', 'gaia_dr4_prerelease')
                            # 'simulated' | 'gaia_dr4' | 'gaia_dr4_prerelease'
# The two DR4 modes share everything except the input file.
DR4_EPOCH = SOURCE_MODE in ('gaia_dr4', 'gaia_dr4_prerelease')
if SOURCE_MODE not in ('simulated', 'gaia_dr4', 'gaia_dr4_prerelease'):
    sys.exit(f"\nWARNING: unknown SOURCE_MODE={SOURCE_MODE!r}. Choose 'simulated', "
             f"'gaia_dr4' or 'gaia_dr4_prerelease'.\n")
# EXOPLANETLAB_SOURCE_ID overrides the default below.

# Pre-release sources with a companion: Gaia-4 1457486023639239296,
# HD 114762 3937211745905473024, Gaia BH3 4318465066420528000.
SOURCE_ID   = int(os.environ.get('EXOPLANETLAB_SOURCE_ID', 4318465066420528000))

# Local Gaia DR4 pre-release epoch-astrometry export (SOURCE_MODE=
# 'gaia_dr4_prerelease' only). None = gaia_dr4_epoch's own default path
# (EXOPLANETLAB_DR4_PRERELEASE_FILE env var, else the file shipped under
# data-release-4-tutorials/extracted_files/).
DR4_PRERELEASE_FILE = os.environ.get('EXOPLANETLAB_DR4_PRERELEASE_FILE')
# Gaia DR4 epoch-astrometry file for SOURCE_MODE='gaia_dr4' (required there),
# in the format of the DR4 pre-release export read by gaia_dr4_epoch.py.
DR4_FILE = os.environ.get('EXOPLANETLAB_DR4_FILE')
if SOURCE_MODE == 'gaia_dr4' and not DR4_FILE:
    sys.exit("\nWARNING: SOURCE_MODE='gaia_dr4' needs the DR4 epoch-astrometry file: "
             "set EXOPLANETLAB_DR4_FILE (or dr4_file in a configuration file).\n")

# Injection-recovery (DR4 modes only, no RV): keep the
# real observing geometry of SOURCE_ID -- epoch times, scan angles, per-epoch
# errors and the single-star-fit projection -- but replace its data by
#     d = (I - H)(s(theta_inj) + n),   n_i ~ N(0, sigma_i^2 + INJECT_EXTRA_NOISE^2),
# i.e. a synthetic companion seen exactly as the pipeline sees real residuals.
# The fit is then run as for real data: seeded from the data only, the injected
# orbit is used for comparison (and written to injection_truth.json).
#   EXOPLANETLAB_INJECT        "P,e,phase,inc_deg,Omega_deg,omega_deg"; the
#                              periastron time is T0 = t_min + phase * P
#   EXOPLANETLAB_INJECT_SNR    total S = sqrt(sum_i [((I-H)s)_i / sigma_i]^2), the
#                              sqrt(Delta chi^2) of the true orbit; sets a0
#   EXOPLANETLAB_INJECT_A0     a0 in mas instead of INJECT_SNR
#   EXOPLANETLAB_INJECT_SEED   noise seed (default 0)
#   EXOPLANETLAB_INJECT_EXTRA_NOISE  white noise (mas) not in the quoted errors
#   EXOPLANETLAB_INJECT_NOISE_NU     Student-t noise instead of Gaussian:
#                                    n_i = f sigma_i t_nu (default 0 = Gaussian),
#                                    like the DR4 pre-release data (nu ~ 2-9)
#   EXOPLANETLAB_INJECT_NOISE_F      the scale f of that noise (default 1)
#   EXOPLANETLAB_INJECT_SS     "plx,pmra,pmdec,rv" (mas, mas/yr, km/s): add
#                              single-star motion instead of zero, the star at
#                              the reference position at J2017.5, propagated
#                              rigorously (astropy) to every epoch. With
#                              EXOPLANETLAB_HIPPARCOS=1 the synthetic Hipparcos
#                              abscissae then take the real-data path
#                              (hipparcos.reference_offset / design_rows): the
#                              test of that path
#   EXOPLANETLAB_INJECT_ACCEL  "accel_ra,accel_dec[,jerk_ra,jerk_dec]" (mas/yr^2,
#                              mas/yr^3): add an acceleration (and jerk) of the
#                              single-star motion, as a wider companion would
#                              (fit it with EXOPLANETLAB_ASTRO_ACCEL)
# Tools/make_injection_jobs.py writes jobs for Tools/run_batch.py; Tools/injection_summary.py
# collects the results.
INJECT = os.environ.get('EXOPLANETLAB_INJECT', '').strip()
INJECT_SNR = os.environ.get('EXOPLANETLAB_INJECT_SNR', '').strip()
INJECT_A0 = os.environ.get('EXOPLANETLAB_INJECT_A0', '').strip()
INJECT_SEED = int(os.environ.get('EXOPLANETLAB_INJECT_SEED', '0'))
INJECT_EXTRA_NOISE = float(os.environ.get('EXOPLANETLAB_INJECT_EXTRA_NOISE', '0.0'))
INJECT_NOISE_NU = float(os.environ.get('EXOPLANETLAB_INJECT_NOISE_NU', '0'))
INJECT_NOISE_F = float(os.environ.get('EXOPLANETLAB_INJECT_NOISE_F', '1.0'))
INJECT_SS = os.environ.get('EXOPLANETLAB_INJECT_SS', '').strip()
INJECT_SS = tuple(float(v) for v in INJECT_SS.split(',')) if INJECT_SS else None
if INJECT_SS is not None and len(INJECT_SS) != 4:
    sys.exit("\nWARNING: EXOPLANETLAB_INJECT_SS needs 'plx,pmra,pmdec,rv'.\n")
INJECT_ACCEL = os.environ.get('EXOPLANETLAB_INJECT_ACCEL', '').strip()
INJECT_ACCEL = tuple(float(v) for v in INJECT_ACCEL.split(',')) if INJECT_ACCEL else None
if INJECT_ACCEL is not None:
    if len(INJECT_ACCEL) not in (2, 4):
        sys.exit("\nWARNING: EXOPLANETLAB_INJECT_ACCEL needs 'accel_ra,accel_dec"
                 "[,jerk_ra,jerk_dec]'.\n")
    INJECT_ACCEL = INJECT_ACCEL + (0.0,) * (4 - len(INJECT_ACCEL))
# EXOPLANETLAB_INJECT_OUTER "P,e,phase,inc_deg,Omega_deg,omega_deg,a0_mas": a
# second, wider companion added to the injected data but not to the reference
# ('true') orbit -- the outer body whose curvature EXOPLANETLAB_ASTRO_ACCEL
# stands for (or fit it with EXOPLANETLAB_N_FIT=2). Gaia rows only.
INJECT_OUTER = os.environ.get('EXOPLANETLAB_INJECT_OUTER', '').strip()
INJECT_OUTER = tuple(float(v) for v in INJECT_OUTER.split(',')) if INJECT_OUTER else None
if INJECT_OUTER is not None and len(INJECT_OUTER) != 7:
    sys.exit("\nWARNING: EXOPLANETLAB_INJECT_OUTER needs "
             "'P,e,phase,inc_deg,Omega_deg,omega_deg,a0_mas'.\n")
# EXOPLANETLAB_INJECT_HARM "u2,v2[,u3,v3]" (mas): add a scan-angle signal
# u2 cos(2 psi) + v2 sin(2 psi) [+ the n = 3 terms] to the injected data -- a
# spurious, time-independent signal (fit it with EXOPLANETLAB_ASTRO_HARM).
INJECT_HARM = os.environ.get('EXOPLANETLAB_INJECT_HARM', '').strip()
INJECT_HARM = tuple(float(v) for v in INJECT_HARM.split(',')) if INJECT_HARM else None
if INJECT_HARM is not None:
    if len(INJECT_HARM) not in (2, 4):
        sys.exit("\nWARNING: EXOPLANETLAB_INJECT_HARM needs 'u2,v2[,u3,v3]'.\n")
    INJECT_HARM = INJECT_HARM + (0.0,) * (4 - len(INJECT_HARM))
INJECTION = None     # filled in the data block below when INJECT is set

# Parallel tempering for astrometry-only joint fits (see _use_pt below). On by
# default: weak signals and hard orbits converge with it that plain emcee misses.
PT_ASTRO = os.environ.get('EXOPLANETLAB_PT_ASTRO', '1') == '1'
# Evidence mode: the tempered joint fit also returns ln Z over the full
# prior (more temperatures, ladder frozen after burn-in, bridge-sampling estimate;
# solver_ptemcee.NTEMPS_EVIDENCE). For the evidence ladder N = 1, 2, 3
# (EXOPLANETLAB_N_FIT) and Tools/evidence_ladder.py (N = 0 and the Bayes
# factors B(N vs N-1) = Z_N / Z_{N-1}; a companion is added when ln B > 5).
EVIDENCE = os.environ.get('EXOPLANETLAB_EVIDENCE', '0') == '1'

# Reproducibility. EXOPLANETLAB_SEED fixes the random numbers of the
# fit: NumPy's global generator is seeded with it, emcee and eryn start their
# own generators from that state, and dynesty / nessai get seeds drawn from
# it. Without it a seed is drawn and recorded, so that any run can be repeated
# (run_info.json in the run folder: seed, EXOPLANETLAB_* settings, package
# versions, git commit). The data are not affected: simulated data and
# injections have their own fixed seeds.
_SEED_ENV = os.environ.get('EXOPLANETLAB_SEED', '').strip()
# EXOPLANETLAB_SEED_RANK = k (default 1): start the one-companion fit from the
# k-th best starting candidate (screen detections and distinct periodogram
# peaks, ranked by likelihood) -- for repeating a fit from other starts.
SEED_RANK = max(1, int(os.environ.get('EXOPLANETLAB_SEED_RANK', '1')))
SEED_CANDIDATES = []
RUN_SEED = (int(_SEED_ENV) if _SEED_ENV
            else int.from_bytes(os.urandom(4), 'little') % (2 ** 31 - 1))
np.random.seed(RUN_SEED)
# With SOLVER='dynesty' (or 'nessai' with EXOPLANETLAB_NESSAI_JOINT_USE_EMCEE=0)
# evidence mode makes the nested sampler's ln Z refer to the same normalised
# prior as the tempered fit (NESTED_EVIDENCE below): it still samples a flat box
# around the posterior (it cannot find a strong signal in the full prior), but
# the likelihood it integrates is L x prior(orbits) x V_box, the prior
# normalised over its whole support, including the isotropic orientation prior
# and the period ordering / separation of N >= 2. Exact as long as the box holds
# all the posterior mass. A second, independent route to ln Z where the tempered
# estimate does not converge.

# Gaia observation parameters. EXOPLANETLAB_MISSION_YR overrides the default.
MISSION_YR  = int(os.environ.get('EXOPLANETLAB_MISSION_YR', 10))  # 5 or 10
                            # Only for SOURCE_MODE='simulated'; DR4 epoch
                            # data are relative to J2017.5
                            # (gaia_dr4_epoch.DR4_EPOCH_REF_JD).
N_TRANSITS  = 70 if MISSION_YR == 5 else 140   # transits per mission

# Instrument noise floor (mas). Used as sigma_inst for simulated data;
# for real Gaia data, sigma_total is computed from the epoch file.
SIGMA_INST  = 0.10   # mas


# Radial-velocity configuration
RV_MODE       = os.environ.get('EXOPLANETLAB_RV_MODE', 'external')   # 'none' | 'simulated' | 'external'
RV_FILE       = (os.environ.get('EXOPLANETLAB_RV_FILE', '').strip()   # another file than the default
                 or 'RVdata/'+str(SOURCE_ID)+'/rv_data.csv') # CSV columns: time(d), rv(m/s), rv_err(m/s) — used when RV_MODE='external'
RV_N_OBS      = 50            # number of synthetic RV epochs (RV_MODE='simulated')
RV_SIGMA_INST = 10.0          # m/s — instrumental RV noise (RV_MODE='simulated')
GAMMA_TRUE    = 0.0           # m/s — systemic velocity (RV_MODE='simulated')
K_TRUE        = [500.0, 200.0]  # m/s per planet (RV_MODE='simulated')
RV_ENABLED    = (RV_MODE != 'none')
# A linear RV trend common to all instruments (EXOPLANETLAB_RV_TREND=1): the
# line-of-sight acceleration of a wider companion, the RV counterpart of the
# astrometric acceleration terms (EXOPLANETLAB_ASTRO_ACCEL). Linear,
# so it is integrated analytically in the joint likelihood: Gaussian prior
# N(0, sigma_p^2) on the slope with the scale sigma_p log-uniform on [EXOPLANETLAB_RV_TREND_MIN,
# _RV_TREND_MAX] m/s/yr, as for the acceleration terms; the reported slope is
# the flat-prior estimate. ln Z(0) and the acceleration evidences are not
# computed with it.
RV_TREND = RV_ENABLED and os.environ.get('EXOPLANETLAB_RV_TREND', '0') == '1'
# K tied to the astrometric orbit (EXOPLANETLAB_RV_COUPLED=1). For a
# dark companion the photocentre orbit is the star's, so
#     K = 2 pi (a0 / parallax) sin i / (P sqrt(1 - e^2)),
# and the RVs then constrain a0, i and the parallax. K stays a parameter of
# the fit: given the orbit, the parallax has a Gaussian conditional
# distribution (the robust single-star fit with the orbit removed), and
# |K| = c / parallax, so K with the density that follows from it stands for
# the parallax (orbit.coupled_k_lnterm, added to ln L where the RV term is;
# the sign of K still selects the node). One companion, Student-t noise and
# DR4 epoch data only; a companion that contributes light makes the
# photocentre orbit smaller than the star's and is not modelled.
RV_COUPLED = RV_ENABLED and os.environ.get('EXOPLANETLAB_RV_COUPLED', '0') == '1'
RV_TREND_MIN = float(os.environ.get('EXOPLANETLAB_RV_TREND_MIN', '1e-3'))
RV_TREND_MAX = float(os.environ.get('EXOPLANETLAB_RV_TREND_MAX', '1e3'))

# RV inference is not supported with the RJ solvers (K varies with N). Checked
# before the run directory is created, so that the run stops cleanly.
if RV_ENABLED and SOLVER in ('rjmcmc', 'rjflow'):
    sys.exit(f"\nWARNING: RV inference is not supported with SOLVER={SOLVER!r} "
             f"(RV_MODE={RV_MODE!r}; K varies with N).\n"
             f"         Set RV_MODE='none', or use SOLVER='emcee', 'dynesty' "
             f"or 'nessai'. Exiting.\n")

# Relative (imaging) astrometry of a resolved companion: the
# companion's position relative to its host at a few epochs, from direct
# imaging, interferometry or speckle. Same P, e, T0 and orientation as the
# photocentre orbit that Gaia sees, opposite in direction and larger by
# s = a_rel / a0 = 1 / (f_mass - f_flux) (orbit.relative_orbit_vec), one
# extra parameter per imaged companion (f_mass = M2 / (M1 + M2) the mass
# fraction, f_flux = L2 / (L1 + L2) the flux fraction). a_rel and P give the
# total mass dynamically (with the parallax), s the mass fraction when the
# companion is dark (f_flux = 0).
#   EXOPLANETLAB_RELAST_MODE  'none' (default) | 'external' | 'simulated'
#   EXOPLANETLAB_RELAST_FILE  default RelAstdata/<SOURCE_ID>/rel_astrometry.csv.
#       '#' comments; "# TIME_REF_JD = <JD>" (JD of time(d) = 0) is required
#       for DR4 data, as for the RV files; "# FORMAT = radec" (default) with
#       columns time(d), dra(mas), ddec(mas), sigma_ra, sigma_dec, rho
#       [, companion], or "# FORMAT = seppa" with time(d), sep(mas), pa(deg,
#       east of north), sigma_sep(mas), sigma_pa(deg), rho [, companion]
#       (converted to dra, ddec with the linearised covariance; fine while
#       sigma_sep << sep). dra includes cos(dec). companion = 1, 2, ... in
#       order of increasing period (default 1). rho may be 0.
#   EXOPLANETLAB_RELAST_SIM_SCALE / _SIM_SIGMA / _SIM_N / _SIM_SPAN
#       simulated mode (simulated sources and injections only): companion 1
#       at s = 50, sigma = 2 mas per coordinate, 8 epochs spread uniformly
#       over [t_min, t_min + SPAN x baseline] with SPAN = 3.
RELAST_MODE = os.environ.get('EXOPLANETLAB_RELAST_MODE', 'none').strip().lower()
RELAST_FILE = os.environ.get('EXOPLANETLAB_RELAST_FILE',
                             f'RelAstdata/{SOURCE_ID}/rel_astrometry.csv')
RELAST_ENABLED = RELAST_MODE != 'none'
RELAST_S_MIN, RELAST_S_MAX = 1.0, 1.0e6   # log-uniform prior on s = a_rel / a0
# EXOPLANETLAB_RELAST_JITTER  1/0: one extra noise term (mas), added in
#     quadrature to both coordinates of every relative position and fitted
#     with the orbit (log-uniform prior; bounds set after the data are
#     loaded). Published imaging errors are often too small, and without this
#     term the scale and the total mass come out too precise. Default: on for
#     'external' data, off for 'simulated' (whose errors are exact).
RELAST_JITTER = RELAST_ENABLED and os.environ.get(
    'EXOPLANETLAB_RELAST_JITTER', '1' if RELAST_MODE == 'external' else '0') == '1'
# EXOPLANETLAB_RELAST_FLUX_RATIO  "r[,sigma]": flux ratio L2 / L1 of the imaged
#     companion to its host in the Gaia G band. Not used in the fit (s is
#     free); it only converts s into the companion's mass: with
#     beta = r / (1 + r), the mass fraction is f = 1 / s + beta and
#     M2 = f M_tot (without it the summary gives M2 for a dark companion).
#     One imaged companion only.
RELAST_FLUX_RATIO = os.environ.get('EXOPLANETLAB_RELAST_FLUX_RATIO', '').strip()
RELAST_FLUX_RATIO = (tuple(float(v) for v in RELAST_FLUX_RATIO.split(','))
                     if RELAST_FLUX_RATIO else None)
if RELAST_FLUX_RATIO is not None:
    RELAST_FLUX_RATIO = RELAST_FLUX_RATIO + (0.0,) * (2 - len(RELAST_FLUX_RATIO))
    if len(RELAST_FLUX_RATIO) != 2 or RELAST_FLUX_RATIO[0] < 0 or RELAST_FLUX_RATIO[1] < 0:
        sys.exit("\nWARNING: EXOPLANETLAB_RELAST_FLUX_RATIO needs 'r' or 'r,sigma' (>= 0).\n")
if RELAST_MODE not in ('none', 'external', 'simulated'):
    sys.exit(f"\nWARNING: EXOPLANETLAB_RELAST_MODE={RELAST_MODE!r}; use 'none', "
             f"'external' or 'simulated'.\n")
if RELAST_ENABLED and SOLVER in ('rjmcmc', 'rjflow'):
    sys.exit(f"\nWARNING: relative astrometry is not supported with SOLVER={SOLVER!r} "
             f"(a per-companion scale needs a fixed N). Exiting.\n")

# Astrometric jitter: one extra per-source noise term s_ast (mas), added in
# quadrature to every along-scan error, sigma_i^2 -> sigma_i^2 + s_ast^2, and
# fitted jointly with the orbit (log-uniform prior, bounds set after the data
# are loaded). Real epoch astrometry is not well described by its formal
# errors alone -- the DR4 pre-release fits give chi2/dof of 1.4-3 (16 for
# HD 114762) -- so without this term the posterior widths are too small by
# up to sqrt(chi2/dof). The likelihood keeps its 2 ln(sigma) normalisation,
# which is what makes s_ast identifiable. Default: on for real data, off for
# SOURCE_MODE='simulated' (whose errors are exact by construction);
# EXOPLANETLAB_ASTRO_JITTER=1/0 overrides. Fixed-N solvers only (emcee,
# dynesty, nessai): the RJ solvers' per-source leaves have no slot for a
# global parameter, so they run without it.
ASTRO_JITTER = os.environ.get(
    'EXOPLANETLAB_ASTRO_JITTER',
    '0' if SOURCE_MODE == 'simulated' else '1') == '1'
if ASTRO_JITTER and SOLVER in ('rjmcmc', 'rjflow'):
    print(f"NOTE: the astrometric jitter term is not supported by "
          f"SOLVER={SOLVER!r}; running without it.")
    ASTRO_JITTER = False

# Astrometric noise model of the joint fit (robust.py):
#   'gauss'    Gaussian errors, sigma_i^2 + s_ast^2 (the jitter above), on the
#              residuals of the single-star fit (default for simulated data).
#   'student'  Student-t errors per CCD, t_nu(0, f^2 sigma_i^2), on the raw
#              along-scan data with the single-star terms fitted robustly for
#              every evaluation. nu and f are fitted (log-uniform priors on
#              [1, 100] and [0.1, 10]); the additive jitter is off. The DR4
#              pre-release excess noise is heavy-tailed CCD outliers with a core
#              at the quoted errors, which one white jitter models badly. The
#              screen and the peels stay Gaussian (starting points). Default
#              for DR4 epoch data.
_ASTRO_NOISE_ENV = os.environ.get('EXOPLANETLAB_ASTRO_NOISE', '').strip().lower()
ASTRO_NOISE = _ASTRO_NOISE_ENV or ('student' if DR4_EPOCH
                                   else 'gauss')
if ASTRO_NOISE not in ('gauss', 'student'):
    sys.exit(f"\nWARNING: EXOPLANETLAB_ASTRO_NOISE={ASTRO_NOISE!r}; use 'gauss' or 'student'.\n")
if ASTRO_NOISE == 'student':
    if SOLVER in ('rjmcmc', 'rjflow'):
        sys.exit(f"\nWARNING: EXOPLANETLAB_ASTRO_NOISE=student is not supported by "
                 f"SOLVER={SOLVER!r} (fixed-N solvers only).\n")
    if ASTRO_JITTER and os.environ.get('EXOPLANETLAB_ASTRO_JITTER', '') == '1':
        print("NOTE: EXOPLANETLAB_ASTRO_NOISE=student replaces the additive astrometric "
              "jitter by the Student-t scale f and tails nu; jitter off.")
    # The Student-t likelihood has no jitter term (f and nu replace it); a jitter
    # parameter would be sampled from its prior alone.
    ASTRO_JITTER = False

# Hipparcos intermediate data, jointly with the DR4 epochs (hipparcos.py
# has the model and the full warning). OPT-IN: off by default.
#   EXOPLANETLAB_HIPPARCOS=1       use the 2007 IAD of this star
#   EXOPLANETLAB_HIP_ID            HIP number (default: the DR3 cross-match,
#                                  which misses some high-pm stars)
#   EXOPLANETLAB_HIP_RV            radial velocity [km/s] for the perspective
#                                  acceleration (default: DR3 radial_velocity)
#   EXOPLANETLAB_HIP_IAD_ZIP       default HIPdata/ResRec_JavaTool_2014.zip
#   EXOPLANETLAB_HIP_ACCEL=1       add a constant acceleration of the single-star
#                                  motion (2 terms, both missions, zero at J2017.5
#                                  with zero rate there): for a long-period
#                                  companion not in the orbit model.
# With an injection (EXOPLANETLAB_INJECT) the Hipparcos data are SYNTHETIC: the
# geometry of HIP_ID, the injected orbit, noise like the injected Gaia noise.
# Needs the Student-t model (the joint single-star refit lives there).
HIPPARCOS = os.environ.get('EXOPLANETLAB_HIPPARCOS', '0') == '1'
HIP_ACCEL = HIPPARCOS and os.environ.get('EXOPLANETLAB_HIP_ACCEL', '0') == '1'
HIP_WARNING = (
    "WARNING -- Hipparcos IAD included (EXOPLANETLAB_HIPPARCOS=1). The Hipparcos\n"
    "  intermediate data have known, only partly understood systematics: the\n"
    "  error model (de-normalised, with its own Student-t nu and f here), corrupted\n"
    "  exported records (repaired heuristically), the Hipparcos-Gaia frame (offset\n"
    "  and spin priors of 0.6 mas and 0.25 mas/yr), and the band (the Hp photocentre\n"
    "  orbit is taken equal to the G-band one, true only for a dark companion).\n"
    "  The results depend on all of these: compare them with the Gaia-only fit.")
if HIPPARCOS:
    if not DR4_EPOCH or ASTRO_NOISE != 'student':
        sys.exit("\nWARNING: EXOPLANETLAB_HIPPARCOS=1 needs DR4 epoch data (SOURCE_MODE='gaia_dr4' or "
                 "'gaia_dr4_prerelease') and EXOPLANETLAB_ASTRO_NOISE=student.\n")
    print('\n' + HIP_WARNING + '\n')
    ASTRO_JITTER = False

# Acceleration terms in the orbit fit (accel.py): the single-star
# motion gets a constant acceleration (7) or acceleration + jerk (9), for the
# curvature of a wider companion that is not in the orbit model. Linear terms,
# refitted with the single-star ones for every likelihood evaluation, with the
# proper prior of accel.py (scales log-uniform on EXOPLANETLAB_ACCEL_MIN/_MAX,
# _JERK_MIN/_MAX), so ln Z compares with the fit without them and with the
# acceleration models without an orbit. The reported terms are the flat-prior
# estimates. DR4 epoch data, both noise models (Gaussian: the columns are
# projected like the orbit, accel.GaussAccel); with Hipparcos use
# EXOPLANETLAB_HIP_ACCEL instead (the same prior, both missions). With RVs,
# EXOPLANETLAB_RV_TREND adds the line-of-sight counterpart.
#   EXOPLANETLAB_ASTRO_ACCEL=0|7|9   (default 0)
# Scan-angle harmonics in the orbit fit (accel.py): the same kind of
# linear terms, an along-scan offset sum_n u_n cos(n psi) + v_n sin(n psi), for
# signals that depend on the scan angle rather than on time; an orbit that
# loses its evidence with them is suspect.
#   EXOPLANETLAB_ASTRO_HARM=0|2|23   (default 0; not with ASTRO_ACCEL)
# ASTRO_ACCEL below holds whichever model is chosen ('accel7', ..., 'harm23').
ASTRO_ACCEL = os.environ.get('EXOPLANETLAB_ASTRO_ACCEL', '0').strip()
if ASTRO_ACCEL not in ('', '0', '7', '9'):
    sys.exit(f"\nWARNING: EXOPLANETLAB_ASTRO_ACCEL={ASTRO_ACCEL!r}; use 0, 7 or 9.\n")
ASTRO_ACCEL = {'7': 'accel7', '9': 'accel9'}.get(ASTRO_ACCEL, '')
_ASTRO_HARM = os.environ.get('EXOPLANETLAB_ASTRO_HARM', '0').strip()
if _ASTRO_HARM not in ('', '0', '2', '23'):
    sys.exit(f"\nWARNING: EXOPLANETLAB_ASTRO_HARM={_ASTRO_HARM!r}; use 0, 2 or 23.\n")
if _ASTRO_HARM not in ('', '0'):
    if ASTRO_ACCEL:
        sys.exit("\nWARNING: EXOPLANETLAB_ASTRO_ACCEL and _ASTRO_HARM cannot be combined.\n")
    ASTRO_ACCEL = 'harm' + _ASTRO_HARM
if ASTRO_ACCEL and (not DR4_EPOCH or HIPPARCOS
                    or SOLVER in ('rjmcmc', 'rjflow')):
    sys.exit("\nWARNING: EXOPLANETLAB_ASTRO_ACCEL needs DR4 epoch data (SOURCE_MODE='gaia_dr4' or 'gaia_dr4_prerelease') "
             "and a fixed-N solver, without Hipparcos (EXOPLANETLAB_HIP_ACCEL there).\n")
_RAW_INJ = None     # raw (unprojected) injected data, for the Student-t model

# Extra unmodelled noise for simulated data (EXOPLANETLAB_SIM_EXTRA_NOISE, mas;
# see the simulated-data block below). It only exists for simulated data, so
# asking for it with a real source would silently do nothing -- stop instead.
SIM_EXTRA_NOISE = float(os.environ.get('EXOPLANETLAB_SIM_EXTRA_NOISE', '0.0'))   # mas; e.g. 0.1
if INJECT and not DR4_EPOCH:
    sys.exit(f"\nWARNING: EXOPLANETLAB_INJECT needs DR4 epoch data (SOURCE_MODE='gaia_dr4' or 'gaia_dr4_prerelease') "
             f"(it borrows that source's observing geometry), not {SOURCE_MODE!r}.\n")
if SIM_EXTRA_NOISE > 0.0 and SOURCE_MODE != 'simulated':
    sys.exit(f"\nWARNING: EXOPLANETLAB_SIM_EXTRA_NOISE={SIM_EXTRA_NOISE:g} only applies "
             f"to simulated data, but SOURCE_MODE={SOURCE_MODE!r}.\n"
             f"         Set EXOPLANETLAB_SOURCE_MODE=simulated (or unset "
             f"EXOPLANETLAB_SIM_EXTRA_NOISE). Exiting.\n")

# Simulated RVs are generated from K_TRUE/GAMMA_TRUE and the reference orbit;
# with a real source they do not correspond to the real star.
if RV_MODE == 'simulated' and SOURCE_MODE != 'simulated':
    print(f"\n{'!'*70}\nWARNING: RV_MODE='simulated' with SOURCE_MODE={SOURCE_MODE!r}: "
          f"the RVs are synthetic\n         (K_TRUE={K_TRUE} m/s) and do not "
          f"belong to this real source.\n         Use RV_MODE='external' for its "
          f"real RVs, or 'none'.\n{'!'*70}\n")

# External RVs belong to a real star. With simulated astrometry the default RV
# file (that of SOURCE_ID) would join a real star's RVs to a simulated orbit,
# so it is used only if EXOPLANETLAB_RV_FILE names a file explicitly. Both
# checks run before the run directory is created.
if RV_MODE == 'external' and SOURCE_MODE == 'simulated' \
        and not os.environ.get('EXOPLANETLAB_RV_FILE', '').strip():
    sys.exit("\nWARNING: SOURCE_MODE='simulated' with RV_MODE='external' (the default) "
             f"would fit the real RVs\n         of source {SOURCE_ID} to simulated "
             "astrometry. Set RV_MODE='none' or 'simulated'\n         "
             "(or name an RV file with EXOPLANETLAB_RV_FILE). Exiting.\n")
if RV_MODE == 'external' and not os.path.isfile(RV_FILE):
    sys.exit(f"\nWARNING: RV_MODE='external' but there is no RV file {RV_FILE!r}.\n"
             "         Set RV_MODE='none' to fit the astrometry alone, or name a file "
             "with\n         EXOPLANETLAB_RV_FILE. Exiting.\n")

# Setting checks that need no data: also before the run directory exists.
# Orientation prior (EXOPLANETLAB_ORIENT_PRIOR). Default 'isotropic' for emcee
# and the nessai/emcee hybrid; 'ti' for dynesty and the RJ samplers, which
# build their priors from transforms and cannot use it.
#   'ti'         uniform in A, B, F, G within +/-TI_MAX. In Campbell
#                elements this is p(a0, i, Omega, omega) ~ a0^3 sin^3 i (the
#                Jacobian of the Thiele-Innes map), which disfavours face-on
#                orbits far more than an isotropic prior (sin^3 i vs sin i).
#   'isotropic'  uniform in cos i, Omega and omega, log-uniform in a0 on
#                [ISO_A0_MIN, ...]: in the sampled A, B, F, G this is the density
#                1 / (a0^4 sin^2 i), added to the log-prior. sin i is floored at
#                sin(ISO_I_FLOOR_DEG): the density diverges at exactly face-on
#                (integrable, but a sampler can stick there); the floor changes
#                the prior only for i < 2 deg.
# The choice matters for nearly face-on orbits (HD 114762).
_ORIENT_ENV      = os.environ.get('EXOPLANETLAB_ORIENT_PRIOR', '').strip().lower()
NESTED_EVIDENCE  = EVIDENCE and (SOLVER == 'dynesty' or
                               (SOLVER == 'nessai' and not NESSAI_JOINT_USE_EMCEE))
ORIENT_PRIOR     = _ORIENT_ENV or ('isotropic' if SOLVER in ('emcee', 'nessai') or NESTED_EVIDENCE
                                   else 'ti')
ISO_A0_MIN       = 1.0e-3     # mas
ISO_I_FLOOR_DEG  = 2.0
if ORIENT_PRIOR not in ('ti', 'isotropic'):
    sys.exit(f"\nWARNING: EXOPLANETLAB_ORIENT_PRIOR={ORIENT_PRIOR!r}; use 'ti' or 'isotropic'.\n")
if ORIENT_PRIOR == 'isotropic' and SOLVER not in ('emcee', 'nessai') and not NESTED_EVIDENCE:
    sys.exit(f"\nWARNING: EXOPLANETLAB_ORIENT_PRIOR=isotropic is supported for SOLVER='emcee' "
             f"and 'nessai', and for 'dynesty' with EXOPLANETLAB_EVIDENCE=1; not {SOLVER!r}.\n")
_ISO_SIN2_FLOOR = np.sin(np.radians(ISO_I_FLOOR_DEG)) ** 2
if NESTED_EVIDENCE and RELAST_ENABLED:
    sys.exit("\nWARNING: nested-sampler evidence (EXOPLANETLAB_EVIDENCE=1 with dynesty / "
             "nessai) is not implemented with relative astrometry.\n")

# Orbit separation for N >= 2 (EXOPLANETLAB_ORBIT_SEPARATION). Without one, an
# N = 2 fit of a single companion can split its orbit into two of nearly the
# same period, which fits noise and is not a stable system.
#   'ratio' (default)  P_{k+1} / P_k >= P_RATIO_MIN (EXOPLANETLAB_P_RATIO_MIN,
#                default 1.2): excludes co-orbital splits and is well below the
#                closest known giant-planet pairs (3:2, e.g. HD 45364).
#   'noncrossing'  consecutive orbits must not cross: a_{k+1} (1 - e_{k+1}) >
#                a_k (1 + e_k), with a ~ P^(2/3) (the host mass cancels). Too
#                strict as a default: resonance-protected crossing orbits exist
#                (HD 82943), and the simulated two-planet benchmark (P = 1450 /
#                2900 d, e = 0.54 / 0.20) crosses.
#   'none'       period ordering P_1 < P_2 < ... only.
# The PT evidence normalises the truncated prior itself (its
# ladder reaches beta = 0); dynesty, which ignores log-priors, does not apply it.
ORBIT_SEPARATION = os.environ.get('EXOPLANETLAB_ORBIT_SEPARATION', 'ratio').strip().lower()
P_RATIO_MIN      = float(os.environ.get('EXOPLANETLAB_P_RATIO_MIN', '1.2'))
if ORBIT_SEPARATION not in ('ratio', 'noncrossing', 'none'):
    sys.exit(f"\nWARNING: EXOPLANETLAB_ORBIT_SEPARATION={ORBIT_SEPARATION!r}; "
             f"use 'ratio', 'noncrossing' or 'none'.\n")

# Planet-mass estimation (optional post-processing; see mass_estimate.py).
# Neither the orbit fit nor the RV semi-amplitude alone fixes an absolute
# mass -- Kepler's third law only ever constrains a *ratio*. An external
# host-star mass is required to anchor the scale; get it from spectroscopy,
# isochrone fitting, or a stellar-parameter catalogue -- it is NOT derived
# from this fit. Set M_STAR_MSUN to None to skip mass estimation entirely.
#   EXOPLANETLAB_M_STAR_MSUN, EXOPLANETLAB_PARALLAX_MAS, EXOPLANETLAB_M_STAR_SIGMA
#   and EXOPLANETLAB_PARALLAX_SIGMA set
#   these without editing this file; they take precedence over the values in
#   the header of the RV file.
def _env_float(name):
    v = os.environ.get('EXOPLANETLAB_' + name, '').strip()
    return float(v) if v else None


M_STAR_MSUN  = _env_float('M_STAR_MSUN')    # Msun -- host star mass (external input)
PARALLAX_MAS = _env_float('PARALLAX_MAS')   # mas  -- system parallax, needed to convert the fitted
                        # angular semi-major axis to AU. Auto-filled from the
                        # RV-file header (or set it); for
                        # SOURCE_MODE='simulated' set it manually. None = skip mass estimation.

# Optional: propagate the external uncertainty of M_STAR_MSUN / PARALLAX_MAS
# into the companion mass. With both sigmas set they become parameters of the
# joint RV fit with a Gaussian prior N(mean, sigma) and no likelihood term, so
# that the mass is computed per posterior sample. None (default) keeps them
# fixed.
M_STAR_SIGMA   = _env_float('M_STAR_SIGMA')     # Msun -- external 1-sigma uncertainty on M_STAR_MSUN
PARALLAX_SIGMA = _env_float('PARALLAX_SIGMA')   # mas  -- external 1-sigma uncertainty on PARALLAX_MAS

# Whether true_params_all holds a real reference orbit (True for
# 'simulated' always) or an
# inert placeholder with no physical meaning (a DR4 mode with no
# gaia_dr4_epoch.KNOWN_TRUTH entry for SOURCE_ID). Every place that prints
# or plots true_params_all as if it were a real answer must be gated on
# this flag; see the diagnostic print below, plot_along_scan's call, and
# the _truth_for override further down.
HAVE_TRUTH = True
# Single-star-fit projector (orbit.SingleStarProjector), or None. Set below for
# the DR4 modes, whose along-scan data are the residuals
# (I - H) d of a weighted single-star fit: every model compared with them must
# be projected the same way, (I - H) m(theta) -- see orbit.SingleStarProjector.
_PROJ = None

if SOURCE_MODE == 'simulated':
    true_params_all = manual_targets()
    # EXOPLANETLAB_SIM_N: simulate the first SIM_N planets of manual_targets()
    # (default: all of them, two).
    _SIM_N = int(os.environ.get('EXOPLANETLAB_SIM_N', len(true_params_all)))
    if not 1 <= _SIM_N <= len(true_params_all):
        sys.exit(f"\nWARNING: EXOPLANETLAB_SIM_N must be 1 to {len(true_params_all)} "
                 f"(the planets of manual_targets() in generate_sources.py).\n")
    N_PLANETS = _SIM_N
    true_params_all = true_params_all[:N_PLANETS]

elif DR4_EPOCH:
    _dr4_path = DR4_FILE if SOURCE_MODE == 'gaia_dr4' else DR4_PRERELEASE_FILE
    print(f"Loading Gaia DR4{' pre-release' if SOURCE_MODE == 'gaia_dr4_prerelease' else ''} "
          f"epoch astrometry for source_id = {SOURCE_ID} …")
    _dr4_epoch = gaia_dr4_epoch.load_source_epoch_data(
        SOURCE_ID, path=_dr4_path)
    _dr4_truth = gaia_dr4_epoch.get_known_truth(SOURCE_ID)
    _PROJ = orbit.SingleStarProjector(_dr4_epoch['proj_design'],
                                      _dr4_epoch['proj_gain'])
    N_PLANETS  = 1
    HAVE_TRUTH = _dr4_truth is not None
    if INJECT:
        if RV_MODE != 'none':
            sys.exit("\nWARNING: EXOPLANETLAB_INJECT needs EXOPLANETLAB_RV_MODE=none "
                     "(injections are astrometry-only).\n")
        if bool(INJECT_SNR) == bool(INJECT_A0):
            sys.exit("\nWARNING: EXOPLANETLAB_INJECT needs exactly one of "
                     "EXOPLANETLAB_INJECT_SNR and EXOPLANETLAB_INJECT_A0.\n")
        _iP, _ie, _iph, _ii, _iOm, _iom = (float(v) for v in INJECT.split(','))
        _t_inj  = _dr4_epoch['t_obs_day']
        _sig    = np.asarray(_dr4_epoch['sigma_mas'], dtype=float)
        _ang    = _dr4_epoch['scan_angle_rad']
        _T0_inj = float(_t_inj.min() + (_iph % 1.0) * _iP)
        _unit   = np.array([_iP, _ie, _T0_inj,
                            *orbit.ti_from_campbell(1.0, np.radians(_iom), np.radians(_iOm),
                                                    np.radians(_ii))])
        _S_unit = float(np.sqrt(np.sum((orbit.orbital_model_AL(
            _t_inj, _unit, _ang, projector=_PROJ) / _sig) ** 2)))
        _a0_inj = (float(INJECT_A0) if INJECT_A0 else float(INJECT_SNR) / _S_unit)
        _truth_inj = np.array([_iP, _ie, _T0_inj,
                               *orbit.ti_from_campbell(_a0_inj, np.radians(_iom),
                                                       np.radians(_iOm), np.radians(_ii))])
        _rng_inj = np.random.default_rng(INJECT_SEED)
        if INJECT_NOISE_NU > 0.0:           # heavy-tailed, as the DR4 pre-release data
            _noise = INJECT_NOISE_F * _sig * _rng_inj.standard_t(INJECT_NOISE_NU, len(_t_inj))
        else:
            _noise = INJECT_NOISE_F * _sig * _rng_inj.standard_normal(len(_t_inj))
        if INJECT_EXTRA_NOISE > 0.0:
            _noise += INJECT_EXTRA_NOISE * _rng_inj.standard_normal(len(_t_inj))
        _s_inj = orbit.orbital_model_AL(_t_inj, _truth_inj, _ang, projector=None)
        if INJECT_SS is not None:
            # single-star motion: the star at the reference position at J2017.5
            import hipparcos
            _xi, _eta = hipparcos.sky_path(
                _dr4_epoch['ra0_deg'], _dr4_epoch['dec0_deg'], gaia_dr4_epoch.DR4_EPOCH_REF_JD,
                'tcb', *INJECT_SS, gaia_dr4_epoch.DR4_EPOCH_REF_JD + _t_inj, 'tcb',
                _dr4_epoch['ra0_deg'], _dr4_epoch['dec0_deg'])
            _s_inj = (_s_inj + _xi * np.sin(_ang) + _eta * np.cos(_ang)
                      + INJECT_SS[0] * _dr4_epoch['proj_design'][:, 2])
        if INJECT_OUTER is not None:
            _oP, _oe, _oph, _oi, _oOm, _oom, _oa0 = INJECT_OUTER
            _s_inj = _s_inj + orbit.orbital_model_AL(
                _t_inj, np.array([_oP, _oe, float(_t_inj.min() + (_oph % 1.0) * _oP),
                                  *orbit.ti_from_campbell(_oa0, np.radians(_oom),
                                                          np.radians(_oOm), np.radians(_oi))]),
                _ang, projector=None)
        if INJECT_HARM is not None:
            _s_inj = _s_inj + (INJECT_HARM[0] * np.cos(2 * _ang) + INJECT_HARM[1] * np.sin(2 * _ang)
                               + INJECT_HARM[2] * np.cos(3 * _ang) + INJECT_HARM[3] * np.sin(3 * _ang))
        if INJECT_ACCEL is not None:
            _pn_inj = list(_dr4_epoch['proj_param_names'])
            _pm_inj = _dr4_epoch['proj_design'][:, [_pn_inj.index('pmra'), _pn_inj.index('pmdec')]]
            _tau_inj = (np.asarray(_t_inj, float) / 365.25)[:, None]
            _s_inj = (_s_inj + (_pm_inj * _tau_inj / 2.0) @ np.array(INJECT_ACCEL[:2])
                      + (_pm_inj * _tau_inj ** 2 / 6.0) @ np.array(INJECT_ACCEL[2:]))
        _dr4_epoch['along_scan_mas'] = _PROJ.apply(_s_inj + _noise)
        _RAW_INJ = _s_inj + _noise      # raw injected data (Student-t model)
        if INJECT_SS is not None:
            # synthetic raw data: the orbit-corrected single-star parameters
            # are recovered and can be compared with INJECT_SS
            _dr4_epoch['proj_data'] = _RAW_INJ
        else:
            # The real source's raw data are not used, so the orbit-corrected
            # single-star parameters would be meaningless: drop the raw data.
            _dr4_epoch.pop('proj_data', None)
        _S_true = _a0_inj * _S_unit
        _snr_epoch = float(np.std(orbit.orbital_model_AL(_t_inj, _truth_inj, _ang,
                                                          projector=_PROJ)) / np.median(_sig))
        INJECTION = dict(P=_iP, e=_ie, phase=_iph % 1.0, T0=_T0_inj, inc_deg=_ii,
                         Omega_deg=_iOm, omega_deg=_iom, a0_mas=_a0_inj,
                         A=float(_truth_inj[3]), B=float(_truth_inj[4]),
                         F=float(_truth_inj[5]), G=float(_truth_inj[6]),
                         snr_total=_S_true, snr_per_epoch=_snr_epoch,
                         seed=INJECT_SEED, extra_noise_mas=INJECT_EXTRA_NOISE,
                         noise_nu=INJECT_NOISE_NU, noise_f=INJECT_NOISE_F,
                         astro_noise=ASTRO_NOISE,
                         geometry_source_id=SOURCE_ID, n_obs=int(len(_t_inj)),
                         baseline_d=float(_t_inj.max() - _t_inj.min()),
                         single_star=(None if INJECT_SS is None else
                                      dict(zip(('parallax', 'pmra', 'pmdec', 'rv_kms'),
                                               INJECT_SS))),
                         accel=(None if INJECT_ACCEL is None else
                                dict(zip(('accel_ra', 'accel_dec', 'jerk_ra', 'jerk_dec'),
                                         INJECT_ACCEL))),
                         harm=(None if INJECT_HARM is None else
                               dict(zip(('harm2_cos', 'harm2_sin', 'harm3_cos', 'harm3_sin'),
                                        INJECT_HARM))),
                         outer=(None if INJECT_OUTER is None else
                                dict(zip(('P', 'e', 'phase', 'inc_deg', 'Omega_deg', 'omega_deg',
                                          'a0_mas'), INJECT_OUTER))))
        HAVE_TRUTH = True
        _dr4_truth = list(_truth_inj)
        print(f"  INJECTION: synthetic companion in the observing geometry of "
              f"{SOURCE_ID} (its own measurements are NOT used):")
        print(f"    P = {_iP:.2f} d, e = {_ie:.3f}, T0 = {_T0_inj:.2f} d, "
              f"i = {_ii:.1f} deg, Omega = {_iOm:.1f} deg, omega = {_iom:.1f} deg")
        print(f"    a0 = {_a0_inj:.4f} mas, total S = {_S_true:.1f}, per-epoch SNR = "
              f"{_snr_epoch:.2f}, noise seed {INJECT_SEED}, extra noise "
              f"{INJECT_EXTRA_NOISE:g} mas, noise "
              + (f"Student-t nu {INJECT_NOISE_NU:g}" if INJECT_NOISE_NU > 0 else "Gaussian")
              + f" x {INJECT_NOISE_F:g}; fitted with ASTRO_NOISE={ASTRO_NOISE}")
    if HAVE_TRUTH:
        true_params_all = np.array([_dr4_truth])
        print("  NOTE: the 'True' orbit is " + ("the injected one" if INJECTION else
              "a literature value (see gaia_dr4_epoch.KNOWN_TRUTH)")
              + ", used for comparison only -- it")
        print("  never initialises or constrains any fit here. The data are "
              "residuals of a single-star fit, so the reference orbit is")
        print("  compared through the same (I-H) projection as the fitted "
              "one (orbit.SingleStarProjector; manual, Sec. 'Residuals of a")
        print("  Single-Star Fit').")
    else:
        # No published orbit for this source (the usual case): an all-zero
        # placeholder row keeps the array-shaped code below working. It is never
        # shown as a real value: everything that prints or plots the truth is
        # gated on HAVE_TRUTH.
        _span_days = float(_dr4_epoch['t_obs_day'].max() - _dr4_epoch['t_obs_day'].min())
        true_params_all = np.array([[2.0 * _span_days, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

else:
    raise ValueError(f"Unknown SOURCE_MODE={SOURCE_MODE!r}. "
                     "Choose 'simulated', 'gaia_dr4', or 'gaia_dr4_prerelease'.")

assert all(true_params_all[i, 0] < true_params_all[i+1, 0]
           for i in range(N_PLANETS - 1)), "Rows must be ordered by ascending period"

# Number of companions FITTED (evidence ladder): EXOPLANETLAB_N_FIT
# overrides N_PLANETS after the true orbits are set, so the data (simulated from
# all rows of true_params_all) stay the same whatever is fitted. Fitted planets
# are matched to true ones by period for comparison; extra ones have no truth.
# N = 0 (no companion) has no joint fit: its evidence is computed directly by
# Tools/evidence_ladder.py.
N_TRUE = len(true_params_all)
_N_FIT_ENV = os.environ.get('EXOPLANETLAB_N_FIT', '').strip()
if _N_FIT_ENV:
    if SOLVER in ('rjmcmc', 'rjflow'):
        sys.exit("\nWARNING: EXOPLANETLAB_N_FIT is for fixed-N solvers; the RJ solvers "
                 "infer N themselves.\n")
    if int(_N_FIT_ENV) < 1:
        sys.exit("\nWARNING: EXOPLANETLAB_N_FIT must be >= 1 (N = 0: Tools/evidence_ladder.py).\n")
    N_PLANETS = int(_N_FIT_ENV)
    print(f"Fitting N = {N_PLANETS} companion(s) (EXOPLANETLAB_N_FIT); "
          f"{N_TRUE} reference orbit(s).")
# Template for fixed orbital parameters (only when FIT_FLAGS fixes some, i.e.
# simulated data): the true row of the same index, zeros beyond the true ones.
_TI_TEMPLATE = np.zeros((N_PLANETS, 7))
_TI_TEMPLATE[:min(N_PLANETS, N_TRUE)] = true_params_all[:min(N_PLANETS, N_TRUE)]

_t_start = datetime.datetime.now()

# Each run gets its own timestamped directory, so that no run overwrites another.
_run_stamp = _t_start.strftime('%Y%m%d_%H%M%S')
# EXOPLANETLAB_RUN_TAG, if set, is appended to the directory name (e.g.
# '-Gaia-BH3-RV'); Tools/run_batch.py sets it for every job, which also keeps the
# names of jobs started in the same second apart.
_run_tag = os.environ.get('EXOPLANETLAB_RUN_TAG', '').strip()
if _run_tag and not _run_tag.startswith('-'):
    _run_tag = '-' + _run_tag
# The solver is already in the name: drop a trailing '-<solver>' from the tag
# (job names such as 'Gaia-BH3-RV-dynesty' keep a batch's names unique).
if _run_tag.lower().endswith('-' + SOLVER.lower()):
    _run_tag = _run_tag[:-(len(SOLVER) + 1)]
outdir = os.path.join('Results', f'{N_PLANETS}plt_{SOLVER}_{_run_stamp}{_run_tag}')
os.makedirs(outdir)


def _write_run_info(finished=False):
    """run_info.json in the run folder: what is needed to repeat the run."""
    import json
    import subprocess
    from importlib import metadata
    vers = {'python': sys.version.split()[0]}
    for pkg in ('numpy', 'scipy', 'emcee', 'eryn', 'dynesty', 'nessai', 'torch', 'astropy',
                'corner', 'gaiasupdate'):
        try:
            vers[pkg] = metadata.version(pkg)
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                          capture_output=True, text=True, timeout=10).stdout.strip():
            git += ' (modified)'
    except Exception:
        git = 'unknown'
    info = dict(seed=RUN_SEED, seed_given=bool(_SEED_ENV), solver=SOLVER,
                source_mode=SOURCE_MODE, source_id=int(SOURCE_ID), n_fit=int(N_PLANETS),
                settings={k: v for k, v in sorted(os.environ.items())
                          if k.startswith('EXOPLANETLAB_')},
                versions=vers, git=git or 'unknown',
                seed_rank=SEED_RANK, seed_candidates=SEED_CANDIDATES,
                started=_t_start.strftime('%Y-%m-%d %H:%M:%S'),
                finished=(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                          if finished else None))
    with open(os.path.join(outdir, 'run_info.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    return info


# Written now, so that a run that stops early can still be repeated
# (Tools/rerun.py); written again at the end, with 'finished' set.
try:
    _write_run_info()
except Exception as _ex:
    print(f"  WARNING: run_info.json not written ({_ex!r})")
sys.stdout = _Tee(os.path.join(outdir, 'run.log'))
if INJECTION is not None:
    import json as _json
    with open(os.path.join(outdir, 'injection_truth.json'), 'w') as _fh:
        _json.dump(INJECTION, _fh, indent=1)
    print(f"INJECTION (geometry of {SOURCE_ID}; its own measurements are NOT used): "
          f"P = {INJECTION['P']:.2f} d, e = {INJECTION['e']:.3f}, T0 = {INJECTION['T0']:.2f} d, "
          f"i = {INJECTION['inc_deg']:.1f}, Omega = {INJECTION['Omega_deg']:.1f}, "
          f"omega = {INJECTION['omega_deg']:.1f} deg, a0 = {INJECTION['a0_mas']:.4f} mas, "
          f"total S = {INJECTION['snr_total']:.1f}, seed {INJECTION['seed']} "
          f"(also in injection_truth.json)")
_stage('set-up (imports, data loading, run directory)', at=_T0_WALL)

print(f"Run started:  {_t_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"SOURCE_ID    = {SOURCE_ID}")
print(f"RV data used = {RV_ENABLED}"
      + (f"  (RV_MODE={RV_MODE!r}, RV_FILE={RV_FILE!r})" if RV_MODE == 'external'
         else f"  (RV_MODE={RV_MODE!r})" if RV_ENABLED else ""))
print(f"Astrometric jitter fitted = {ASTRO_JITTER}")
print(f"Astrometric noise model    = {ASTRO_NOISE}")


# ============================================================
# Data: load from Gaia or simulate
# ============================================================

if SOURCE_MODE == 'simulated':
    obs_days = MISSION_YR * 365.25

    np.random.seed(77)
    t_obs       = np.sort(np.random.uniform(0.0, obs_days, N_TRANSITS))
    scan_angles = np.random.uniform(0.0, np.pi, N_TRANSITS)
    along_scan_err = np.full(N_TRANSITS, SIGMA_INST)

    # True combined along-scan signal
    along_scan_true = np.zeros(N_TRANSITS)
    for _params in true_params_all:
        along_scan_true += orbit.orbital_model_AL(t_obs, _params, scan_angles, projector=_PROJ)

    along_scan_obs = along_scan_true + along_scan_err * np.random.randn(N_TRANSITS)
    sigma_total    = along_scan_err.copy()   # no astrometric term for simulated data

    # Optional unmodelled extra noise (EXOPLANETLAB_SIM_EXTRA_NOISE, mas): added
    # to the data but not to the quoted errors, the situation the astrometric
    # jitter term exists for. Drawn from its own random stream, so that the
    # noise realisation above is unchanged.
    if SIM_EXTRA_NOISE > 0.0:
        along_scan_obs = along_scan_obs + SIM_EXTRA_NOISE * np.random.default_rng(
            20260923).standard_normal(N_TRANSITS)
        print(f"Injected extra (unmodelled) noise: {SIM_EXTRA_NOISE:.4f} mas per epoch "
              f"(quoted errors stay at {np.median(sigma_total):.4f} mas)")

    # Summary-file noise line: a clean instrument floor by construction here.
    SUMMARY_SIGMA       = float(np.median(along_scan_err))
    SUMMARY_SIGMA_LABEL = 'Instrument noise'

    ra_deg  = 285.0   # placeholder star position (not used for simulated sigma)
    dec_deg = -18.0

elif DR4_EPOCH:
    ra_deg  = _dr4_epoch['ra0_deg']
    dec_deg = _dr4_epoch['dec0_deg']
    print(f"\nTarget: RA = {ra_deg:.6f} deg,  Dec = {dec_deg:.6f} deg")
    print(f"Gaia G = {_dr4_epoch['g_mag']:.2f}")

    t_obs          = _dr4_epoch['t_obs_day']
    along_scan_obs = _dr4_epoch['along_scan_mas']
    scan_angles    = _dr4_epoch['scan_angle_rad']
    sigma_total    = _dr4_epoch['sigma_mas']

    print(f"Loaded {_dr4_epoch['n_used']} used epochs from the DR4{' pre-release' if SOURCE_MODE == 'gaia_dr4_prerelease' else ''} "
          f"export ({_dr4_epoch['n_outliers']} AGIS-downweighted epochs excluded)")
    print("\n--- Single-star (DR4-like) fit residual, before orbit fitting ---")
    print(f"  AL residual RMS     = {_dr4_epoch['resid_rms_mas']:.4f} mas")
    print(f"  AGIS excess noise   = {_dr4_epoch['excess_noise_mas']:.4f} mas  "
          f"(large => single-star model doesn't fit -- expected for a real binary)")
    print(f"  Median sigma (meas) = {np.median(sigma_total)*1e3:.2f} μas")
    print("----------------------------------------------------------------------")

    # Summary-file noise line: no instrument/astrometric decomposition
    # exists for real DR4 epoch data -- this is the total measured
    # per-epoch uncertainty, not an instrument-only floor.
    SUMMARY_SIGMA       = float(np.median(sigma_total))
    SUMMARY_SIGMA_LABEL = 'Median measured sigma'


N_epochs = len(t_obs)

# ============================================================
# Radial-velocity data (optional)
# ============================================================

if RV_MODE == 'simulated':
    np.random.seed(43)
    t_rv    = np.sort(np.random.uniform(0.0, t_obs.max(), RV_N_OBS))
    rv_true = np.zeros(RV_N_OBS)
    for _i, _params in enumerate(true_params_all):
        _P, _e, _T0, _A, _B, _F, _G = _params
        _omega   = orbit.omega_from_ti(_A, _B, _F, _G)
        rv_true += orbit.rv_model_single(t_rv, _P, _e, _T0, K_TRUE[_i], _omega)
    rv_true  += GAMMA_TRUE
    rv_obs    = rv_true + RV_SIGMA_INST * np.random.randn(RV_N_OBS)
    rv_sigma  = np.full(RV_N_OBS, RV_SIGMA_INST)
    # Single synthetic instrument -- every per-instrument code path below
    # (gamma/jitter parameters, plotting) still runs, just with N_RV_INST=1.
    RV_INSTRUMENTS = ['sim']
    rv_inst_idx    = np.zeros(RV_N_OBS, dtype=int)
    print(f"\nSimulated {RV_N_OBS} RV epochs (sigma = {RV_SIGMA_INST:.1f} m/s)")
    for _i in range(N_TRUE):
        print(f"  K_true[{_i+1}] = {K_TRUE[_i]:.1f} m/s  "
              f"(RV SNR = {K_TRUE[_i] / RV_SIGMA_INST:.1f})")
    print(f"  gamma_true = {GAMMA_TRUE:.1f} m/s")

elif RV_MODE == 'external':
    # CSV columns: time(d), rv(m/s), rv_err(m/s), instrument. Each instrument
    # gets its own gamma / jitter pair in the joint fit. '#' starts a comment.
    # Header lines "# KEY = VALUE" may give the host-star mass and parallax with
    # their uncertainties (e.g. RVdata/1457486023639239296/rv_data.csv); they
    # only fill values that are not set above.
    _csv_star_keys = {'M_STAR_MSUN': None, 'M_STAR_SIGMA': None,
                       'PARALLAX_MAS': None, 'PARALLAX_SIGMA': None}
    _t_rv, _rv_obs, _rv_sigma, _rv_inst = [], [], [], []
    _time_ref_jd = None
    with open(RV_FILE) as _f:
        for _line in _f:
            _raw = _line.strip()
            if _raw.startswith('#'):
                _kv = _raw.lstrip('#').split('=')
                if len(_kv) == 2 and _kv[0].strip() in _csv_star_keys:
                    _csv_star_keys[_kv[0].strip()] = float(_kv[1].strip())
                elif len(_kv) == 2 and _kv[0].strip() == 'TIME_REF_JD':
                    _time_ref_jd = float(_kv[1].strip())
                continue
            _line = _line.split('#', 1)[0].strip()
            if not _line:
                continue
            _fields = [_x.strip() for _x in _line.split(',')]
            if len(_fields) < 4:
                raise ValueError(
                    f"{RV_FILE}: expected 4 columns (time, rv, rv_err, "
                    f"instrument), got {len(_fields)}: {_line!r}")
            _t_rv.append(float(_fields[0]))
            _rv_obs.append(float(_fields[1]))
            _rv_sigma.append(float(_fields[2]))
            _rv_inst.append(_fields[3])
    t_rv      = np.array(_t_rv)
    rv_obs    = np.array(_rv_obs)
    rv_sigma  = np.array(_rv_sigma)
    RV_N_OBS  = len(t_rv)

    # Time zero-point of the RV file. The fit shares one T0 between the
    # astrometry (TCB days since J2017.5) and the RVs, so a file on another
    # zero-point would corrupt the RV phase silently. Files must declare
    # "# TIME_REF_JD = <JD>" (the JD of time(d) = 0); for the DR4 modes
    # the times are shifted onto gaia_dr4_epoch.DR4_EPOCH_REF_JD here, and a
    # file without the key is refused.
    if DR4_EPOCH:
        if _time_ref_jd is None:
            raise ValueError(
                f"{RV_FILE}: no '# TIME_REF_JD = <JD>' header line. For "
                f"the DR4 modes the RV time zero-point must "
                f"be declared (JD of time(d)=0, e.g. 2457936.875 for J2017.5) "
                f"so it can be checked against the astrometry's "
                f"({gaia_dr4_epoch.DR4_EPOCH_REF_JD}).")
        _rv_shift = _time_ref_jd - gaia_dr4_epoch.DR4_EPOCH_REF_JD
        if _rv_shift != 0.0:
            print(f"  RV times declared relative to JD {_time_ref_jd}; shifting by "
                  f"{_rv_shift:+.4f} d onto the DR4 reference epoch "
                  f"({gaia_dr4_epoch.DR4_EPOCH_REF_JD}).")
            t_rv = t_rv + _rv_shift

    RV_INSTRUMENTS = sorted(set(_rv_inst))
    _inst_map       = {_name: _idx for _idx, _name in enumerate(RV_INSTRUMENTS)}
    rv_inst_idx     = np.array([_inst_map[_name] for _name in _rv_inst], dtype=int)

    print(f"\nLoaded {RV_N_OBS} RV epochs from {RV_FILE!r}")
    print(f"  Instruments ({len(RV_INSTRUMENTS)}): " + ', '.join(
        f"{_name} [{int(np.sum(rv_inst_idx == _i))}]"
        for _i, _name in enumerate(RV_INSTRUMENTS)))

    if M_STAR_MSUN is None and _csv_star_keys['M_STAR_MSUN'] is not None:
        M_STAR_MSUN = _csv_star_keys['M_STAR_MSUN']
    if M_STAR_SIGMA is None and _csv_star_keys['M_STAR_SIGMA'] is not None:
        M_STAR_SIGMA = _csv_star_keys['M_STAR_SIGMA']
    if PARALLAX_MAS is None and _csv_star_keys['PARALLAX_MAS'] is not None:
        PARALLAX_MAS = _csv_star_keys['PARALLAX_MAS']
    if PARALLAX_SIGMA is None and _csv_star_keys['PARALLAX_SIGMA'] is not None:
        PARALLAX_SIGMA = _csv_star_keys['PARALLAX_SIGMA']
    if any(v is not None for v in _csv_star_keys.values()):
        print(f"  Host star (from {RV_FILE!r}): "
              f"M_star = {M_STAR_MSUN} +/- {M_STAR_SIGMA} Msun, "
              f"parallax = {PARALLAX_MAS} +/- {PARALLAX_SIGMA} mas")

else:
    t_rv = rv_obs = rv_sigma = rv_inst_idx = None
    RV_INSTRUMENTS = []

# ============================================================
# Relative (imaging) astrometry (optional; see RELAST_MODE)
# ============================================================
# rel_t, rel_x (dra*), rel_y (ddec) [mas], rel_comp (0-based companion index),
# and each point's inverse covariance (rel_ia, rel_ib, rel_ic: the xx, xy, yy
# entries) and Gaussian normalisation rel_lnorm, so the likelihood is
# sum [rel_lnorm - (ia dx^2 + 2 ib dx dy + ic dy^2) / 2].
rel_t = rel_x = rel_y = rel_comp = None
RELAST_COMP = []          # companions (0-based) with relative data, one scale each
RELAST_TRUE_SCALE = None
if RELAST_ENABLED:
    if RELAST_MODE == 'simulated':
        if SOURCE_MODE != 'simulated' and INJECTION is None:
            sys.exit("\nWARNING: EXOPLANETLAB_RELAST_MODE=simulated needs a known orbit: "
                     "SOURCE_MODE='simulated' or an injection (EXOPLANETLAB_INJECT).\n")
        RELAST_TRUE_SCALE = float(os.environ.get('EXOPLANETLAB_RELAST_SIM_SCALE', '50'))
        _sig_rel = float(os.environ.get('EXOPLANETLAB_RELAST_SIM_SIGMA', '2.0'))
        _n_rel = int(os.environ.get('EXOPLANETLAB_RELAST_SIM_N', '8'))
        _span_rel = float(os.environ.get('EXOPLANETLAB_RELAST_SIM_SPAN', '3.0'))
        _rng_rel = np.random.default_rng(47 + INJECT_SEED)
        _t0r, _t1r = float(t_obs.min()), float(t_obs.max())
        rel_t = np.sort(_rng_rel.uniform(_t0r, _t0r + _span_rel * (_t1r - _t0r), _n_rel))
        _xr, _yr = orbit.relative_orbit_vec(rel_t, np.asarray(true_params_all[:1], float),
                                            [RELAST_TRUE_SCALE])
        rel_x = _xr[0] + _sig_rel * _rng_rel.standard_normal(_n_rel)
        rel_y = _yr[0] + _sig_rel * _rng_rel.standard_normal(_n_rel)
        _sx = _sy = np.full(_n_rel, _sig_rel)
        _rho = np.zeros(_n_rel)
        rel_comp = np.zeros(_n_rel, dtype=int)
        print(f"\nSimulated {_n_rel} relative-astrometry epochs of companion 1 "
              f"(s = a_rel / a0 = {RELAST_TRUE_SCALE:g}, sigma = {_sig_rel:g} mas, "
              f"t = {rel_t.min():.0f}..{rel_t.max():.0f} d)")
    else:
        if not os.path.exists(RELAST_FILE):
            sys.exit(f"\nWARNING: EXOPLANETLAB_RELAST_MODE=external but {RELAST_FILE!r} "
                     f"does not exist (EXOPLANETLAB_RELAST_FILE). Exiting.\n")
        _rows, _fmt, _tref = [], 'radec', None
        for _line in open(RELAST_FILE):
            _raw = _line.strip()
            if _raw.startswith('#'):
                _kv = [_s.strip() for _s in _raw.lstrip('#').split('=')]
                if len(_kv) == 2 and _kv[0] == 'TIME_REF_JD':
                    _tref = float(_kv[1])
                elif len(_kv) == 2 and _kv[0] == 'FORMAT':
                    _fmt = _kv[1].lower()
                continue
            _raw = _raw.split('#', 1)[0].strip()
            if not _raw:
                continue
            _f = [float(_s) for _s in _raw.split(',')]
            if len(_f) not in (6, 7):
                sys.exit(f"\nWARNING: {RELAST_FILE}: expected 6 or 7 columns, got "
                         f"{len(_f)}: {_raw!r}\n")
            _rows.append(_f + [1.0] * (7 - len(_f)))
        if _fmt not in ('radec', 'seppa'):
            sys.exit(f"\nWARNING: {RELAST_FILE}: FORMAT = {_fmt!r}; use radec or seppa.\n")
        _a = np.array(_rows, dtype=float)
        rel_t, rel_comp = _a[:, 0], _a[:, 6].astype(int) - 1
        if _fmt == 'radec':
            rel_x, rel_y, _sx, _sy, _rho = _a[:, 1], _a[:, 2], _a[:, 3], _a[:, 4], _a[:, 5]
        else:
            # (sep, PA) -> (dra*, ddec) = sep (sin PA, cos PA), covariance J C J'
            _sep, _pa = _a[:, 1], np.radians(_a[:, 2])
            _ss, _sp, _r = _a[:, 3], np.radians(_a[:, 4]), _a[:, 5]
            rel_x, rel_y = _sep * np.sin(_pa), _sep * np.cos(_pa)
            _J = np.array([[np.sin(_pa), _sep * np.cos(_pa)],
                           [np.cos(_pa), -_sep * np.sin(_pa)]])          # (2, 2, n)
            _C = np.array([[_ss ** 2, _r * _ss * _sp], [_r * _ss * _sp, _sp ** 2]])
            _Cx = np.einsum('ikn,kln,jln->ijn', _J, _C, _J)
            _sx, _sy = np.sqrt(_Cx[0, 0]), np.sqrt(_Cx[1, 1])
            _rho = _Cx[0, 1] / (_sx * _sy)
        if DR4_EPOCH:
            if _tref is None:
                sys.exit(f"\nWARNING: {RELAST_FILE}: no '# TIME_REF_JD = <JD>' header line "
                         f"(JD of time(d) = 0, e.g. {gaia_dr4_epoch.DR4_EPOCH_REF_JD} for "
                         f"J2017.5); required for DR4 data, as for the RV files.\n")
            rel_t = rel_t + (_tref - gaia_dr4_epoch.DR4_EPOCH_REF_JD)
        print(f"\nLoaded {len(rel_t)} relative-astrometry epochs from {RELAST_FILE!r} "
              f"(format {_fmt}); t = {rel_t.min():.0f}..{rel_t.max():.0f} d")
    if np.any(rel_comp < 0) or np.any(rel_comp >= N_PLANETS):
        sys.exit(f"\nWARNING: relative astrometry for companion(s) "
                 f"{sorted(set((rel_comp + 1).tolist()))} but N = {N_PLANETS} fitted.\n")
    if np.any(np.abs(_rho) >= 1) or np.any(_sx <= 0) or np.any(_sy <= 0):
        sys.exit("\nWARNING: relative astrometry: errors must be > 0 and |rho| < 1.\n")
    RELAST_COMP = sorted(set(rel_comp.tolist()))
    _det = (_sx * _sy) ** 2 * (1.0 - _rho ** 2)
    rel_ia = _sy ** 2 / _det
    rel_ib = -_rho * _sx * _sy / _det
    rel_ic = _sx ** 2 / _det
    rel_lnorm = -np.log(2.0 * np.pi) - 0.5 * np.log(_det)
    rel_sx, rel_sy, rel_rho = _sx, _sy, _rho
    # jitter bounds: from a hundredth of the median error to the scatter of
    # the positions themselves
    _sig_med_rel = float(np.median(np.concatenate([_sx, _sy])))
    RELAST_JIT_MIN = 0.01 * _sig_med_rel
    RELAST_JIT_MAX = max(100.0 * _sig_med_rel,
                         float(np.sqrt(np.mean(rel_x ** 2 + rel_y ** 2))))
    if RELAST_FLUX_RATIO is not None and len(RELAST_COMP) != 1:
        sys.exit("\nWARNING: EXOPLANETLAB_RELAST_FLUX_RATIO is for one imaged companion.\n")
    print(f"  Companion(s) {[k + 1 for k in RELAST_COMP]}: median sigma "
          f"{np.median(_sx):.3g} / {np.median(_sy):.3g} mas (dra*, ddec); one scale "
          f"s = a_rel / a0 each (log-uniform on [{RELAST_S_MIN:g}, {RELAST_S_MAX:g}])")
    if RELAST_JITTER:
        print(f"  Imaging jitter fitted (log-uniform on [{RELAST_JIT_MIN:.3g}, "
              f"{RELAST_JIT_MAX:.3g}] mas; EXOPLANETLAB_RELAST_JITTER=0 turns it off)")


# Span = last minus first epoch. (t_obs.max() alone is the span only when time
# starts at 0, as in simulation; DR4 epochs are relative to J2017.5 and start
# negative, e.g. BH3 -979..+882 d = 1861 d, not 882 d.)
_obs_span_days = float(t_obs.max() - t_obs.min())
print(f"\nObservations: {N_epochs} epochs over {_obs_span_days:.1f} days "
      f"({_obs_span_days/365.25:.2f} yr)")
for i in range(N_TRUE if HAVE_TRUTH else N_PLANETS):
    if HAVE_TRUTH:
        P_i, e_i, T0_i, A_i, B_i, F_i, G_i = true_params_all[i]
        _sig_i = np.std(orbit.orbital_model_AL(t_obs, true_params_all[i], scan_angles, projector=_PROJ))
        snr_i  = _sig_i / np.median(sigma_total)
        print(f"  Planet {i+1}: P = {P_i:.1f} d  e = {e_i:.3f}  "
              f"a0 ≈ {np.sqrt(A_i**2+B_i**2+F_i**2+G_i**2):.2f} mas  "
              f"SNR ≈ {snr_i:.1f}")
    else:
        print(f"  Planet {i+1}: no reference orbit available -- blind fit")
print("----------------------------------------------------------------------")


# ============================================================
# Parameter metadata  (identical for every planet)
# ============================================================

PARAM_NAMES  = ['P',    'e',  'T0',    'A',    'B',    'F',    'G']
PARAM_UNITS  = ['days', '',   'days',  'mas',  'mas',  'mas',  'mas']
PARAM_SCALES = np.ones(7)

# Prior bounds
EPSILON = 1e-4    # eccentricity floor
P_MIN   = 1.0     # days
# Periods beyond ~2x the mission baseline are unconstrained by the data
# (aliasing) and just bloat the prior volume nested sampling has to search.
P_MAX   = 2 * MISSION_YR * 365.25     # days
if HIPPARCOS:
    # the baseline now starts with Hipparcos (first observations 1989 Nov,
    # JD ~2447860): twice the Hipparcos-to-Gaia span, ~66 yr for DR4
    P_MAX = max(P_MAX, 2.0 * (float(t_obs.max()) + gaia_dr4_epoch.DR4_EPOCH_REF_JD - 2447860.0))
E_MIN   = EPSILON
E_MAX   = 1.0 - EPSILON
T0_LO   = t_obs.min() - P_MAX
T0_HI   = t_obs.max() + P_MAX
TI_MAX  = 500.0   # maximum Thiele-Innes constant magnitude (mas)

# The orientation prior (ORIENT_PRIOR) is set and checked before the run
# directory is created, with the other setting checks; the note goes to run.log.
if not _ORIENT_ENV and ORIENT_PRIOR == 'ti':
    print(f"NOTE: SOLVER={SOLVER!r} cannot use the isotropic orientation prior; "
          f"using uniform Thiele-Innes priors (EXOPLANETLAB_ORIENT_PRIOR=ti).")



def _orbits_separated(Pa, ea, Pb, eb):
    """True where orbit b (Pb > Pa) is far enough from orbit a (see ORBIT_SEPARATION)."""
    return priors.orbits_separated(Pa, ea, Pb, eb, ORBIT_SEPARATION, P_RATIO_MIN)


def _orient_log_prior(A, B, F, G):
    """Extra log-prior density in (A, B, F, G) turning the uniform
    Thiele-Innes prior into the isotropic one (0 for ORIENT_PRIOR='ti').
    Works element-wise on arrays; -inf where a0 < ISO_A0_MIN. (priors.py)"""
    return priors.orient_log_prior(A, B, F, G, ORIENT_PRIOR, ISO_A0_MIN, _ISO_SIN2_FLOOR)

PARAM_LO = [P_MIN, E_MIN, T0_LO, -TI_MAX, -TI_MAX, -TI_MAX, -TI_MAX]
PARAM_HI = [P_MAX, E_MAX, T0_HI,  TI_MAX,  TI_MAX,  TI_MAX,  TI_MAX]

#                  P      e      T0     A      B      F      G
FIT_FLAGS = [     True,  True,  True,  True,  True,  True,  True ]

FIT_IDX    = [i for i, flag in enumerate(FIT_FLAGS) if flag]
ndim_per   = len(FIT_IDX)


# ============================================================
# Fast N-determination screening
# ============================================================
# Periodogram + bootstrap-FAP signal counting (detect.py), in minutes. Never
# changes N_PLANETS (the simulated truth is tied to it).
RUN_DETECT_SCREEN = True
# Robust screen (detect.py): the periodogram and its bootstrap work on
# the Student-t score of the residual instead of the residual itself. Default
# on with the Student-t noise model (heavy-tailed epoch data), off with the
# Gaussian one; EXOPLANETLAB_SCREEN_ROBUST=0|1 overrides.
SCREEN_ROBUST = os.environ.get('EXOPLANETLAB_SCREEN_ROBUST',
                               '1' if ASTRO_NOISE == 'student' else '0') == '1'

# For trans-dimensional solvers (rjmcmc/rjflow), soft-cap this run's
# N_MAX to (detected + 1): skip peel/search effort on model orders the
# fast screen found no evidence for, while leaving one slot of headroom
# for the Bayesian search to still find a real signal the fast (and
# approximate) screen missed. Only ever LOWERS N_MAX, never raises it
# above what config.py set. Does nothing for fixed-N solvers (nessai,
# emcee), where N_PLANETS -- not N_MAX -- controls the run.
CAP_N_MAX_FROM_DETECT_SCREEN = True

if RUN_DETECT_SCREEN:
    import detect
    _stage('signal screen (detect.py)')
    from config import N_MAX as _DETECT_N_MAX, N_POOL as _DETECT_N_POOL
    print("\n--- Fast periodogram + bootstrap-FAP signal screening "
          f"({_DETECT_N_POOL} worker process(es)) ---")
    _detect_result = detect.detect_signals(
        t_obs, scan_angles, along_scan_obs, sigma_total, p_min=P_MIN, p_max=P_MAX,
        n_max_screen=_DETECT_N_MAX, n_pool=_DETECT_N_POOL, projector=_PROJ,
        robust=SCREEN_ROBUST)
    print(f"Fast screening detected N = {_detect_result.n_detected} signal(s)")

    if CAP_N_MAX_FROM_DETECT_SCREEN and SOLVER in ('rjmcmc', 'rjflow'):
        _soft_cap = _detect_result.n_detected + 1
        if _soft_cap < solver.N_MAX:
            print(f"Capping this run's N_MAX: {solver.N_MAX} -> {_soft_cap} "
                  f"(detected {_detect_result.n_detected} + 1 slot of headroom "
                  f"for the Bayesian search; config.py's N_MAX is unchanged)")
            solver.N_MAX = _soft_cap
    print("----------------------------------------------------------------------")


# RV parameter bounds. K and gamma must bracket the scale of this source's own
# RV data, not a fixed "typical" value: the RVs of Gaia BH3 span ~114 km/s, and
# a fixed bound would pin the parameter at its wall.
_rv_span = float(rv_obs.max() - rv_obs.min()) if RV_ENABLED and len(rv_obs) else 0.0
#
# K_MIN is negative: (Omega, omega) and (Omega + pi, omega + pi) give the same
# Thiele-Innes constants but opposite radial velocities, and
# orbit.omega_from_ti returns one of the two branches. A negative K is the
# other branch; it is converted to the conventional (K > 0, omega) pair for
# display.
K_MAX     = max(5000.0, _rv_span)   # m/s — RV semi-amplitude upper bound
K_MIN     = -K_MAX     # m/s  — RV semi-amplitude lower bound (signed, see above)

# One (gamma, jitter) pair per RV instrument: spectrographs have their own
# zero-point and noise floor. The jitter is added in quadrature to rv_err and
# has a log-uniform prior, hence JITTER_MIN > 0.
N_RV_INST = len(RV_INSTRUMENTS) if RV_ENABLED else 0

# The gamma bounds are padded around the range of the loaded RVs, so that they
# hold for relative velocities (Gaia-4, HD 114762) and for absolute ones (Gaia
# BH3: -290 to -405 km/s) alike.
np.random.seed(RUN_SEED)    # again: the data block may have seeded its own draws
_stage('priors, parameter layout, along-scan plot')
if RV_ENABLED:
    _rv_pad   = max(50000.0, 2.0 * _rv_span)
    GAMMA_MIN = float(rv_obs.min()) - _rv_pad   # m/s — systemic velocity lower bound (per instrument)
    GAMMA_MAX = float(rv_obs.max()) + _rv_pad   # m/s — systemic velocity upper bound (per instrument)
else:
    GAMMA_MIN = -50000.0
    GAMMA_MAX =  50000.0
JITTER_MIN =      0.01  # m/s  — RV jitter lower bound (per instrument, > 0)
# The upper bound scales with the data (half the RV range, at least 1000 m/s):
# the quoted errors of some instruments exceed any fixed bound (Gaia BH3's RVS:
# 1100-2200 m/s).
JITTER_MAX =   max(1000.0, 0.5 * _rv_span)   # m/s  — RV jitter upper bound (per instrument)

# Joint parameter layout: [TI_1..N | K_1..N | gamma_1..M | jitter_1..M
#                           | (M_star, parallax) | (s_ast)]
# where N = N_PLANETS, M = N_RV_INST; the (M_star, parallax) pair is only
# present when FIT_MSTAR_PARALLAX is True (not the default; see below).
ndim_rv   = (N_PLANETS + 2 * N_RV_INST) if RV_ENABLED else 0

# M_star / parallax uncertainty (M_STAR_SIGMA / PARALLAX_SIGMA set): propagated
# into the companion mass for RV-enabled, non-RJ runs.
PROPAGATE_MSTAR_PARALLAX = (RV_ENABLED and SOLVER not in ('rjmcmc', 'rjflow')
                            and M_STAR_MSUN is not None and M_STAR_SIGMA is not None
                            and PARALLAX_MAS is not None and PARALLAX_SIGMA is not None)
# M_star and parallax have no likelihood term, so their posterior is their
# prior, independent of the fitted parameters. By default they are therefore
# not sampled: for each posterior draw used for the mass they are drawn from
# their priors, which gives the same mass posterior without two slowly mixing
# dimensions. EXOPLANETLAB_MSTAR_PLX_IN_CHAIN=1 samples them.
MSTAR_PLX_IN_CHAIN = os.environ.get('EXOPLANETLAB_MSTAR_PLX_IN_CHAIN', '0') == '1'
FIT_MSTAR_PARALLAX = PROPAGATE_MSTAR_PARALLAX and MSTAR_PLX_IN_CHAIN   # columns in the chain
ndim_mstar_plx = 2 if FIT_MSTAR_PARALLAX else 0


def _draw_positive_normal(mean, sigma, n, rng=np.random):
    """n draws from N(mean, sigma) truncated to > 0 (the prior that
    log_prior_multi applies when M_star/parallax are sampled)."""
    out = rng.normal(mean, sigma, n)
    bad = out <= 0.0
    while np.any(bad):
        out[bad] = rng.normal(mean, sigma, int(bad.sum()))
        bad = out <= 0.0
    return out

# Astrometric jitter (see ASTRO_JITTER above): appended as the LAST element of
# the joint vector, so no existing TI/K/gamma/jitter/M_star/parallax index
# moves. Log-uniform (Jeffreys, as for P and the RV jitters) from 1 uas to
# the larger of 1 mas and three times the data's own scatter.
ndim_astro_jit   = 1 if ASTRO_JITTER else 0
ASTRO_JITTER_MIN = 1.0e-3                                          # mas
ASTRO_JITTER_MAX = max(1.0, 3.0 * float(np.std(along_scan_obs)))   # mas

# Student-t noise parameters (see ASTRO_NOISE): nu and f, appended last, so
# again no other index moves (the jitter is off in this mode).
ndim_astro_noise = (2 + (2 if HIPPARCOS else 0)) if ASTRO_NOISE == 'student' else 0
NU_MIN, NU_MAX = 1.0, 100.0
F_MIN, F_MAX   = 0.1, 10.0

# Relative-astrometry scales s = a_rel / a0 (see RELAST_MODE): one per imaged
# companion, appended after everything else, and after them the imaging
# jitter (RELAST_JITTER), the last parameter.
ndim_relast = len(RELAST_COMP) + (1 if RELAST_JITTER else 0)

ndim_total = (ndim_per if SOLVER in ('rjmcmc', 'rjflow')
              else N_PLANETS * ndim_per + ndim_rv + ndim_mstar_plx + ndim_astro_jit
              + ndim_astro_noise + ndim_relast)
ASTRO_JIT_IDX = ndim_total - ndim_relast - 1 if ASTRO_JITTER else None
NU_IDX = ndim_total - ndim_relast - ndim_astro_noise if ASTRO_NOISE == 'student' else None
F_IDX  = NU_IDX + 1 if ASTRO_NOISE == 'student' else None
# Hipparcos' own Student-t (nu_hip, f_hip), right after Gaia's
NUH_IDX = NU_IDX + 2 if HIPPARCOS else None
FH_IDX  = NU_IDX + 3 if HIPPARCOS else None
RELAST_IDX = list(range(ndim_total - ndim_relast, ndim_total - ndim_relast + len(RELAST_COMP)))
RELAST_JIT_IDX = ndim_total - 1 if RELAST_JITTER else None
RELAST_POS = dict(zip(RELAST_COMP, RELAST_IDX))    # companion -> its scale's index

ROBUST = None
if ASTRO_NOISE == 'student':
    import robust
    if _RAW_INJ is not None:                           # injection
        ROBUST = robust.StudentTSingleStar(_RAW_INJ, _PROJ.X, sigma_total,
                                           _dr4_epoch['proj_prior_prec'])
    elif _PROJ is not None:                            # real DR4 epoch data
        ROBUST = robust.StudentTSingleStar(_dr4_epoch['proj_data'], _dr4_epoch['proj_design'],
                                           sigma_total, _dr4_epoch['proj_prior_prec'])
    else:                                              # simulated: no single-star fit
        ROBUST = robust.StudentTSingleStar(along_scan_obs, np.zeros((len(along_scan_obs), 0)),
                                           sigma_total)
ROBUST_BASE = ROBUST     # the single-star model without acceleration terms
ACC_GAUSS = None         # accel.GaussAccel: the terms in the Gaussian model
if ASTRO_ACCEL:
    import accel
    _ssn = list(_dr4_epoch['proj_param_names'])
    _acc_cols = (np.asarray(t_obs, float) / 365.25, _ssn.index('pmra'), _ssn.index('pmdec'))
    if ASTRO_NOISE == 'student':
        ROBUST = accel.augment(ROBUST_BASE, *_acc_cols, ASTRO_ACCEL, prior=True,
                               psi=scan_angles)
        _dr4_epoch['proj_param_names'] = _ssn + accel.NAMES[ASTRO_ACCEL]
    else:
        ACC_GAUSS = accel.GaussAccel(accel.projected_columns(
            _dr4_epoch['proj_design'], *_acc_cols, ASTRO_ACCEL, _PROJ, psi=scan_angles),
            ASTRO_ACCEL)
    print(f"Extra single-star terms ({ASTRO_ACCEL}, {accel.DESCRIPTION[ASTRO_ACCEL]}): "
          f"{', '.join(accel.NAMES[ASTRO_ACCEL])}; scales log-uniform on "
          + ', '.join(f"[{lo:g}, {hi:g}] {u}" for (lo, hi), u in
                      zip(accel.SCALES[ASTRO_ACCEL], accel.UNITS[ASTRO_ACCEL][::2])))

# Hipparcos IAD: the joint single-star fit -- Gaia rows as above plus
# the Hipparcos rows, sharing position / parallax / proper motion, with four
# frame terms (Hipparcos rows only) -- replaces ROBUST. See hipparcos.py.
HIP = None
if HIPPARCOS:
    import hipparcos
    _ss_names = list(_dr4_epoch['proj_param_names'])
    _pG = ROBUST.p
    # Gaia-only robust single-star fit without the orbit (nu = 3): only the
    # expansion point of the linearisation (an error of 1 mas/yr there costs
    # < 1e-5 mas), not a reference to compare Hipparcos with.
    _b_ref = ROBUST.fit_beta(ROBUST.d[None, :], ROBUST.sig2[None, :], np.array([[3.0]]))[0][0]
    _ipl, _ipa, _ipd = (_ss_names.index(k) for k in ('parallax', 'pmra', 'pmdec'))
    _hip_id_env = os.environ.get('EXOPLANETLAB_HIP_ID', '').strip()
    _hip_rv_env = os.environ.get('EXOPLANETLAB_HIP_RV', '').strip()
    _xm = (None, None, None)
    if not (_hip_id_env and (_hip_rv_env or INJECTION is not None)):
        try:
            _xm = hipparcos.gaia_crossmatch(SOURCE_ID)
        except Exception as _ex:
            print(f"  WARNING: Gaia DR3 cross-match query failed ({_ex!r})")
    _hip_id = int(_hip_id_env) if _hip_id_env else _xm[0]
    if _hip_id is None:
        sys.exit(f"\nWARNING: no Hipparcos cross-match for source {SOURCE_ID} in Gaia DR3; "
                 f"set EXOPLANETLAB_HIP_ID.\n")
    _hdr_h, _c_h, _notes_h = hipparcos.load_iad(
        _hip_id, os.environ.get('EXOPLANETLAB_HIP_IAD_ZIP', hipparcos.DEFAULT_ZIP))
    _t_h = _c_h['t_jd'] - gaia_dr4_epoch.DR4_EPOCH_REF_JD          # days, as t_obs
    if INJECTION is not None:
        # synthetic: this star's Hipparcos geometry and errors, the injected
        # orbit, zero frame terms; zero single-star terms (as the injected
        # Gaia data), or with INJECT_SS the same rigorous path as the Gaia data
        _dra_h, _ddec_h = orbit.astrometric_orbit_vec(_t_h, np.asarray(true_params_all, float))
        _rng_h = np.random.default_rng(INJECT_SEED + 777)
        _eps_h = (_rng_h.standard_t(INJECT_NOISE_NU, len(_t_h)) if INJECT_NOISE_NU > 0
                  else _rng_h.standard_normal(len(_t_h)))
        _d_h = (hipparcos.orbit_along_scan(_c_h, _dra_h.sum(0), _ddec_h.sum(0))
                + INJECT_NOISE_F * _c_h['sigma'] * _eps_h)
        _notes_h.append('SYNTHETIC Hipparcos data (injection): geometry of this HIP star only')
    if INJECTION is not None and INJECT_SS is None:
        _rv_h, _persp_h, _dx0, _dy0, _pm_h = None, 0.0, 0.0, 0.0, None
        _Xg_h, _Xf_h, _ = hipparcos.design_rows(_c_h, _pG, 0.0, 0.0, 0.0, 0.0)
    else:
        if INJECTION is not None:
            # abscissae relative to the catalogue position at J1991.25, as
            # load_iad gives them for real data
            _xi_h, _eta_h = hipparcos.sky_path(
                _dr4_epoch['ra0_deg'], _dr4_epoch['dec0_deg'], gaia_dr4_epoch.DR4_EPOCH_REF_JD,
                'tcb', *INJECT_SS, _c_h['t_jd'], 'tt', _hdr_h['radeg'], _hdr_h['dedeg'])
            _c_h['d'] = (_c_h['cpsi'] * _xi_h + _c_h['spsi'] * _eta_h
                         + _c_h['parf'] * INJECT_SS[0] + _d_h)
            _rv_h = INJECT_SS[3]
            _notes_h.append(f'SYNTHETIC single-star motion (INJECT_SS): plx {INJECT_SS[0]:g} mas, '
                            f'pm ({INJECT_SS[1]:g}, {INJECT_SS[2]:g}) mas/yr, rv {INJECT_SS[3]:g} '
                            f'km/s; real-data path (reference_offset, design_rows)')
        else:
            _rv_h = float(_hip_rv_env) if _hip_rv_env else _xm[1]
        if _rv_h is None:
            print("  WARNING: no radial velocity (EXOPLANETLAB_HIP_RV, DR3): perspective "
                  "acceleration neglected (rv = 0).")
            _rv_h = 0.0
        _dx0, _dy0, _pm_h, _persp_h = hipparcos.reference_offset(
            _hdr_h, _dr4_epoch['ra0_deg'], _dr4_epoch['dec0_deg'], _b_ref[_ipl],
            _b_ref[_ipa], _b_ref[_ipd], _rv_h)
        _Xg_h, _Xf_h, _d_h = hipparcos.design_rows(_c_h, _pG, _dx0, _dy0,
                                                   _b_ref[_ipa], _b_ref[_ipd], _pm_h)
    _nG, _nH = len(ROBUST.d), len(_d_h)
    _X_j = np.block([[ROBUST.X, np.zeros((_nG, 4))], [_Xg_h, _Xf_h]])
    _lam_j = np.concatenate([ROBUST.lam, hipparcos.frame_prior_precision()])
    _names_j = _ss_names + hipparcos.FRAME_NAMES
    if HIP_ACCEL:
        # position += a tau^2 / 2, tau in years from J2017.5 (Gaia's pm columns
        # are its position columns times tau, exactly); flat prior
        _tau_g = np.asarray(t_obs, float) / 365.25
        _tau_h = _c_h['t'] - hipparcos.T_HIP_TO_GAIA_YR
        _acc = np.vstack([ROBUST.X[:, [_ipa, _ipd]] * (_tau_g[:, None] / 2.0),
                          np.column_stack([_c_h['cpsi'], _c_h['spsi']]) * (_tau_h[:, None] ** 2 / 2.0)])
        _X_j = np.hstack([_X_j, _acc])
        _lam_j = np.concatenate([_lam_j, [0.0, 0.0]])
        _names_j = _names_j + ['accel_ra', 'accel_dec']
        _notes_h.append('constant acceleration of the single-star motion fitted, both '
                        'missions (EXOPLANETLAB_HIP_ACCEL=1)')
        # with the proper prior of accel.py, so that ln Z compares with the fit
        # without the terms; the reported values are the flat-prior estimates
        import accel
    ROBUST = robust.StudentTSingleStar(
        np.concatenate([ROBUST.d, _d_h]), _X_j,
        np.concatenate([np.asarray(sigma_total, float), _c_h['sigma']]), _lam_j,
        groups=np.r_[np.zeros(_nG, int), np.ones(_nH, int)],
        extra=accel.prior_term('accel7', _X_j.shape[1] - 2) if HIP_ACCEL else None)
    _dr4_epoch['proj_param_names'] = _names_j
    HIP = dict(hip_id=_hip_id, hdr=_hdr_h, c=_c_h, t_day=_t_h, n=_nH, notes=_notes_h,
               rv=_rv_h, persp=_persp_h, offset=(_dx0, _dy0))
    print(f"Hipparcos IAD: HIP {_hip_id}, solution type {_hdr_h['isol_n']}, {_nH} records "
          f"(J{1991.25 + _c_h['t'].min():.2f}-{1991.25 + _c_h['t'].max():.2f}), median error "
          f"{np.median(_c_h['sigma']):.2f} mas")
    for _n in _notes_h:
        print(f"  {_n}")
    if INJECTION is None or INJECT_SS is not None:
        print(f"  Gaia reference path propagated to J1991.25 with rv = {_rv_h:+.2f} km/s "
              f"(perspective acceleration {_persp_h:.2f} mas). Consistency of the missions: "
              f"see the fitted frame terms (hip_frame_*, hip_spin_*; priors "
              f"{hipparcos.FRAME_POS_SIGMA} mas, {hipparcos.FRAME_SPIN_SIGMA} mas/yr) and "
              f"nu_hip, f_hip.")

# Sample the noise terms (RV jitters, astrometric jitter) as ln(jitter) in the
# emcee / tempered joint fits (solver_emcee.LogCoords). Their priors are
# log-uniform over several decades, and the jitter of an instrument with one or
# two RVs is almost pure prior: in linear units that is an extremely skewed
# distribution, which the stretch move explores slowly. In ln(jitter) the prior
# is flat and the posterior is the same. Everything outside the sampler (p0,
# likelihood, results) stays in physical units. dynesty already samples them
# uniformly in ln(jitter) through its log-uniform prior transforms.
# EXOPLANETLAB_LOG_JITTER=0 samples them in linear units.
LOG_JITTER = os.environ.get('EXOPLANETLAB_LOG_JITTER', '1') == '1'
JITTER_IDX = []
if SOLVER not in ('rjmcmc', 'rjflow'):
    if RV_ENABLED:
        _jit0 = N_PLANETS * ndim_per + N_PLANETS + N_RV_INST
        JITTER_IDX += list(range(_jit0, _jit0 + N_RV_INST))
    if ASTRO_JITTER:
        JITTER_IDX.append(ASTRO_JIT_IDX)
    if ASTRO_NOISE == 'student':
        JITTER_IDX += [NU_IDX, F_IDX]      # nu and f are sampled in log too
        if HIPPARCOS:
            JITTER_IDX += [NUH_IDX, FH_IDX]
    JITTER_IDX += RELAST_IDX               # and the relative-astrometry scales
    if RELAST_JITTER:
        JITTER_IDX.append(RELAST_JIT_IDX)  # and the imaging jitter

# RV instruments with fewer than RV_JITTER_MIN_N points get no free jitter: it
# is held at JITTER_MIN. With one point the instrument's gamma absorbs it, so
# the jitter's posterior is its prior and (gamma, jitter) form a funnel that
# nested samplers cannot explore. The parameter stays in the vector (layout,
# summaries and plots unchanged); only the joint-fit samplers see the reduced
# vector (_FREE_IDX). EXOPLANETLAB_RV_JITTER_MIN_N=1 fits every jitter.
RV_JITTER_MIN_N = int(os.environ.get('EXOPLANETLAB_RV_JITTER_MIN_N', '3'))
FIXED_JITTER_INST = []
if RV_ENABLED and SOLVER not in ('rjmcmc', 'rjflow'):
    _n_per_inst = np.bincount(np.asarray(rv_inst_idx, dtype=int), minlength=N_RV_INST)
    FIXED_JITTER_INST = [k for k in range(N_RV_INST) if _n_per_inst[k] < RV_JITTER_MIN_N]
FIXED_IDX = [N_PLANETS * ndim_per + N_PLANETS + N_RV_INST + k for k in FIXED_JITTER_INST]
FIXED_VAL = JITTER_MIN
_FREE_IDX = [k for k in range(ndim_total) if k not in FIXED_IDX]
if FIXED_JITTER_INST:
    print(f"RV jitter held at {FIXED_VAL:g} m/s (fewer than {RV_JITTER_MIN_N} RVs): "
          + ", ".join(f"{RV_INSTRUMENTS[k]} [{int(_n_per_inst[k])}]" for k in FIXED_JITTER_INST))


def _expand_free(x):
    """Reduced (free-parameter) vector(s) -> full vector(s), fixed entries set."""
    x = np.asarray(x, dtype=float)
    full = np.empty(x.shape[:-1] + (ndim_total,))
    full[..., _FREE_IDX] = x
    full[..., FIXED_IDX] = FIXED_VAL
    return full


# Reduced-vector wrappers for the joint-fit samplers. Module-level functions (not
# lambdas) so that pool workers can pickle them; they read the globals above.
_JOINT_LL = None    # the joint likelihood when it is not log_likelihood_multi


def _ll_free(p, *args):
    return (_JOINT_LL or log_likelihood_multi)(_expand_free(p), *args)


def _lp_free(p):
    return _LP_JOINT_FULL(_expand_free(p))


def _pt_free(u):
    uu = np.zeros(ndim_total)
    uu[_FREE_IDX] = np.asarray(u, dtype=float)
    return np.asarray(_PT_JOINT_FULL(uu), dtype=float)[_FREE_IDX]


def _lpost_vec_free(X, *args):
    return log_posterior_multi_vec(_expand_free(X), *args)


def _ll_vec_free(X, *args):
    return log_likelihood_multi_vec(_expand_free(X), *args)


def _lp_vec_free(X):
    return log_prior_multi_vec(_expand_free(X))


def _expand_result(r):
    """Joint-fit result on the reduced vector -> the same on the full vector
    (fixed parameters at FIXED_VAL, their tau 0)."""
    nred = len(_FREE_IDX)
    upd = {}
    for key in ('samples', 'chain', 'chain_burn', 'best_params', 'weighted_samples'):
        v = getattr(r, key, None)
        if v is not None and np.ndim(v) >= 1 and np.shape(v)[-1] == nred:
            upd[key] = _expand_free(v)
    for key in ('tau', 'tau_err'):
        v = getattr(r, key, None)
        if v is not None and np.ndim(v) == 1 and len(v) == nred:
            full = np.zeros(ndim_total)
            full[_FREE_IDX] = np.asarray(v, dtype=float)
            upd[key] = full
    return SimpleNamespace(**{**vars(r), **upd})

# Prior transforms for the unit hypercube (used by nessai)
_TRANSFORMS_SINGLE = [
    lambda u: P_MIN * (P_MAX / P_MIN) ** u,           # 0: P   — log-uniform
    lambda u: E_MIN + u * (E_MAX - E_MIN),             # 1: e   — uniform
    lambda u: T0_LO + u * (T0_HI - T0_LO),            # 2: T0  — uniform
    lambda u: -TI_MAX + u * 2.0 * TI_MAX,              # 3: A   — uniform
    lambda u: -TI_MAX + u * 2.0 * TI_MAX,              # 4: B   — uniform
    lambda u: -TI_MAX + u * 2.0 * TI_MAX,              # 5: F   — uniform
    lambda u: -TI_MAX + u * 2.0 * TI_MAX,              # 6: G   — uniform
]

_PRIOR_TRANSFORMS_MULTI = []
for _pi in range(N_PLANETS):
    _PRIOR_TRANSFORMS_MULTI.extend(_TRANSFORMS_SINGLE)

_RV_TRANSFORMS = []
if RV_ENABLED:
    for _ in range(N_PLANETS):
        _RV_TRANSFORMS.append(lambda u, lo=K_MIN, hi=K_MAX: lo + u * (hi - lo))
    for _ in range(N_RV_INST):
        _RV_TRANSFORMS.append(lambda u, lo=GAMMA_MIN, hi=GAMMA_MAX: lo + u * (hi - lo))
    for _ in range(N_RV_INST):
        # Log-uniform, same construction as the Jeffreys prior on P above.
        _RV_TRANSFORMS.append(lambda u, lo=JITTER_MIN, hi=JITTER_MAX: lo * (hi / lo) ** u)

_MSTAR_PLX_TRANSFORMS = []
if FIT_MSTAR_PARALLAX:
    # Gaussian prior via inverse-CDF -- standard construction for a nested
    # sampler's unit-hypercube prior_transform (emcee doesn't use this path;
    # it gets the same Gaussian as an explicit density term in log_prior_multi).
    _MSTAR_PLX_TRANSFORMS.append(
        lambda u, m=M_STAR_MSUN, s=M_STAR_SIGMA: m + s * _norm_dist.ppf(u))
    _MSTAR_PLX_TRANSFORMS.append(
        lambda u, m=PARALLAX_MAS, s=PARALLAX_SIGMA: m + s * _norm_dist.ppf(u))

_ASTRO_JIT_TRANSFORMS = []
if ASTRO_JITTER:
    _ASTRO_JIT_TRANSFORMS.append(
        lambda u, lo=ASTRO_JITTER_MIN, hi=ASTRO_JITTER_MAX: lo * (hi / lo) ** u)

_ASTRO_NOISE_TRANSFORMS = []
if ASTRO_NOISE == 'student':
    _ASTRO_NOISE_TRANSFORMS = [lambda u: NU_MIN * (NU_MAX / NU_MIN) ** u,
                               lambda u: F_MIN * (F_MAX / F_MIN) ** u] * (2 if HIPPARCOS else 1)

_RELAST_TRANSFORMS = [lambda u: RELAST_S_MIN * (RELAST_S_MAX / RELAST_S_MIN) ** u
                      for _ in RELAST_IDX]
if RELAST_JITTER:
    _RELAST_TRANSFORMS.append(
        lambda u, lo=RELAST_JIT_MIN, hi=RELAST_JIT_MAX: lo * (hi / lo) ** u)

_ALL_TRANSFORMS = (_PRIOR_TRANSFORMS_MULTI + _RV_TRANSFORMS + _MSTAR_PLX_TRANSFORMS
                   + _ASTRO_JIT_TRANSFORMS + _ASTRO_NOISE_TRANSFORMS + _RELAST_TRANSFORMS)

# Plot the observed along-scan data. For 'gaia_dr4_prerelease' the reference
# signal is not drawn on this panel (drop the SOURCE_MODE condition below to
# draw it); orbit_sky.png shows the reference orbit.
_show_true_along_scan = HAVE_TRUTH and not DR4_EPOCH
if SOURCE_MODE == 'simulated':
    _along_scan_true_plot = along_scan_true
elif _show_true_along_scan:
    _along_scan_true_plot = orbit.orbital_model_AL(t_obs, true_params_all[0], scan_angles, projector=_PROJ)
else:
    _along_scan_true_plot = None   # blind fit, or gaia_dr4_prerelease literature truth

plot_orbit.plot_along_scan(
    t_obs, along_scan_obs, _along_scan_true_plot,
    sigma_total, true_params_all if _show_true_along_scan else None, outdir=outdir,
)


# ============================================================
# Helpers
# ============================================================

def unpack_i(active_params, planet_idx):
    """Return the full 7-vector for planet planet_idx; fixed params from true_params_all."""
    if SOURCE_MODE != 'simulated' and len(FIT_IDX) < 7:
        raise NotImplementedError(
            "Fixing orbital parameters (FIT_FLAGS with a False entry) would "
            "take the fixed values from the reference/literature orbit, "
            "which real-data fits must not depend on. Fit all 7 parameters "
            "or use SOURCE_MODE='simulated'.")
    p = _TI_TEMPLATE[planet_idx].copy()
    for j, gi in enumerate(FIT_IDX):
        p[gi] = active_params[j]
    return p


def model_all(all_active_params):
    """Sum along-scan signal from all N_PLANETS."""
    signal = np.zeros(N_epochs)
    for i in range(N_PLANETS):
        signal += orbit.orbital_model_AL(
            t_obs, unpack_i(all_active_params[i*ndim_per:(i+1)*ndim_per], i), scan_angles,
            projector=_PROJ)
    return signal


def _rv_model_all(all_active_params, t, inst_idx=None):
    """Sum RV signal from all N_PLANETS.

    Params layout is [TI_1..N, K_1..N, gamma_1..M, jitter_1..M] (M =
    N_RV_INST). jitter doesn't enter the mean model (it's a likelihood-side
    noise term -- see log_likelihood_multi), only gamma does.

    inst_idx : (len(t),) int array mapping each point in t to its RV
        instrument's index in RV_INSTRUMENTS (real epochs -- rv_inst_idx),
        or None for a synthetic time grid with no instrument association
        (e.g. a smooth plotting grid), in which case gamma is omitted
        (pure Keplerian signal, zero offset) -- see plot_orbit.plot_rv,
        which instead shifts each instrument's *data* by its own fitted
        gamma so both land on the same zero-point.
    """
    k_start     = N_PLANETS * ndim_per
    gamma_start = k_start + N_PLANETS
    if inst_idx is None:
        rv = np.zeros(len(t))
    else:
        gammas = all_active_params[gamma_start:gamma_start + N_RV_INST]
        rv     = np.asarray(gammas, dtype=float)[np.asarray(inst_idx)]
    for i in range(N_PLANETS):
        ti = unpack_i(all_active_params[i*ndim_per:(i+1)*ndim_per], i)
        P, e, T0, A, B, F, G = ti
        omega  = orbit.omega_from_ti(A, B, F, G)
        K_i    = all_active_params[k_start + i]
        rv    += orbit.rv_model_single(t, P, e, T0, float(K_i), omega)
    return rv


def extract_planet_result(result_joint, planet_idx):
    """Slice a single-planet result namespace from the joint result."""
    cols    = slice(planet_idx * ndim_per, (planet_idx + 1) * ndim_per)
    tau     = result_joint.tau[cols] if result_joint.tau is not None else None
    chain_j = result_joint.chain[:, :, cols] if result_joint.chain is not None else None
    burn_j  = result_joint.chain_burn[:, :, cols] if result_joint.chain_burn is not None else None
    _ws     = getattr(result_joint, 'weighted_samples', None)
    return SimpleNamespace(
        samples             = result_joint.samples[:, cols],
        weighted_samples    = _ws[:, cols] if _ws is not None else None,
        weights             = getattr(result_joint, 'weights', None),
        log_probs           = result_joint.log_probs,
        # log_probs is the *joint* log-posterior; keep its dimensionality so
        # corner_samples sizes the chi-square credible-region cut correctly
        # even on this single-planet slice.
        logp_df             = (result_joint.chain.shape[2]
                               if result_joint.chain is not None
                               else result_joint.samples.shape[1]),
        best_params         = result_joint.best_params[cols],
        chain               = chain_j,
        chain_burn          = burn_j,
        logprob_burn        = result_joint.logprob_burn,
        acceptance_fraction = result_joint.acceptance_fraction,
        tau                 = tau,
        tau_err             = result_joint.tau_err,
        nsteps_prod         = result_joint.nsteps_prod,
        logz                = getattr(result_joint, 'logz', None),
        logzerr             = getattr(result_joint, 'logzerr', None),
    )


# ── Corner-plot sample preparation ───────────────────────────────────────────
# An emcee joint chain can leave a fraction of walkers strung along a
# low-barrier degeneracy ridge (the Thiele-Innes orientation degeneracy for a
# lower-SNR planet; the P–P' 2:1-crosstalk ridge) that never fully
# equilibrate.  Unthinned, those excursions render as thin "streamer" tracks.
#
#   EXOPLANETLAB_PLOT_JOINT_DISCARD  -- drop this many production steps from the
#     head (start-up transient; does little for the ongoing excursions).
#   EXOPLANETLAB_PLOT_LOGP_NSIGMA  -- primary, run-independent streamer filter:
#     keep only samples with lnP >= lnP_peak - 0.5*chi2_ppf(Phi(NSIGMA), d),
#     i.e. the full NSIGMA-equivalent credible region.  A real secondary mode
#     (comparable lnP) stays inside it; a stuck / in-transit walker does not.
#     chi2 quantile via Wilson-Hilferty (no scipy).  Set 0 to disable.
#   EXOPLANETLAB_PLOT_LOGL_DROP_PCT  -- optional extra percentile floor applied
#     after the NSIGMA cut.  Default 0.
PLOT_JOINT_DISCARD = int(os.environ.get('EXOPLANETLAB_PLOT_JOINT_DISCARD', '2000'))
PLOT_LOGP_NSIGMA   = float(os.environ.get('EXOPLANETLAB_PLOT_LOGP_NSIGMA', '3.7'))
PLOT_LOGL_DROP_PCT = float(os.environ.get('EXOPLANETLAB_PLOT_LOGL_DROP_PCT', '0.0'))

# ── Periastron-epoch (T0) display fold ──────────────────────────────────────
# T0 is only defined modulo the period: T0, T0 +/- P, T0 +/- 2P ... describe the
# same orbit with an identical likelihood, and the (wide) T0 prior spans several
# of those aliases, so the raw fitted T0 is often reported k periods away from
# the true one.  With EXOPLANETLAB_FOLD_T0=1 (default) every reported / plotted
# T0 is folded into the single period cycle that starts at the first observation
# epoch, purely for readability -- the fit itself is untouched, and the raw
# (unfolded) fitted T0 is still printed alongside.  Set 0 to show raw T0 only.
FOLD_T0 = os.environ.get('EXOPLANETLAB_FOLD_T0', '1') == '1'


def _t0_anchor(col, period, t_ref):
    """Circular-mean T0 of a sample column, mapped into [t_ref, t_ref + period)."""
    ang  = 2.0 * np.pi * (np.ravel(np.asarray(col, dtype=float)) - t_ref) / period
    cang = np.arctan2(np.nanmean(np.sin(ang)), np.nanmean(np.cos(ang)))
    return t_ref + (cang % (2.0 * np.pi)) * period / (2.0 * np.pi)


def _fold_t0(x, period, anchor):
    """Fold T0 value(s) into the period cycle centred on ``anchor`` (+/- P/2).

    ``period`` is a scalar or, for posterior samples, each sample's own period
    (same shape as ``x``). Use the per-sample periods: a tempered chain visits
    T0 copies many cycles away, and folding a copy k cycles out with one common
    period P0 misplaces it by k (P - P0), which inflates the folded T0 spread
    (x2.1 on Gaia-4 without RV). Values with an invalid period are
    returned unchanged."""
    x = np.asarray(x, dtype=float)
    period = np.asarray(period, dtype=float)
    ok = np.isfinite(period) & (period > 0.0)
    if not np.any(ok):
        return x
    p = np.where(ok, period, 1.0)
    return np.where(ok, anchor + ((x - anchor + 0.5 * p) % p) - 0.5 * p, x)


def _chi2_half_quantile(df, nsigma):
    """0.5 * chi2_ppf(Phi(nsigma), df) via Wilson-Hilferty (no scipy).
    Accurate to <0.3 nat for df <= 14, nsigma <= 4 -- ample for a plot cut."""
    df = max(int(df), 1)
    h  = 2.0 / (9.0 * df)
    return 0.5 * df * max(1.0 - h + float(nsigma) * (h ** 0.5), 0.0) ** 3


def corner_samples(result, ndim, discard=None, logl_drop_pct=None, nsigma=None):
    """Flat posterior samples for corner plots.

    (1) Drops the first ``discard`` production steps (a start-up transient).
    (2) Keeps only samples within the ``nsigma``-equivalent credible region of
        the peak joint log-posterior:
        lnP >= lnP_peak - 0.5 * chi2_ppf(Phi(nsigma), df).  Run-independent --
        no per-run tuning; a real secondary mode is not touched.
    (3) Optionally drops a further ``logl_drop_pct`` percent by lnP.
    Each cut is skipped if it would keep < 40% of the samples.
    Falls back to ``result.samples`` when there is no step-resolved chain or no
    aligned log-posterior array (nested samplers -- no streamer problem)."""
    if discard is None:
        discard = PLOT_JOINT_DISCARD
    if logl_drop_pct is None:
        logl_drop_pct = PLOT_LOGL_DROP_PCT
    if nsigma is None:
        nsigma = PLOT_LOGP_NSIGMA

    ch   = getattr(result, 'chain', None)
    logp = getattr(result, 'log_probs', None)
    df   = int(getattr(result, 'logp_df', ndim) or ndim)

    if ch is None or logp is None:
        return result.samples

    nsteps, nw = ch.shape[0], ch.shape[1]
    logp = np.asarray(logp).reshape(nsteps, nw)
    d    = discard if 0 < discard < nsteps - 2000 else 0
    samp = ch[d:].reshape(-1, ndim)
    logp = logp[d:].reshape(-1)

    if logp.shape[0] != samp.shape[0]:
        return samp
    finite = np.isfinite(logp)
    if finite.sum() <= 100:
        return samp
    n_guard = max(100, int(0.40 * samp.shape[0]))

    if nsigma > 0.0:
        logp_peak = np.percentile(logp[finite], 99.9)
        thr  = logp_peak - _chi2_half_quantile(df, nsigma)
        keep = finite & (logp >= thr)
        if keep.sum() >= n_guard:
            samp, logp, finite = samp[keep], logp[keep], finite[keep]

    if 0.0 < logl_drop_pct < 100.0:
        thr  = np.percentile(logp[finite], logl_drop_pct)
        keep = finite & (logp >= thr)
        if keep.sum() >= n_guard:
            samp = samp[keep]

    return samp


def corner_input(result, ndim, discard=None):
    """(data, weights) for a corner plot.

    A nessai result carries the full weighted nested samples; a weighted corner
    over those is far smoother than the equal-weight, ESS-capped ``samples``.
    Everything else returns ``(corner_samples(...), None)``."""
    ws = getattr(result, 'weighted_samples', None)
    w  = getattr(result, 'weights', None)
    if ws is not None and w is not None and ws.ndim == 2 and ws.shape[1] == ndim \
            and len(w) == len(ws):
        return ws, w
    return corner_samples(result, ndim, discard=discard), None


# ============================================================
# Bayesian inference — prior and likelihood
# ============================================================

def log_prior_single(active_params, planet_idx):
    """Log-prior for one planet: Jeffreys on P, uniform on e, flat on the rest."""
    return priors.log_prior_orbit(
        unpack_i(active_params, planet_idx), P_MIN, P_MAX, E_MIN, T0_LO, T0_HI, TI_MAX,
        ORIENT_PRIOR, ISO_A0_MIN, _ISO_SIN2_FLOOR, jeffreys_p=(0 in FIT_IDX))


def _peel_box(cand, long_period=False):
    """Per-global-index (lo, hi) prior box for a nessai peel around the
    fast-screen candidate `cand` (full 7-vector) -- see NESSAI_PEEL_BOX.
    long_period: the screen accepted the candidate although its period is
    pinned at P_MAX (an orbit longer than the baseline, detect.py
    BOUNDARY_FAP_MAX), so its period is unknown: the box then spans P from
    half the baseline to P_MAX instead of P_c +- PEEL_BOX_P_FRAC."""
    P_c, T0_c = float(cand[0]), float(cand[2])
    ti_c = np.asarray(cand[3:7], dtype=float)
    w_ti = PEEL_BOX_TI_SCALE * float(np.max(np.abs(ti_c))) + PEEL_BOX_TI_FLOOR
    P_lo = (0.5 * float(t_obs.max() - t_obs.min()) if long_period
            else P_c / (1.0 + PEEL_BOX_P_FRAC))
    lo = np.array([max(P_MIN, P_lo), E_MIN,
                   max(T0_LO, T0_c - 0.5 * P_c),
                   *np.maximum(-TI_MAX, ti_c - w_ti)])
    hi = np.array([min(P_MAX, P_c * (1.0 + PEEL_BOX_P_FRAC)), E_MAX,
                   min(T0_HI, T0_c + 0.5 * P_c),
                   *np.minimum(TI_MAX, ti_c + w_ti)])
    return lo, hi


def _peel_box_prior(lo, hi):
    """(prior_transform, log_prior) over the FIT_IDX parameters for a peel
    box: log-uniform in P (same Jeffreys form as log_prior_single), uniform
    in everything else, -inf outside the box."""
    lo_a = lo[FIT_IDX]
    hi_a = hi[FIT_IDX]

    def _pt(u):
        x = lo_a + np.asarray(u) * (hi_a - lo_a)
        if 0 in FIT_IDX:
            j = FIT_IDX.index(0)
            x[j] = lo_a[j] * (hi_a[j] / lo_a[j]) ** u[j]
        return x

    def _lp(p):
        p = np.asarray(p)
        if np.any(p <= lo_a) or np.any(p >= hi_a):
            return -np.inf
        return -np.log(p[FIT_IDX.index(0)]) if 0 in FIT_IDX else 0.0

    return _pt, _lp


def _warn_peel_box_edge(best, lo, hi, label):
    """A best fit hugging the box edge means the box may have cut the mode."""
    for j, gi in enumerate(FIT_IDX):
        if gi == 2:
            continue   # T0 wraps (degenerate mod P); an edge value is harmless
        w = hi[gi] - lo[gi]
        if min(best[j] - lo[gi], hi[gi] - best[j]) < 0.02 * w:
            print(f"  WARNING: {label} best-fit {PARAM_NAMES[gi]}="
                  f"{best[j]:.4g} is at the edge of its peel prior box "
                  f"[{lo[gi]:.4g}, {hi[gi]:.4g}] -- the box may have truncated "
                  f"the mode; rerun with EXOPLANETLAB_NESSAI_PEEL_BOX=0 to check.")


def log_prior_astro(all_active_params):
    """Astrometric part of the joint prior: per-planet TI priors + period ordering.

    Reads only the first N_PLANETS*ndim_per entries, so it is valid both for the
    full joint vector and for the astrometry-only vector used by
    _emcee_joint_prerefine (no K/gamma/jitter/M_star/parallax columns)."""
    if 0 in FIT_IDX:
        p_col  = FIT_IDX.index(0)
        periods = [all_active_params[i*ndim_per + p_col] for i in range(N_PLANETS)]
        if any(Pa >= Pb for Pa, Pb in zip(periods, periods[1:])):
            return -np.inf
        if N_PLANETS > 1:
            rows = [unpack_i(all_active_params[i*ndim_per:(i+1)*ndim_per], i)
                    for i in range(N_PLANETS)]
            if not all(_orbits_separated(a[0], a[1], b[0], b[1])
                       for a, b in zip(rows, rows[1:])):
                return -np.inf
    lp = 0.0
    for i in range(N_PLANETS):
        lp_i = log_prior_single(all_active_params[i*ndim_per:(i+1)*ndim_per], i)
        if not np.isfinite(lp_i):
            return -np.inf
        lp += lp_i
    return lp


def log_prior_multi(all_active_params):
    """Joint prior: per-planet TI priors + period-ordering + optional K/gamma/jitter bounds."""
    lp = log_prior_astro(all_active_params)
    if not np.isfinite(lp):
        return -np.inf
    if RV_ENABLED:
        k_start     = N_PLANETS * ndim_per
        gamma_start = k_start + N_PLANETS
        jit_start   = gamma_start + N_RV_INST
        k_params    = all_active_params[k_start:k_start + N_PLANETS]
        gammas      = all_active_params[gamma_start:gamma_start + N_RV_INST]
        jitters     = all_active_params[jit_start:jit_start + N_RV_INST]
        if np.any(np.asarray(k_params) < K_MIN) or np.any(np.asarray(k_params) > K_MAX):
            return -np.inf
        if np.any(np.asarray(gammas) < GAMMA_MIN) or np.any(np.asarray(gammas) > GAMMA_MAX):
            return -np.inf
        jitters = np.asarray(jitters)
        if np.any(jitters < JITTER_MIN) or np.any(jitters > JITTER_MAX):
            return -np.inf
        lp -= np.sum(np.log(jitters))   # Jeffreys prior on jitter, same as P
        if FIT_MSTAR_PARALLAX:
            mstar_idx = jit_start + N_RV_INST
            m_star    = all_active_params[mstar_idx]
            plx       = all_active_params[mstar_idx + 1]
            if m_star <= 0.0 or plx <= 0.0:
                return -np.inf
            # Gaussian prior, no likelihood term -- see M_STAR_SIGMA comment.
            lp += (-0.5 * ((m_star - M_STAR_MSUN) / M_STAR_SIGMA) ** 2
                   -0.5 * ((plx - PARALLAX_MAS) / PARALLAX_SIGMA) ** 2)
    return lp + _log_prior_astro_jitter(all_active_params)


def _log_prior_relast(all_active_params):
    """Log-uniform priors on the relative-astrometry scales s = a_rel / a0 and
    on the imaging jitter."""
    lp = 0.0
    for k in RELAST_IDX:
        sc = all_active_params[k]
        if not (RELAST_S_MIN <= sc <= RELAST_S_MAX):
            return -np.inf
        lp -= np.log(sc)
    if RELAST_JITTER:
        jr = all_active_params[RELAST_JIT_IDX]
        if not (RELAST_JIT_MIN <= jr <= RELAST_JIT_MAX):
            return -np.inf
        lp -= np.log(jr)
    return lp


def _log_prior_astro_jitter(all_active_params):
    """Log-uniform priors on the parameters appended after the orbit, RV and
    M_star/parallax blocks: the astrometric jitter or, in Student-t mode, nu and
    f, plus the relative-astrometry scales."""
    lp = _log_prior_relast(all_active_params) if RELAST_ENABLED else 0.0
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_prior_noise(all_active_params)


def _log_prior_noise(all_active_params):
    """Log-uniform prior on the astrometric jitter (0 when it is not fitted),
    or, in Student-t mode, on nu and f."""
    if ASTRO_NOISE == 'student':
        lp = 0.0
        for _i, _j in ((NU_IDX, F_IDX),) + (((NUH_IDX, FH_IDX),) if HIPPARCOS else ()):
            nu, f = all_active_params[_i], all_active_params[_j]
            if not (NU_MIN <= nu <= NU_MAX and F_MIN <= f <= F_MAX):
                return -np.inf
            lp -= np.log(nu) + np.log(f)
        return lp
    if not ASTRO_JITTER:
        return 0.0
    s = all_active_params[ASTRO_JIT_IDX]
    if not (ASTRO_JITTER_MIN <= s <= ASTRO_JITTER_MAX):
        return -np.inf
    return -np.log(s)


def _astro_models_raw(X):
    """Unprojected along-scan signal of all planets for a (k, ndim) array."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = np.zeros((X.shape[0], N_epochs))
    for i in range(N_PLANETS):
        m += orbit.orbital_model_AL_vec(t_obs, _planet_rows(X, i), scan_angles)
    return m


def _relast_loglike_vec(X):
    """ln L of the relative (imaging) astrometry for a (k, ndim) array; (k,).
    Each imaged companion's position relative to its host is its own
    photocentre orbit times -s (orbit.relative_orbit_vec); the reflex of the
    host due to other companions is neglected. With RELAST_JITTER the square
    of the jitter is added to both variances of every point."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    ll = np.zeros(X.shape[0])
    for c in RELAST_COMP:
        m = rel_comp == c
        xr, yr = orbit.relative_orbit_vec(rel_t[m], _planet_rows(X, c), X[:, RELAST_POS[c]])
        dx, dy = rel_x[m][None, :] - xr, rel_y[m][None, :] - yr
        if RELAST_JITTER:
            ll += orbit.relative_astrometry_lnlike(dx, dy, rel_sx[m], rel_sy[m], rel_rho[m],
                                                   X[:, RELAST_JIT_IDX])
        else:
            ll += np.sum(rel_lnorm[m] - 0.5 * (rel_ia[m] * dx * dx + 2.0 * rel_ib[m] * dx * dy
                                                + rel_ic[m] * dy * dy), axis=1)
    return ll


def _hip_models(X):
    """Photocentre orbit of all planets on the Hipparcos scans, (k, N_hip)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = np.zeros((X.shape[0], HIP['n']))
    for i in range(N_PLANETS):
        dra, ddec = orbit.astrometric_orbit_vec(HIP['t_day'], _planet_rows(X, i))
        m += hipparcos.orbit_along_scan(HIP['c'], dra, ddec)
    return m


def _robust_models(X, gaia_models=None):
    """Orbit models for ROBUST's rows: Gaia, then Hipparcos if included."""
    m = _astro_models_raw(X) if gaia_models is None else gaia_models
    return np.hstack([m, _hip_models(X)]) if HIPPARCOS else m


def _noise_nu(X):
    X = np.atleast_2d(X)
    return X[:, [NU_IDX, NUH_IDX]] if HIPPARCOS else X[:, NU_IDX]


def _noise_f(X):
    X = np.atleast_2d(X)
    return X[:, [F_IDX, FH_IDX]] if HIPPARCOS else X[:, F_IDX]


_RV_COUPLED_J = None      # column of the parallax among the single-star terms


def _rv_coupled_scale(X):
    """c = |K| x parallax [m/s mas] of the first companion for a (k, ndim) array."""
    from generate_sources import campbell_elements
    P, e, _, A, B, F, G = _planet_rows(np.atleast_2d(X), 0).T
    a0, _, _, inc = campbell_elements(A, B, F, G)
    return orbit.implied_rv_scale(P, e, a0, inc)


def _rv_coupled_setup():
    """Check that the coupling can be used, and find the parallax column."""
    global _RV_COUPLED_J
    if N_PLANETS != 1:
        raise ValueError("EXOPLANETLAB_RV_COUPLED=1 is implemented for one companion "
                         f"(N = {N_PLANETS} fitted): one parallax cannot stand for several K.")
    if ASTRO_NOISE != 'student' or not DR4_EPOCH:
        raise ValueError("EXOPLANETLAB_RV_COUPLED=1 needs the DR4 epoch data and the Student-t "
                         "noise model (the parallax comes from its single-star fit).")
    _RV_COUPLED_J = list(_dr4_epoch['proj_param_names']).index('parallax')


def _rv_coupled_term(X, beta, H):
    """ln of the coupling factor (orbit.coupled_k_lnterm) for a (k, ndim) array,
    from the fitted single-star terms beta (k, p) and their Fisher matrix H."""
    if _RV_COUPLED_J is None:
        _rv_coupled_setup()
    j = _RV_COUPLED_J
    sd = np.sqrt(np.linalg.inv(H)[:, j, j])
    return orbit.coupled_k_lnterm(np.atleast_2d(X)[:, N_PLANETS * ndim_per], _rv_coupled_scale(X),
                                  beta[:, j], sd, K_MAX - K_MIN)


def _rv_coupled_k(X):
    """|K| implied by the orbit and its own best parallax, for a (k, ndim) array."""
    if _RV_COUPLED_J is None:
        _rv_coupled_setup()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    _, beta, _ = ROBUST._terms(_robust_models(X), _noise_nu(X), _noise_f(X))
    return _rv_coupled_scale(X) / beta[:, _RV_COUPLED_J]


def log_likelihood_astro(all_active_params, data, sigma_tot):
    """Along-scan (astrometric) Gaussian log-likelihood only; reads just the TI
    columns, so it is valid for the astrometry-only vector too (see
    log_prior_astro)."""
    resid = (data - model_all(all_active_params)) / sigma_tot
    return -0.5 * np.sum(resid**2 + 2.0 * np.log(sigma_tot))


_RV_TREND_LS = np.linspace(np.log(RV_TREND_MIN), np.log(RV_TREND_MAX), 48)
_RV_TREND_LW = np.log(np.r_[0.5, np.ones(46), 0.5] * (_RV_TREND_LS[1] - _RV_TREND_LS[0])) \
    - np.log(_RV_TREND_LS[-1] - _RV_TREND_LS[0])


def _rv_trend_x():
    """Years from the mean RV epoch: the slope's regressor."""
    t = np.asarray(t_rv, dtype=float)
    return (t - t.mean()) / 365.25


def _rv_trend_marginal(resid, se2):
    """ln of the integral over a common RV slope b (see RV_TREND) of
    exp(-sum (r - b x)^2 / 2 se2) / exp(-sum r^2 / 2 se2), for residuals r
    (K, n_rv) and variances se2 (K, n_rv): with A = sum x^2 / se2 and
    B = sum x r / se2, -ln(1 + sigma_p^2 A) / 2 + B^2 sigma_p^2 / (2 (1 + sigma_p^2 A))
    for each prior scale sigma_p, integrated over ln sigma_p; (K,)."""
    from scipy.special import logsumexp
    x = _rv_trend_x()[None, :]
    A = np.sum(x * x / se2, axis=1)[:, None]
    B = np.sum(x * resid / se2, axis=1)[:, None]
    v = np.exp(2.0 * _RV_TREND_LS)[None, :]
    t = -0.5 * np.log1p(v * A) + 0.5 * B * B * v / (1.0 + v * A)
    return logsumexp(t + _RV_TREND_LW[None, :], axis=1)


def _rv_resid_rows(X):
    """RV residuals data - model (K, n_rv) and variances (K, n_rv) for (K, ndim)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    k_start = N_PLANETS * ndim_per
    gamma_start = k_start + N_PLANETS
    jit_start = gamma_start + N_RV_INST
    rv = X[:, gamma_start:jit_start][:, rv_inst_idx]
    for i in range(N_PLANETS):
        P, e, T0, A, B, F, G = _planet_rows(X, i).T
        rv = rv + orbit.rv_model_vec(t_rv, P, e, T0, X[:, k_start + i],
                                     orbit.omega_from_ti(A, B, F, G))
    jts = X[:, jit_start:jit_start + N_RV_INST][:, rv_inst_idx]
    return rv_obs[None, :] - rv, rv_sigma[None, :] ** 2 + jts ** 2


def log_likelihood_multi(all_active_params, data, sigma_tot):
    """Joint Gaussian log-likelihood: along-scan + optional RV term."""
    if ASTRO_NOISE == 'student':
        _x = np.asarray(all_active_params, dtype=float)[None, :]
        _ll, _beta, _H = ROBUST._terms(_robust_models(_x), _noise_nu(_x), _noise_f(_x))
        ll = float(_ll[0])
        if RV_COUPLED:
            ll += float(_rv_coupled_term(_x, _beta, _H)[0])
    else:
        if ASTRO_JITTER:
            sigma_tot = np.sqrt(sigma_tot**2 + all_active_params[ASTRO_JIT_IDX]**2)
        if ACC_GAUSS is not None:
            ll = float(ACC_GAUSS.loglike(
                (data - model_all(all_active_params))[None, :],
                np.broadcast_to(np.asarray(sigma_tot, dtype=float) ** 2, np.shape(data)))[0])
        else:
            ll = log_likelihood_astro(all_active_params, data, sigma_tot)
    if RV_ENABLED:
        k_start     = N_PLANETS * ndim_per
        gamma_start = k_start + N_PLANETS
        jit_start   = gamma_start + N_RV_INST
        jitters     = np.asarray(all_active_params[jit_start:jit_start + N_RV_INST])
        rv_sigma_eff = np.sqrt(rv_sigma**2 + jitters[rv_inst_idx]**2)
        rv_pred      = _rv_model_all(all_active_params, t_rv, inst_idx=rv_inst_idx)
        rv_resid     = (rv_obs - rv_pred) / rv_sigma_eff
        ll          += -0.5 * np.sum(rv_resid**2 + 2.0 * np.log(rv_sigma_eff))
        if RV_TREND:
            ll += float(_rv_trend_marginal((rv_obs - rv_pred)[None, :],
                                           (rv_sigma_eff ** 2)[None, :])[0])
    if RELAST_ENABLED:
        ll += float(_relast_loglike_vec(np.asarray(all_active_params, dtype=float)[None, :])[0])
    return ll


# ── Vectorised prior / likelihood / posterior (all walkers in one call) ──────
# The same maths as log_prior_multi / log_likelihood_multi for a (k, ndim) array
# of walkers: one Kepler solve, one projection and one chi^2 for the whole
# ensemble. Used by solver_emcee / solver_ptemcee for the joint fit
# (EXOPLANETLAB_VECTORIZE, default on), after a run-time check that it agrees
# with the scalar functions, which stay in use everywhere else (dynesty,
# nessai, RJ solvers, peels).
VECTORIZE = os.environ.get('EXOPLANETLAB_VECTORIZE', '1') == '1'


def _planet_rows(X, i):
    """(k, 7) full [P, e, T0, A, B, F, G] for planet i from a (k, ndim) array --
    the vectorised unpack_i (fixed parameters from true_params_all)."""
    full = np.tile(np.asarray(_TI_TEMPLATE[i], dtype=float), (X.shape[0], 1))
    full[:, FIT_IDX] = X[:, i * ndim_per:(i + 1) * ndim_per]
    return full


def log_prior_multi_vec(X):
    """log_prior_multi for a (k, ndim) array; returns (k,)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    k = X.shape[0]
    ok = np.ones(k, dtype=bool)
    lp = np.zeros(k)
    periods, eccs = [], []
    for i in range(N_PLANETS):
        P, e, T0, A, B, F, G = _planet_rows(X, i).T
        ok &= (P_MIN < P) & (P < P_MAX) & (E_MIN <= e) & (e < 1.0) & (T0_LO < T0) & (T0 < T0_HI)
        for c in (A, B, F, G):
            ok &= (-TI_MAX < c) & (c < TI_MAX)
        if 0 in FIT_IDX:
            lp -= np.log(np.where(P > 0.0, P, 1.0))
        if ORIENT_PRIOR != 'ti':
            _olp = _orient_log_prior(A, B, F, G)
            ok &= np.isfinite(_olp)
            lp += np.where(np.isfinite(_olp), _olp, 0.0)
        periods.append(P)
        eccs.append(e)
    if 0 in FIT_IDX:
        for Pa, Pb in zip(periods, periods[1:]):
            ok &= Pa < Pb
        for Pa, ea, Pb, eb in zip(periods, eccs, periods[1:], eccs[1:]):
            ok &= _orbits_separated(Pa, ea, Pb, eb)
    if RV_ENABLED:
        k_start     = N_PLANETS * ndim_per
        gamma_start = k_start + N_PLANETS
        jit_start   = gamma_start + N_RV_INST
        Ks  = X[:, k_start:gamma_start]
        gms = X[:, gamma_start:jit_start]
        jts = X[:, jit_start:jit_start + N_RV_INST]
        ok &= np.all((Ks >= K_MIN) & (Ks <= K_MAX), axis=1)
        ok &= np.all((gms >= GAMMA_MIN) & (gms <= GAMMA_MAX), axis=1)
        ok &= np.all((jts >= JITTER_MIN) & (jts <= JITTER_MAX), axis=1)
        lp -= np.sum(np.log(np.where(jts > 0.0, jts, 1.0)), axis=1)
        if FIT_MSTAR_PARALLAX:
            m_star = X[:, jit_start + N_RV_INST]
            plx    = X[:, jit_start + N_RV_INST + 1]
            ok &= (m_star > 0.0) & (plx > 0.0)
            lp += (-0.5 * ((m_star - M_STAR_MSUN) / M_STAR_SIGMA) ** 2
                   -0.5 * ((plx - PARALLAX_MAS) / PARALLAX_SIGMA) ** 2)
    if ASTRO_JITTER:
        s = X[:, ASTRO_JIT_IDX]
        ok &= (ASTRO_JITTER_MIN <= s) & (s <= ASTRO_JITTER_MAX)
        lp -= np.log(np.where(s > 0.0, s, 1.0))
    if ASTRO_NOISE == 'student':
        for _i, _j in ((NU_IDX, F_IDX),) + (((NUH_IDX, FH_IDX),) if HIPPARCOS else ()):
            nu, f = X[:, _i], X[:, _j]
            ok &= (NU_MIN <= nu) & (nu <= NU_MAX) & (F_MIN <= f) & (f <= F_MAX)
            lp -= np.log(np.where(nu > 0.0, nu, 1.0)) + np.log(np.where(f > 0.0, f, 1.0))
    for k in RELAST_IDX:
        sc = X[:, k]
        ok &= (RELAST_S_MIN <= sc) & (sc <= RELAST_S_MAX)
        lp -= np.log(np.where(sc > 0.0, sc, 1.0))
    if RELAST_JITTER:
        jr = X[:, RELAST_JIT_IDX]
        ok &= (RELAST_JIT_MIN <= jr) & (jr <= RELAST_JIT_MAX)
        lp -= np.log(np.where(jr > 0.0, jr, 1.0))
    return np.where(ok, lp, -np.inf)


def log_likelihood_multi_vec(X, data, sigma_tot):
    """log_likelihood_multi for a (k, ndim) array; returns (k,). Rows outside
    the prior's support get -inf (they are never accepted anyway, and parts of
    the model are undefined there, e.g. e >= 1)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = np.full(X.shape[0], -np.inf)
    valid = np.isfinite(log_prior_multi_vec(X))
    if not np.any(valid):
        return out
    Xv = X[valid]
    m = np.zeros((Xv.shape[0], len(data)))
    rows = [_planet_rows(Xv, i) for i in range(N_PLANETS)]
    for Q in rows:
        m += orbit.orbital_model_AL_vec(t_obs, Q, scan_angles)
    if ASTRO_NOISE == 'student':
        ll, _beta, _H = ROBUST._terms(_robust_models(Xv, m), _noise_nu(Xv), _noise_f(Xv))
        if RV_COUPLED:
            ll = ll + _rv_coupled_term(Xv, _beta, _H)
    else:
        if _PROJ is not None:
            m = m - (_PROJ.X @ (_PROJ.K @ m.T)).T
        sig2 = np.broadcast_to(np.asarray(sigma_tot, dtype=float) ** 2, m.shape)
        if ASTRO_JITTER:
            sig2 = sig2 + (Xv[:, ASTRO_JIT_IDX] ** 2)[:, None]
        if ACC_GAUSS is not None:
            ll = ACC_GAUSS.loglike(data - m, sig2)
        else:
            ll = -0.5 * np.sum((data - m) ** 2 / sig2 + np.log(sig2), axis=1)
    if RV_ENABLED:
        k_start     = N_PLANETS * ndim_per
        gamma_start = k_start + N_PLANETS
        jit_start   = gamma_start + N_RV_INST
        rv = Xv[:, gamma_start:jit_start][:, rv_inst_idx]
        for i, Q in enumerate(rows):
            P, e, T0, A, B, F, G = Q.T
            rv = rv + orbit.rv_model_vec(t_rv, P, e, T0, Xv[:, k_start + i],
                                         orbit.omega_from_ti(A, B, F, G))
        jts = Xv[:, jit_start:jit_start + N_RV_INST][:, rv_inst_idx]
        se2 = rv_sigma[None, :] ** 2 + jts ** 2
        ll += -0.5 * np.sum((rv_obs - rv) ** 2 / se2 + np.log(se2), axis=1)
        if RV_TREND:
            ll = ll + _rv_trend_marginal(rv_obs[None, :] - rv, se2)
    if RELAST_ENABLED:
        ll = ll + _relast_loglike_vec(Xv)
    out[valid] = ll
    return out


def log_posterior_multi_vec(X, data, sigma_tot):
    """log_prior_multi_vec + log_likelihood_multi_vec; returns (k,)."""
    lp = log_prior_multi_vec(X)
    out = np.full(lp.shape, -np.inf)
    ok = np.isfinite(lp)
    if np.any(ok):
        out[ok] = lp[ok] + log_likelihood_multi_vec(np.atleast_2d(X)[ok], data, sigma_tot)
    return out


def _iso_log_norm(n=2_000_000, seed=20261003):
    """ln of the integral of the isotropic orientation density over its
    support in (A, B, F, G) (priors.iso_log_norm)."""
    return priors.iso_log_norm(TI_MAX, ISO_A0_MIN, _ISO_SIN2_FLOOR, n=n, seed=seed)


def _sep_log_norm(n=2_000_000, seed=20261004):
    """ln of the prior probability that N orbits drawn independently from the
    single-orbit prior are in period order and separated (priors.sep_log_norm)."""
    return priors.sep_log_norm(N_PLANETS, P_MIN, P_MAX, E_MIN, ORBIT_SEPARATION, P_RATIO_MIN,
                               n=n, seed=seed)


_LOGNORM_ORBITS = None      # set where the nested evidence is set up
_T0_PHASE = {}              # T0 column -> (P column, anchor) in that mode


def _phase_to_t0(x):
    """Replace the T0 phases of a nested-evidence vector (or rows) by T0."""
    if not _T0_PHASE:
        return x
    x = np.array(x, dtype=float, copy=True)
    for kT, (kP, anc) in _T0_PHASE.items():
        x[..., kT] = anc + x[..., kT] * x[..., kP]
    return x


def _log_prior_orbits_normalised(x):
    """ln of the normalised prior density of the orbital elements of all
    planets (with the period ordering / separation), as the tempered fit's
    ln Z implies it; -inf outside the support."""
    lp = log_prior_astro(x)
    return lp + _LOGNORM_ORBITS if np.isfinite(lp) else -np.inf


def prior_transform_multi(u):
    """Map unit hypercube to the joint parameter space (TI params + optional K, γ)."""
    return np.array([_ALL_TRANSFORMS[k](u[k]) for k in range(ndim_total)])


# Local indices of log-uniform parameters (P only, at physical index 0).
_LUF_LOCAL_IDX = tuple(j for j, gi in enumerate(FIT_IDX) if gi == 0)


def log_like_rj(leaf_params_log, data, sigma_tot):
    """RJMCMC likelihood for a variable number of planets.

    leaf_params_log has shape (n_active, ndim_per).  The period P (local
    index 0) is passed in log-space; it is exponentiated before evaluating
    the model.
    """
    signal = np.zeros(N_epochs)
    for p_log in leaf_params_log:
        p = p_log.copy()
        for j in _LUF_LOCAL_IDX:
            p[j] = np.exp(p_log[j])
        signal += orbit.orbital_model_AL(t_obs, p, scan_angles, projector=_PROJ)
    resid = (data - signal) / sigma_tot
    return -0.5 * np.sum(resid**2 + 2.0 * np.log(sigma_tot))


# ============================================================
# Initialisation — peel phase (N > 1) or direct guess (N = 1)
# ============================================================

# Independence of the solution: for real data the reference orbit (literature
# or archive) is used only for comparison, never to initialise a fit. Only
# SOURCE_MODE='simulated' may start from true_params_all. Real-data modes start
# from a placeholder, which the N = 1 path below replaces by a data-driven seed
# from the fast screen (detect.py) or the RV periodogram.
if SOURCE_MODE == 'simulated':
    _guess_src = true_params_all[:N_PLANETS]
    if N_PLANETS > N_TRUE:
        # More companions fitted than injected (EXOPLANETLAB_N_FIT, evidence
        # ladder): the extra ones start blind -- a small valid orbit -- at
        # periods above the injected ones, so P_1 < P_2 < ... holds.
        _a0_x = max(20.0 * ISO_A0_MIN,
                    3.0 * float(np.median(sigma_total)) / np.sqrt(len(t_obs)))
        _ti_x = orbit.ti_from_campbell(_a0_x, 0.5, 0.5, np.radians(60.0))
        _p_top = float(np.max(true_params_all[:, 0]))
        _extra_rows = [[min(_p_top * 1.5 ** (k + 1), 0.9 * P_MAX), 0.0, 0.0, *_ti_x]
                       for k in range(N_PLANETS - N_TRUE)]
        _guess_src = np.vstack([true_params_all, _extra_rows])
else:
    _guess_span = float(t_obs.max() - t_obs.min())
    # Blind start (when the fast screen finds nothing to seed from): a small but
    # valid orbit, not A = B = F = G = 0, which is outside the isotropic
    # orientation prior (a0 >= ISO_A0_MIN) and would leave every walker with
    # zero prior probability. a0 = 3 x median error / sqrt(N) is the size of a
    # barely detectable signal; the orientation (i = 60 deg) is arbitrary.
    _a0_blind = max(20.0 * ISO_A0_MIN,
                    3.0 * float(np.median(sigma_total)) / np.sqrt(len(t_obs)))
    _ti_blind = orbit.ti_from_campbell(_a0_blind, 0.5, 0.5, np.radians(60.0))
    _guess_src  = np.array([[2.0 * _guess_span, 0.0, 0.0, *_ti_blind]
                            for _ in range(N_PLANETS)])

initial_guess_all = np.array([
    [tp[0] * 1.02,         # P:  slight overestimate
     min(tp[1] + 0.05, 0.95),  # e:  small offset, capped below 1
     tp[2] + tp[0] * 0.05,     # T0: offset by 5% of period
     tp[3] * 1.1,          # A:  10% larger
     tp[4] * 1.1,          # B
     tp[5] * 1.1,          # F
     tp[6] * 1.1]          # G
    for tp in _guess_src
])
# Clip eccentricity to [EPSILON, E_MAX]
initial_guess_all[:, 1] = np.clip(initial_guess_all[:, 1], EPSILON, E_MAX)

if N_PLANETS == 1:
    _stage('seed selection (screen candidates, RV periodogram, polish)')
    # Real-data sources are never seeded from the reference orbit (see the
    # independence note above). The N = 1 warm-start peel below is short,
    # untempered and astrometry-only, so it needs a data-driven start: the
    # best-scoring candidate of the fast screen, or an RV-anchored one that
    # scores better.
    _seed_from_detect = (
        SOURCE_MODE != 'simulated'
        and '_detect_result' in globals()
        and _detect_result.n_detected >= 1
    )
    if SOURCE_MODE != 'simulated' and not _seed_from_detect:
        print("  NOTE: the fast-screen found no candidate to seed from; the "
              "N=1 warm-start falls back to an uninformative blind start "
              "(the reference orbit is deliberately never used to initialise "
              "a real-data fit).")
    if _seed_from_detect:
        def _seed_joint_score(p7):
            """logL of one candidate orbit against the data the joint fit
            will see: astrometry, plus (if enabled) RV with K and each
            instrument's gamma solved by weighted linear least squares
            (jitter ignored -- this only ranks starting points)."""
            p7 = np.asarray(p7, dtype=float)
            _r = (along_scan_obs - orbit.orbital_model_AL(t_obs, p7, scan_angles, projector=_PROJ)) / sigma_total
            _s = -0.5 * float(np.sum(_r ** 2))
            if not RV_ENABLED:
                return _s
            _om    = orbit.omega_from_ti(*p7[3:7])
            _shape = orbit.rv_model_single(t_rv, p7[0], p7[1], p7[2], 1.0, _om)
            _X = np.zeros((len(t_rv), N_RV_INST + 1))
            _X[np.arange(len(t_rv)), rv_inst_idx] = 1.0
            _X[:, -1] = _shape
            _w = 1.0 / rv_sigma
            _coef = np.linalg.lstsq(_X * _w[:, None], rv_obs * _w, rcond=None)[0]
            if abs(_coef[-1]) > K_MAX:
                return -np.inf
            return _s - 0.5 * float(np.sum(((rv_obs - _X @ _coef) / rv_sigma) ** 2))

        # Candidates: the screen's detections and fits started at the other
        # distinct peaks of the first periodogram, since the highest peak can
        # be an alias. They are ranked by the likelihood of the noise model of
        # the fit; EXOPLANETLAB_SEED_RANK = k starts from the k-th best, so that
        # a fit can be repeated from other starts and the highest ln Z kept
        # (Tools/evidence_ladder.py does that for tags '...-N<k>r<rank>').
        def _seed_rank_score(p7):
            _sc = _seed_joint_score(p7)
            if ASTRO_NOISE != 'student' or not np.isfinite(_sc):
                return _sc
            from scipy.special import gammaln as _gl
            p7 = np.asarray(p7, dtype=float)
            _res = along_scan_obs - orbit.orbital_model_AL(t_obs, p7, scan_angles, projector=_PROJ)
            _nu, _f = detect._t_fit(_res, sigma_total)
            _r = _res / (_f * sigma_total)
            _lt = float(len(_r) * (_gl((_nu + 1) / 2) - _gl(_nu / 2) - 0.5 * np.log(_nu * np.pi)
                                   - np.log(_f))
                        - (_nu + 1) / 2 * np.sum(np.log1p(_r ** 2 / _nu)))
            # replace the Gaussian astrometric term by the Student-t one
            return _lt + (_sc + 0.5 * float(np.sum((_res / sigma_total) ** 2)))

        # Every candidate is refitted as a single orbit to the full data
        # (within +/-20% of its period, e up to 0.9): the screen's detections
        # were fitted to peeled residuals, jointly with each other, and are
        # poor one-orbit fits where there are two signals.
        _raw = [(np.asarray(_p, dtype=float), 'screen') for _p in _detect_result.params]
        _raw += [(np.array([_pk, _e_pk, _t0_pk]), 'periodogram peak') for _pk, _pw_pk, _e_pk, _t0_pk in
                 (_detect_result.screens[0].get('top_peaks', []) if _detect_result.screens else [])]
        # (with the robust screen: to the data clipped at 3 f sigma, as the
        # screen's own refinement -- least squares on heavy-tailed data moves
        # an orbit to fit the outliers)
        _y_rf, _s_rf = ((detect._robustify(along_scan_obs, sigma_total)[1],
                         detect._robustify(along_scan_obs, sigma_total)[2])
                        if SCREEN_ROBUST else (along_scan_obs, sigma_total))
        _cands = []
        for _c_raw, _c_src in _raw:
            try:
                _x_c, _ll_c, _ok_c = detect._refine_local_fit(
                    t_obs, scan_angles, _y_rf, _s_rf,
                    float(_c_raw[0]), min(float(_c_raw[1]), 0.9), float(_c_raw[2]),
                    P_MIN, P_MAX, 0.9, TI_MAX, 8, 0)
            except Exception as _ex:
                print(f"  Candidate at {_c_raw[0]:.1f} d not refined ({_ex})")
                _x_c, _ok_c = _c_raw, len(_c_raw) == 7
            if _ok_c and not any(abs(np.log(_x_c[0] / _c[0][0])) < 0.05 for _c in _cands):
                _cands.append((np.asarray(_x_c, dtype=float), _c_src))
        if not _cands:
            _cands = [(np.asarray(_p, dtype=float), 'screen') for _p in _detect_result.params]
        _cands = sorted(((_seed_rank_score(_c[0]), _c[0], _c[1]) for _c in _cands),
                        key=lambda _c: -_c[0])
        print(f"  Starting candidates, ranked by ln L "
              f"({'Student-t' if ASTRO_NOISE == 'student' else 'Gaussian'} noise):")
        for _k, (_sc_k, _p_k, _src_k) in enumerate(_cands):
            print(f"    {_k + 1}. P = {_p_k[0]:9.2f} d, e = {_p_k[1]:.3f}, ln L = {_sc_k:10.2f}"
                  f"  ({_src_k}){'   <- start' if _k == min(SEED_RANK, len(_cands)) - 1 else ''}")
        if SEED_RANK > len(_cands):
            print(f"  NOTE: EXOPLANETLAB_SEED_RANK = {SEED_RANK} but there are only "
                  f"{len(_cands)} candidates; the last one is used.")
        SEED_CANDIDATES = [dict(rank=_k + 1, P=float(_c[1][0]), e=float(_c[1][1]),
                                lnL=float(_c[0]), source=_c[2]) for _k, _c in enumerate(_cands)]
        _k_seed   = min(SEED_RANK, len(_cands)) - 1
        _dc_best  = _cands[_k_seed][1]
        _dc_score = _seed_joint_score(_dc_best)
        _dc_src   = (f"the best-scoring ({_cands[_k_seed][2]})" if _k_seed == 0 else
                     f"the rank-{_k_seed + 1} ({_cands[_k_seed][2]})")

        # The screen is astrometry-only and can land on an alias when the
        # astrometric signal is marginal, while the RVs often constrain the
        # period far better. With RVs, an RV-only periodogram is run too; a
        # strong, unambiguous period from it is refined against the astrometry
        # and competes with the screen's candidate.
        if RV_ENABLED and SEED_RANK == 1:
            try:
                from astropy.timeseries import LombScargle
                _rv_resid = rv_obs.astype(float).copy()
                for _ii in range(N_RV_INST):
                    _m = (rv_inst_idx == _ii)
                    _rv_resid[_m] -= np.mean(rv_obs[_m])
                _ls_freq, _ls_power = LombScargle(t_rv, _rv_resid, rv_sigma).autopower(
                    minimum_frequency=1.0 / min(P_MAX, 5000.0),
                    maximum_frequency=1.0 / P_MIN)
                _rv_p_best  = float(1.0 / _ls_freq[np.argmax(_ls_power)])
                _rv_p_power = float(_ls_power.max())
                print(f"  RV-only Lomb-Scargle (per-instrument mean-subtracted): "
                      f"best P={_rv_p_best:.3f}d (peak power={_rv_p_power:.3f})")
                if _rv_p_power > 0.3:   # a real, unambiguous periodic signal
                    _cp_rv, _sp_rv = np.cos(scan_angles), np.sin(scan_angles)
                    _e_grid = np.linspace(0.0, 0.6, 7)
                    _pw, _e0, _T0_0 = detect._profile_power_at_period(
                        t_obs, along_scan_obs, sigma_total, _rv_p_best,
                        _cp_rv, _sp_rv, _e_grid, 16)
                    _rv_x, _rv_logl, _rv_ok = detect._refine_local_fit(
                        t_obs, scan_angles, along_scan_obs, sigma_total,
                        _rv_p_best, _e0, _T0_0, P_MIN, P_MAX, 0.6, TI_MAX, 8, 0)
                    if _rv_ok:
                        # Taken only if it fits the joint data better: with few
                        # RV points a spurious periodogram peak is easy, and it
                        # must not override a well-fitting screen candidate.
                        _rv_score = _seed_joint_score(_rv_x)
                        print(f"  RV-anchored candidate refined against the "
                              f"astrometry: P={_rv_x[0]:.2f}d "
                              f"e={_rv_x[1]:.3f}; joint score "
                              f"{_rv_score:.1f} vs {_dc_score:.1f} for the "
                              f"screening candidate (P={_dc_best[0]:.2f}d)")
                        if _rv_score > _dc_score:
                            _dc_best = _rv_x
                            _dc_src  = "the RV-anchored (RV Lomb-Scargle + astrometry-refined)"
            except Exception as _ex:
                print(f"  RV-only period search skipped ({_ex})")

        # Polish the chosen candidate. The screen caps e at 0.6 and refines P
        # only within +/-20%, which leaves a higher-e or longer-period signal on
        # that boundary (Gaia BH3). Global search over (P, e, T0) around the
        # candidate, with A, B, F, G solved by linear least squares from the
        # projected astrometry and, with RVs, K and each instrument's gamma from
        # the RV data: the joint chi2 of the data the joint fit will see.
        from scipy.optimize import differential_evolution as _de

        def _seed_objective(P_, e_, T0_):
            _M = 2.0 * np.pi * (t_obs - T0_) / P_
            _E = orbit.kepler_eq(_M, e_)
            _Xk, _Yk = np.cos(_E) - e_, np.sqrt(1.0 - e_ ** 2) * np.sin(_E)
            _cp, _sp = np.cos(scan_angles), np.sin(scan_angles)
            _B = np.column_stack([_Xk * _cp, _Xk * _sp, _Yk * _cp, _Yk * _sp])   # A,B,F,G
            if _PROJ is not None:
                _B = _PROJ.apply(_B)
            _w  = 1.0 / sigma_total
            _ti = np.linalg.lstsq(_B * _w[:, None], along_scan_obs * _w, rcond=None)[0]
            _c  = float(np.sum(((along_scan_obs - _B @ _ti) * _w) ** 2))
            if RV_ENABLED:
                _om = orbit.omega_from_ti(*_ti)
                _Xr = np.zeros((len(t_rv), N_RV_INST + 1))
                _Xr[np.arange(len(t_rv)), rv_inst_idx] = 1.0
                _Xr[:, -1] = orbit.rv_model_single(t_rv, P_, e_, T0_, 1.0, _om)
                _wr = 1.0 / rv_sigma
                _cf = np.linalg.lstsq(_Xr * _wr[:, None], rv_obs * _wr, rcond=None)[0]
                if abs(_cf[-1]) > K_MAX:
                    return 1e30, _ti
                _c += float(np.sum(((rv_obs - _Xr @ _cf) * _wr) ** 2))
            return _c, _ti

        try:
            _t0e  = float(t_obs.min())
            _plo  = max(P_MIN, 0.4 * float(_dc_best[0]))
            _phi  = min(P_MAX, 2.5 * float(_dc_best[0]))
            if SEED_RANK > 1:       # an alternative start stays on its own peak
                _plo = max(P_MIN, float(_dc_best[0]) / 1.2)
                _phi = min(P_MAX, float(_dc_best[0]) * 1.2)
            _cur  = _seed_objective(float(_dc_best[0]), float(_dc_best[1]), float(_dc_best[2]))[0]
            _de_r = _de(lambda x: _seed_objective(np.exp(x[0]), x[1],
                                                  _t0e + x[2] * np.exp(x[0]))[0],
                        [(np.log(_plo), np.log(_phi)), (0.0, 0.95), (0.0, 1.0)],
                        seed=0, popsize=20, maxiter=150, tol=1e-9, polish=True)
            if _de_r.fun < _cur:
                _Pp = float(np.exp(_de_r.x[0]))
                _ep = float(_de_r.x[1])
                _Tp = _t0e + float(_de_r.x[2]) * _Pp
                _tip = _seed_objective(_Pp, _ep, _Tp)[1]
                print(f"  Polished the seed: chi2 {_cur:.0f} -> {_de_r.fun:.0f}  "
                      f"(P {float(_dc_best[0]):.1f} -> {_Pp:.1f} d, "
                      f"e {float(_dc_best[1]):.3f} -> {_ep:.3f})")
                _dc_best = np.array([_Pp, _ep, _Tp, *_tip])
                _dc_src  = _dc_src + " (polished)"
        except Exception as _ex:
            print(f"  Seed polishing skipped ({_ex})")

        print(f"  Seeding the N=1 warm-start from {_dc_src} candidate "
              f"(P={_dc_best[0]:.2f}d, e={_dc_best[1]:.3f}) rather than "
              f"true_params_all (no usable literature orbit for this "
              f"source/data).")
        peel_best = [np.asarray(_dc_best, dtype=float)[FIT_IDX]]
    else:
        peel_best = [initial_guess_all[0][FIT_IDX]]
    peel_results = []

    _stage('warm-start peel (N=1)')
    # Always run a warm-start peel, whatever the solver: a joint search over the
    # full prior can miss the mode.
    _peel_label = PEEL_SOLVER.upper()
    print(f"\n{'='*66}")
    print(f"  [{SOLVER} warm-start] {_peel_label} N=1 — estimating peel posterior …")
    print(f"{'='*66}")

    def _peel_logl_1(p, data, sigma_tot):
        sig = orbit.orbital_model_AL(t_obs, unpack_i(p, 0), scan_angles, projector=_PROJ)
        resid = (data - sig) / sigma_tot
        return -0.5 * np.sum(resid**2 + 2.0 * np.log(sigma_tot))

    def _peel_logp_1(p):
        return log_prior_single(p, 0)

    if PEEL_SOLVER == 'nessai':
        _active_txs_1 = [_TRANSFORMS_SINGLE[gi] for gi in FIT_IDX]
        _pt_peel_1    = lambda u, _t=_active_txs_1: np.array([t(u[j])
                                                               for j, t in enumerate(_t)])
        # Box around the data-driven seed (never around a truth-derived
        # initial guess -- see the independence note at initial_guess_all).
        _box_cand = (_dc_best if _seed_from_detect
                     else _detect_result.params[0]
                     if ('_detect_result' in globals()
                         and _detect_result.n_detected >= 1) else None)
        _box_1 = None
        if NESSAI_PEEL_BOX and _box_cand is not None:
            _box_1 = _peel_box(_box_cand)
            _pt_peel_1, _peel_logp_1 = _peel_box_prior(*_box_1)
            print(f"  Prior box around fast-screen candidate: "
                  f"P [{_box_1[0][0]:.1f}, {_box_1[1][0]:.1f}] d, "
                  f"T0 [{_box_1[0][2]:.1f}, {_box_1[1][2]:.1f}] d, "
                  f"A..G +-{0.5 * (_box_1[1][3] - _box_1[0][3]):.2f} mas")
        result_peel1  = solver_nessai.run(
            _peel_logl_1, _peel_logp_1, _pt_peel_1,
            args=(along_scan_obs, sigma_total),
            ndim=ndim_per,
            output=os.path.join(outdir, 'nessai_peel_1'),
            nlive=NLIVE_PEEL,
            stopping_criterion=STOPPING_PEEL,
        )
        if _box_1 is not None:
            _warn_peel_box_edge(result_peel1.best_params, *_box_1, "N=1 peel")
    else:
        p0_peel1 =solver_emcee.make_p0(peel_best[0], PARAM_LO, PARAM_HI, FIT_IDX)
        p0_peel1[:, 1] = np.clip(p0_peel1[:, 1], EPSILON, E_MAX)
        result_peel1 = solver_emcee.run(
            _peel_logl_1, _peel_logp_1, None,
            args=(along_scan_obs, sigma_total),
            p0=p0_peel1,
            nsteps_burn=NSTEPS_PEEL_BURN,
            nsteps_prod=NSTEPS_PEEL_PROD,
        )

    print(f"\n{_peel_label} warm-start best-fit:")
    for gi in FIT_IDX:
        j   = FIT_IDX.index(gi)
        val = result_peel1.best_params[j]
        print(f"  {PARAM_NAMES[gi]:6s} = {val:.4f} {PARAM_UNITS[gi]}")

    _peel_acc = getattr(result_peel1, 'acceptance_fraction', None)
    if _peel_acc is not None and float(_peel_acc) < 0.05:
        print(f"  WARNING: the N=1 warm-start peel accepted only "
              f"{float(_peel_acc):.3f} of proposals -- its walkers are "
              f"effectively frozen (a pathological start), so the best-fit "
              f"above should not be trusted as a seed for the joint fit.")

    peel_best    = [result_peel1.best_params]
    peel_results = [result_peel1]

else:
    peel_best    = []
    peel_results = []
    peel_data    = along_scan_obs.copy()
    _stage('peels (multi-planet warm start)')
    _probe_peel_result = None   # sub-threshold survey-probe peel, kept for the
                                # RJ birth proposal (strongest unexplained peak)

    _is_td = SOLVER in ('rjmcmc', 'rjflow')
    N_PEEL = solver.N_MAX if _is_td else N_PLANETS

    for peel_i in range(N_PEEL):
        _extra = peel_i >= N_PLANETS
        print(f"\n{'='*66}")
        if _extra:
            print(f"  Peel {peel_i+1}/{N_PEEL} — survey probe on residual")
        else:
            print(f"  Peel {peel_i+1}/{N_PEEL} — planet {peel_i+1} on residual data")
        print(f"{'='*66}")

        _si = 0 if _extra else peel_i

        def _peel_logl(p, data, sigma_tot, si=_si):
            sig = orbit.orbital_model_AL(t_obs, unpack_i(p, si), scan_angles, projector=_PROJ)
            resid = (data - sig) / sigma_tot
            return -0.5 * np.sum(resid**2 + 2.0 * np.log(sigma_tot))

        def _peel_logp(p, si=_si):
            return log_prior_single(p, si)

        if PEEL_SOLVER == 'nessai':
            _active_txs = [_TRANSFORMS_SINGLE[gi] for gi in FIT_IDX]
            _pt_peel    = lambda u, _t=_active_txs: np.array([t(u[j])
                                                               for j, t in enumerate(_t)])
            _box = None
            if (NESSAI_PEEL_BOX and not _extra
                    and '_detect_result' in globals()
                    and peel_i < _detect_result.n_detected):
                _lp_flags = getattr(_detect_result, 'long_period', None)
                _box = _peel_box(_detect_result.params[peel_i],
                                 long_period=bool(_lp_flags and _lp_flags[peel_i]))
                _pt_peel, _peel_logp = _peel_box_prior(*_box)
                print(f"  Prior box around fast-screen candidate {peel_i+1}: "
                      f"P [{_box[0][0]:.1f}, {_box[1][0]:.1f}] d, "
                      f"T0 [{_box[0][2]:.1f}, {_box[1][2]:.1f}] d, "
                      f"A..G +-{0.5 * (_box[1][3] - _box[0][3]):.2f} mas")
            res_i = solver_nessai.run(
                _peel_logl, _peel_logp, _pt_peel,
                args=(peel_data, sigma_total),
                ndim=ndim_per,
                output=os.path.join(outdir, f'nessai_peel_{peel_i+1}'),
                nlive=NLIVE_PEEL,
                stopping_criterion=STOPPING_PEEL,
                max_iteration=SURVEY_PROBE_MAX_ITERATION if _extra else None,
            )
            if _box is not None:
                _warn_peel_box_edge(res_i.best_params, *_box, f"Peel {peel_i+1}")
        else:
            guess_i = initial_guess_all[0][FIT_IDX].copy() if _extra else \
                      initial_guess_all[peel_i][FIT_IDX]
            p0_i = solver_emcee.make_p0(guess_i, PARAM_LO, PARAM_HI, FIT_IDX)
            p0_i[:, 1] = np.clip(p0_i[:, 1], EPSILON, E_MAX)
            res_i = solver_emcee.run(
                _peel_logl, _peel_logp, None,
                args=(peel_data, sigma_total),
                p0=p0_i,
                nsteps_burn=NSTEPS_PEEL_BURN,
                nsteps_prod=NSTEPS_PEEL_PROD,
            )

        h_fit       = orbit.orbital_model_AL(t_obs, unpack_i(res_i.best_params, _si), scan_angles, projector=_PROJ)
        logl_signal = -0.5 * np.sum(((peel_data - h_fit) / sigma_total)**2
                                     + 2.0 * np.log(sigma_total))
        logl_noise  = -0.5 * np.sum((peel_data / sigma_total)**2
                                     + 2.0 * np.log(sigma_total))
        delta_logl  = logl_signal - logl_noise
        snr_peel    = np.sqrt(max(0.0, 2.0 * delta_logl))

        print(f"\nPeel {peel_i+1} best-fit:")
        for gi in FIT_IDX:
            j = FIT_IDX.index(gi)
            print(f"  {PARAM_NAMES[gi]:6s} = {res_i.best_params[j]:.4f} {PARAM_UNITS[gi]}")
        print(f"  Δln L = {delta_logl:.1f}   SNR ≈ {snr_peel:.1f}")

        if _extra and snr_peel < SURVEY_PROBE_SNR_MIN:
            print(f"  Survey probe SNR ≈ {snr_peel:.1f} < {SURVEY_PROBE_SNR_MIN} "
                  f"-- no additional signal found here (whether because none "
                  f"exists, or the {SURVEY_PROBE_MAX_ITERATION}-iteration cap "
                  f"was hit first). Discarding this slot.")
            if PROBE_CAP_NMAX:
                print(f"  EXOPLANETLAB_PROBE_CAP_NMAX=1: effective N capped at "
                      f"{len(peel_best)}.")
                solver.N_MAX = len(peel_best)
            else:
                print(f"  Keeping N_MAX = {solver.N_MAX}; the joint search runs "
                      f"with {len(peel_best)} active source(s) and "
                      f"{solver.N_MAX - len(peel_best)} dormant leaf/leaves.")
                # Keep this probe fit as an extra RJ birth-proposal component
                # so a birth can be proposed where an undetected planet would sit.
                _probe_peel_result = res_i
            break

        peel_best.append(res_i.best_params)
        peel_results.append(res_i)
        peel_data = peel_data - h_fit

    # Sort peel results by ascending period
    if len(peel_results) > 1 and 0 in FIT_IDX:
        p_col    = FIT_IDX.index(0)
        _order   = sorted(range(len(peel_best)), key=lambda i: peel_best[i][p_col])
        peel_best    = [peel_best[i]    for i in _order]
        peel_results = [peel_results[i] for i in _order]

    # The joint start must lie inside the prior, including ORBIT_SEPARATION.
    # When more companions are fitted than there are signals (the evidence
    # ladder, EXOPLANETLAB_N_FIT), a peel can re-fit the residual of a strong
    # companion at its own period (Gaia-4, N = 2). The weaker orbit of such a
    # crossing pair restarts blind -- a small valid orbit -- at a period just
    # clear of the stronger one's; the tempered joint fit then finds its place.
    if len(peel_best) > 1 and 0 in FIT_IDX and not _is_td:
        _pc = FIT_IDX.index(0)
        _a0_x = max(20.0 * ISO_A0_MIN,
                    3.0 * float(np.median(sigma_total)) / np.sqrt(len(t_obs)))
        _ti_x = orbit.ti_from_campbell(_a0_x, 0.5, 0.5, np.radians(60.0))
        for _ in range(len(peel_best)):
            _rows = [unpack_i(pb, i) for i, pb in enumerate(peel_best)]
            _bad = next((k for k in range(1, len(_rows))
                         if not _orbits_separated(_rows[k-1][0], _rows[k-1][1],
                                                  _rows[k][0], _rows[k][1])), None)
            if _bad is None:
                break
            _amp = [float(np.sum(np.asarray(r[3:7]) ** 2)) for r in _rows]
            _weak, _strong = ((_bad, _bad - 1) if _amp[_bad] <= _amp[_bad - 1]
                              else (_bad - 1, _bad))
            _Ps, _es = _rows[_strong][0], _rows[_strong][1]
            # the nearest period ratio (above, else below) that clears the
            # stronger orbit with some margin, for a blind orbit of e = 0.1
            _Pn = next((_Ps * _f for _f in (1.5, 2.0, 3.0, 5.0)
                        if _Ps * _f < 0.9 * P_MAX
                        and _orbits_separated(_Ps, _es, _Ps * _f / 1.2, 0.1)),
                       next((_Ps / _f for _f in (1.5, 2.0, 3.0, 5.0)
                             if _orbits_separated(_Ps / _f * 1.2, 0.1, _Ps, _es)),
                            _Ps / 5.0))
            _new = np.array([_Pn, 0.1, 0.0, *_ti_x])
            print(f"  Start point: peel orbits at P = {_rows[_bad-1][0]:.1f} and "
                  f"{_rows[_bad][0]:.1f} d too close (ORBIT_SEPARATION={ORBIT_SEPARATION}); the weaker "
                  f"restarts blind at P = {_Pn:.1f} d.")
            peel_best[_weak] = _new[FIT_IDX]
            _order = sorted(range(len(peel_best)), key=lambda i: peel_best[i][_pc])
            peel_best    = [peel_best[i]    for i in _order]
            peel_results = [peel_results[i] for i in _order]


# Ranked starts for fits with two or more orbits. The peel fits one orbit after
# another, each to the residual of the previous ones, and so gives one start;
# where the periodogram has aliases, or two companions of similar strength, the
# tempered fit then mixes period modes or stays in one. A beam search over the
# periodogram peaks gives alternatives: at each level every kept combination is
# extended by the START_BRANCH highest distinct peaks of the periodogram of its
# residual (robust, as the screen), each refitted, the orbits then re-fitted in
# turn against the others; the START_BEAM best by likelihood are kept. The fit
# starts from the best of these and the peel start; EXOPLANETLAB_SEED_RANK = k
# starts from the k-th. EXOPLANETLAB_START_SEARCH=0 keeps the peel start.
START_SEARCH = (os.environ.get('EXOPLANETLAB_START_SEARCH', '1').strip() != '0'
                and N_PLANETS >= 2 and SOLVER == 'emcee' and SOURCE_MODE != 'simulated'
                and len(FIT_IDX) == 7 and len(peel_best) == N_PLANETS)
START_BRANCH, START_BEAM = 3, 4
if START_SEARCH:
    _stage('ranked starts (beam search over periodogram peaks)')

    def _ss_model(rows):
        _m = np.zeros(len(t_obs))
        for _r in rows:
            _m = _m + orbit.orbital_model_AL(t_obs, np.asarray(_r, float), scan_angles,
                                             projector=_PROJ)
        return _m

    def _ss_score(rows):
        _res = along_scan_obs - _ss_model(rows)
        if ASTRO_NOISE != 'student':
            return -0.5 * float(np.sum((_res / sigma_total) ** 2))
        from scipy.special import gammaln as _gl
        _nu, _f = detect._t_fit(_res, sigma_total)
        _r = _res / (_f * sigma_total)
        return float(len(_r) * (_gl((_nu + 1) / 2) - _gl(_nu / 2) - 0.5 * np.log(_nu * np.pi)
                                - np.log(_f))
                     - (_nu + 1) / 2 * np.sum(np.log1p(_r ** 2 / _nu)))

    def _ss_valid(rows):
        _r = sorted(rows, key=lambda x: x[0])
        return all(bool(_orbits_separated(_r[k - 1][0], _r[k - 1][1], _r[k][0], _r[k][1]))
                   for k in range(1, len(_r)))

    def _ss_clip(res):
        """Residual and errors for a least-squares refit: clipped at 3 f sigma
        with the robust screen (as the screen's own refinement), else as is."""
        if not SCREEN_ROBUST:
            return res, sigma_total
        _rb = detect._robustify(res, sigma_total)
        return _rb[1], _rb[2]

    def _ss_refit(rows, rounds=2):
        """Re-fit each orbit in turn against the residual of the others."""
        rows = [np.asarray(_r, float) for _r in rows]
        for _ in range(rounds):
            for k in range(len(rows)):
                _res = along_scan_obs - _ss_model(rows[:k] + rows[k + 1:])
                _yc, _sc = _ss_clip(_res)
                try:
                    _x, _ll, _ok = detect._refine_local_fit(
                        t_obs, scan_angles, _yc, _sc, float(rows[k][0]),
                        min(float(rows[k][1]), 0.9), float(rows[k][2]),
                        P_MIN, P_MAX, 0.9, TI_MAX, 2, 0)
                except Exception:
                    _ok = False
                if _ok:
                    rows[k] = np.asarray(_x, float)
        return rows

    _ss_cp, _ss_sp = np.cos(scan_angles), np.sin(scan_angles)
    _ss_periods = detect._make_period_grid(float(t_obs.max() - t_obs.min()), P_MIN, P_MAX, 4)
    _ss_egrid = np.linspace(0.0, 0.6, 6)
    _ss_beam = [[]]
    for _lvl in range(N_PLANETS):
        _ss_new = []
        for _combo in _ss_beam:
            _res = along_scan_obs - _ss_model(_combo)
            _y, _s = ((detect._robustify(_res, sigma_total)[0], detect._robustify(_res, sigma_total)[2])
                      if SCREEN_ROBUST else (_res, sigma_total))
            _pw, _es, _t0s = detect._profile_grid(t_obs, _y, _s, _ss_periods, _ss_cp, _ss_sp,
                                                  _ss_egrid, 16, n_pool=1)
            _top = []
            for _j in np.argsort(-_pw):
                if (all(abs(np.log(_ss_periods[_j] / _ss_periods[_k])) > detect.TOP_PEAK_SEP
                        for _k in _top)
                        and all(abs(np.log(_ss_periods[_j] / _r[0])) > np.log(P_RATIO_MIN)
                                for _r in _combo)):
                    _top.append(_j)
                if len(_top) == START_BRANCH:
                    break
            _yc, _sc = _ss_clip(_res)
            for _j in _top:
                try:
                    _x, _ll, _ok = detect._refine_local_fit(
                        t_obs, scan_angles, _yc, _sc, float(_ss_periods[_j]),
                        float(_es[_j]), float(_t0s[_j]), P_MIN, P_MAX, 0.9, TI_MAX, 4, 0)
                except Exception:
                    _ok = False
                if _ok:
                    _ss_new.append(_combo + [np.asarray(_x, float)])
        _ss_scored = []
        for _combo in _ss_new:
            _combo = sorted(_ss_refit(_combo) if len(_combo) > 1 else _combo, key=lambda x: x[0])
            if not _ss_valid(_combo):
                continue
            if any(all(abs(np.log(_a[0] / _b[0])) < 0.05 for _a, _b in zip(_combo, _c[1]))
                   for _c in _ss_scored):
                continue
            _ss_scored.append((_ss_score(_combo), _combo, 'periodogram peaks'))
        _ss_scored.sort(key=lambda c: -c[0])
        if not _ss_scored:
            break
        _ss_beam = [_c[1] for _c in _ss_scored[:START_BEAM]]
    _ss_peel = sorted([unpack_i(_pb, _i) for _i, _pb in enumerate(peel_best)], key=lambda x: x[0])
    if _ss_scored and len(_ss_scored[0][1]) == N_PLANETS:
        if not any(all(abs(np.log(_a[0] / _b[0])) < 0.05 for _a, _b in zip(_ss_peel, _c[1]))
                   for _c in _ss_scored):
            _ss_scored.append((_ss_score(_ss_peel), _ss_peel, 'peel'))
        else:
            _ss_scored = [(_c[0], _c[1], _c[2] + (' = peel' if all(
                abs(np.log(_a[0] / _b[0])) < 0.05 for _a, _b in zip(_ss_peel, _c[1])) else ''))
                for _c in _ss_scored]
        _ss_scored.sort(key=lambda c: -c[0])
        _k_seed = min(SEED_RANK, len(_ss_scored)) - 1
        print(f"  Starting candidates for {N_PLANETS} orbits, ranked by ln L "
              f"({'Student-t' if ASTRO_NOISE == 'student' else 'Gaussian'} noise):")
        for _k, (_sc_k, _c_k, _src_k) in enumerate(_ss_scored):
            print(f"    {_k + 1}. P = " + ', '.join(f"{_r[0]:8.2f}" for _r in _c_k)
                  + f" d, e = " + ', '.join(f"{_r[1]:.2f}" for _r in _c_k)
                  + f", ln L = {_sc_k:10.2f}  ({_src_k}){'   <- start' if _k == _k_seed else ''}")
        SEED_CANDIDATES = [dict(rank=_k + 1, P=[float(_r[0]) for _r in _c[1]],
                                e=[float(_r[1]) for _r in _c[1]], lnL=float(_c[0]), source=_c[2])
                           for _k, _c in enumerate(_ss_scored)]
        peel_best = [np.asarray(_r, float)[FIT_IDX] for _r in _ss_scored[_k_seed][1]]
    else:
        print("  No complete starting combination found; the peel start is kept.")
    if os.environ.get('EXOPLANETLAB_STOP_AFTER_START', '0').strip() == '1':
        sys.exit("\nEXOPLANETLAB_STOP_AFTER_START=1: stopping after the start search.\n")


# ============================================================
# Joint inference
# ============================================================

_stage('joint-fit preparation (start point, optimisation, pre-refinement, checks)')
initial_joint = np.concatenate(peel_best)
if RV_ENABLED:
    _k_init = (np.array((list(K_TRUE) + [100.0] * N_PLANETS)[:N_PLANETS])
               if RV_MODE == 'simulated' else np.full(N_PLANETS, 500.0))
    if RV_MODE == 'simulated':
        _gamma_init  = np.array([GAMMA_TRUE])
        _jitter_init = np.full(N_RV_INST, RV_SIGMA_INST)
    else:
        # Per-instrument gamma: that instrument's own median RV is a much
        # better zero-order guess than 0 -- offsets between spectrographs
        # are often 10s-100s of m/s. Jitter: that instrument's own median
        # quoted error, floored at JITTER_MIN.
        _gamma_init  = np.array([np.median(rv_obs[rv_inst_idx == _i])
                                  for _i in range(N_RV_INST)])
        # Start strictly inside the bounds (a start value clipped onto a
        # bound leaves every walker identical in that dimension, and the
        # ensemble stretch move can then never move it).
        _jitter_init = np.array([
            float(np.clip(np.median(rv_sigma[rv_inst_idx == _i]),
                          10.0 * JITTER_MIN, 0.1 * JITTER_MAX))
            for _i in range(N_RV_INST)])
    initial_joint = np.concatenate([initial_joint, _k_init, _gamma_init, _jitter_init])
    if FIT_MSTAR_PARALLAX:
        initial_joint = np.concatenate(
            [initial_joint, [M_STAR_MSUN], [PARALLAX_MAS]])
if ASTRO_JITTER:
    # Start from the excess scatter of the data about the peel model,
    # sqrt(mean(r^2) - mean(sigma^2)), floored at 10% of the median error and
    # kept strictly inside the bounds (a start clipped onto a bound would
    # leave every walker identical in this dimension).
    _r_peel = along_scan_obs - model_all(initial_joint)
    _s_ast0 = np.sqrt(max(np.mean(_r_peel**2) - np.mean(sigma_total**2),
                          (0.1 * np.median(sigma_total))**2))
    _s_ast0 = float(np.clip(_s_ast0, 10.0 * ASTRO_JITTER_MIN, 0.5 * ASTRO_JITTER_MAX))
    initial_joint = np.concatenate([initial_joint, [_s_ast0]])
if ASTRO_NOISE == 'student':
    # Moderate tails and the quoted errors (the DR4 pre-release sources give
    # nu ~ 2-8, f ~ 0.8-1); the fit moves them freely.
    initial_joint = np.concatenate([initial_joint, [5.0, 1.0] * (2 if HIPPARCOS else 1)])
if RELAST_IDX:
    # Each scale from a least-squares fit of the relative positions to the
    # peel orbit's photocentre track (s = -<r . p> / <p . p>), or the ratio of
    # their amplitudes if the peel orbit is so far off that the sign comes out
    # wrong; the joint start optimisation below refines it with the orbit.
    _s0 = []
    for _c in RELAST_COMP:
        _m = rel_comp == _c
        _px, _py = orbit.astrometric_orbit_vec(
            rel_t[_m], np.asarray(unpack_i(initial_joint[_c * ndim_per:(_c + 1) * ndim_per], _c))[None, :])
        _pp = float(np.sum(_px ** 2 + _py ** 2))
        _rp = -float(np.sum(rel_x[_m] * _px[0] + rel_y[_m] * _py[0]))
        _sc = (_rp / _pp if _rp > 0 else
               np.sqrt(np.sum(rel_x[_m] ** 2 + rel_y[_m] ** 2) / max(_pp, 1e-30)))
        _s0.append(float(np.clip(_sc, 1.01 * RELAST_S_MIN, 0.99 * RELAST_S_MAX)))
    initial_joint = np.concatenate([initial_joint, _s0])
    print("  Relative-astrometry scale start: " + ", ".join(
        f"s_{_c + 1} = {_v:.4g}" for _c, _v in zip(RELAST_COMP, _s0)))
    if RELAST_JITTER:
        # a tenth of the median error: small, and well inside the bounds
        initial_joint = np.concatenate([initial_joint, [10.0 * RELAST_JIT_MIN]])

active_names_joint = (
    [PARAM_NAMES[gi] for gi in FIT_IDX]
    if N_PLANETS == 1 else
    [PARAM_NAMES[gi] + f'_p{j+1}' for j in range(N_PLANETS) for gi in FIT_IDX]
)
if RV_ENABLED:
    active_names_joint += ([f'K_p{j+1}' for j in range(N_PLANETS)]
                            + [f'gamma_{_name}' for _name in RV_INSTRUMENTS]
                            + [f'jitter_{_name}' for _name in RV_INSTRUMENTS])
    if FIT_MSTAR_PARALLAX:
        active_names_joint += ['M_star', 'parallax']
if ASTRO_JITTER:
    active_names_joint += ['jitter_astro']
if ASTRO_NOISE == 'student':
    active_names_joint += ['nu_astro', 'f_astro'] + (['nu_hip', 'f_hip'] if HIPPARCOS else [])
active_names_joint += ['s_rel' if N_PLANETS == 1 else f's_rel_p{_c + 1}' for _c in RELAST_COMP]
if RELAST_JITTER:
    active_names_joint += ['jitter_rel']


def _emcee_joint_prerefine(purpose):
    """One full emcee joint fit over the wide multi-planet prior.

    Used as an unbiased warm start (rjmcmc/rjflow) and as the source of
    dynesty's search box.  The raw single-planet peels are contaminated by
    inter-planet crosstalk -- worst at a small-integer period ratio, e.g. the
    2:1-commensurate 2-planet test -- whereas this joint fit is not.  Returns a
    solver_emcee result (planet params only; no RV columns)."""
    _nd = N_PLANETS * ndim_per
    _nw = max(2 * _nd + 2, solver_emcee.NWALKERS)
    _p0 = np.asarray(initial_joint[:_nd], dtype=float) \
        + solver_emcee.EPSILON * np.random.randn(_nw, _nd)
    for _pj in range(N_PLANETS):
        for _lj, _gi in enumerate(FIT_IDX):
            _c = _pj * ndim_per + _lj
            _p0[:, _c] = np.clip(_p0[:, _c], PARAM_LO[_gi], PARAM_HI[_gi])
    if 1 in FIT_IDX:
        _ec = FIT_IDX.index(1)
        for _pj in range(N_PLANETS):
            _c = _pj * ndim_per + _ec
            _p0[:, _c] = np.clip(_p0[:, _c], EPSILON, E_MAX)
    print(f"\n{'='*66}")
    print(f"  Emcee joint pre-refinement ({purpose}): {_nd} parameters, {_nw} walkers")
    print(f"{'='*66}")
    # Astrometry-only likelihood / prior: this vector has no K, gamma, jitter,
    # M_star or parallax columns, which the RV-aware joint functions index.
    _r = solver_emcee.run(
        log_likelihood_astro, log_prior_astro, prior_transform_multi,
        args=(along_scan_obs, sigma_total), p0=_p0, ndim=_nd,
    )
    solver_emcee.print_diagnostics(_r, active_names_joint[:_nd])
    return _r


# --- dynesty search box -----------------------------------------------------
# A single nested run over the full log-uniform-period prior cannot find a
# strong, narrow peak, so dynesty samples a flat per-parameter box. Two ways to
# place that box:
#   'emcee'       -- median +/- DYNESTY_BOX_NSIG*sigma of a full emcee joint
#                    fit (tight, unbiased; costs one ~1-min emcee fit).
#   'periodogram' -- centred on the detect.py screen's per-signal estimates
#                    (no emcee fit needed): tight on P (+/-DYNESTY_BOX_P_FRAC),
#                    one period wide on T0, +/-DYNESTY_BOX_TI_PAD (or
#                    3x|value|) mas on A,B,F,G, full prior on e.  Falls back
#                    to 'emcee' if the screen is off or under-detected.
DYNESTY_BOX_SOURCE  = os.environ.get('EXOPLANETLAB_DYNESTY_BOX_SOURCE', 'emcee')
DYNESTY_BOX_NSIG    = float(os.environ.get('EXOPLANETLAB_DYNESTY_BOX_NSIG', '10'))
DYNESTY_BOX_P_FRAC  = float(os.environ.get('EXOPLANETLAB_DYNESTY_BOX_P_FRAC', '0.05'))
DYNESTY_BOX_TI_PAD  = float(os.environ.get('EXOPLANETLAB_DYNESTY_BOX_TI_PAD', '6.0'))

_joint_solver = solver   # overridden below for the nessai/emcee joint hybrid

# ── RJMCMC path ────────────────────────────────────────────────────────────────
if SOLVER in ('rjmcmc', 'rjflow'):
    _label = 'RJMCMC' if SOLVER == 'rjmcmc' else 'RJFlow RJMCMC'
    _stage('joint fit (RJ, incl. emcee pre-refinement)')

    # Emcee joint pre-refinement (see RJ_WARM_START_USE_EMCEE). These shadow
    # variables default to the raw peel results and are used by the RJ solver
    # only; peel_best / peel_results stay as the peel left them.
    _rj_warm_best    = peel_best
    _rj_warm_results = peel_results

    # The pre-refinement and the joint functions are bound to N_PLANETS sources,
    # but the survey probe can append an extra above-threshold peel (common at
    # a 2:1 period ratio). The warm start then uses the N_PLANETS strongest
    # peels; the RJ solver still gets the full list and infers N. If the peel
    # found fewer than N_PLANETS, skip.
    _abfg_local = [FIT_IDX.index(_g) for _g in (3, 4, 5, 6) if _g in FIT_IDX]
    _p_local_pk = FIT_IDX.index(0) if 0 in FIT_IDX else 0

    def _peel_amp(pb):
        return float(np.sqrt(sum(float(pb[i]) ** 2 for i in _abfg_local))) \
            if _abfg_local else 0.0

    _ws_src_best = peel_best
    _run_warm = RJ_WARM_START_USE_EMCEE and len(peel_best) >= N_PLANETS
    if _run_warm and len(peel_best) > N_PLANETS:
        _rank = sorted(range(len(peel_best)), key=lambda i: _peel_amp(peel_best[i]),
                       reverse=True)[:N_PLANETS]
        _rank = sorted(_rank, key=lambda i: float(peel_best[i][_p_local_pk]))
        _ws_src_best = [peel_best[i] for i in _rank]
        # The displaced (weaker) peel(s) are survey-probe artifacts; keep the
        # strongest as an RJ birth hint rather than an in-model seed.
        _dropped = [i for i in range(len(peel_best)) if i not in _rank]
        if _dropped and _probe_peel_result is None:
            _probe_peel_result = peel_results[max(_dropped, key=lambda i: _peel_amp(peel_best[i]))]
        print(f"\n  [{_label}] peel produced {len(peel_best)} sources "
              f"(N_PLANETS={N_PLANETS}); warm-starting on the {N_PLANETS} "
              f"strongest (periods "
              f"{', '.join(f'{float(peel_best[i][_p_local_pk]):.0f} d' for i in _rank)}).")
    elif RJ_WARM_START_USE_EMCEE and len(peel_best) < N_PLANETS:
        print(f"\n  [{_label}] peel produced only {len(peel_best)} sources "
              f"(< N_PLANETS={N_PLANETS}); skipping the emcee joint warm start.")

    if _run_warm:
        _ndim_full_joint = N_PLANETS * ndim_per
        _nwalkers_pre = max(2 * _ndim_full_joint + 2, solver_emcee.NWALKERS)
        _ws_center = np.concatenate([np.asarray(b, dtype=float) for b in _ws_src_best])
        _p0_pre = _ws_center + solver_emcee.EPSILON * np.random.randn(
            _nwalkers_pre, _ndim_full_joint)
        for _pj in range(N_PLANETS):
            for _lj, _gi in enumerate(FIT_IDX):
                _col = _pj * ndim_per + _lj
                _p0_pre[:, _col] = np.clip(_p0_pre[:, _col], PARAM_LO[_gi], PARAM_HI[_gi])
        if 1 in FIT_IDX:
            _e_col = FIT_IDX.index(1)
            for _pj in range(N_PLANETS):
                _col = _pj * ndim_per + _e_col
                _p0_pre[:, _col] = np.clip(_p0_pre[:, _col], EPSILON, E_MAX)

        print(f"\n{'='*66}")
        print(f"  Emcee joint pre-refinement (warm start for {_label}): "
              f"{_ndim_full_joint} parameters, {_nwalkers_pre} walkers")
        print(f"{'='*66}")
        _pre_result = solver_emcee.run(
            log_likelihood_multi, log_prior_multi, prior_transform_multi,
            args=(along_scan_obs, sigma_total),
            p0=_p0_pre, ndim=_ndim_full_joint,
        )
        solver_emcee.print_diagnostics(_pre_result, active_names_joint)

        _rj_warm_best = [_pre_result.best_params[_pj * ndim_per:(_pj + 1) * ndim_per]
                          for _pj in range(N_PLANETS)]
        # Subsample (see _RJ_WARM_START_MAX_SAMPLES).
        # Drop the samples of walkers stuck in a secondary maximum first: a
        # few of them dominate the covariance, the KDE and the flow built
        # from the samples, and no proposal made from those is accepted. A
        # sample of a d-dimensional Gaussian posterior lies more than
        # chi2_d(1e-6)/2 below the maximum once in a million draws.
        from scipy.stats import chi2 as _chi2
        _lp = np.asarray(_pre_result.log_probs, dtype=float)
        _cut = _lp.max() - 0.5 * _chi2.isf(1.0e-6, _ndim_full_joint)
        _pre_ok = np.flatnonzero(_lp >= _cut)
        print(f"  Warm start: {len(_lp) - len(_pre_ok)} of {len(_lp)} samples lie more "
              f"than {_lp.max() - _cut:.1f} below the best ln-probability "
              f"and are not used.")
        _n_pre_samples = len(_pre_ok)
        if _n_pre_samples > _RJ_WARM_START_MAX_SAMPLES:
            _pre_idx = np.random.choice(
                _pre_ok, _RJ_WARM_START_MAX_SAMPLES, replace=False)
        else:
            _pre_idx = _pre_ok
        _rj_warm_results = [
            SimpleNamespace(
                samples=_pre_result.samples[_pre_idx, _pj * ndim_per:(_pj + 1) * ndim_per])
            for _pj in range(N_PLANETS)
        ]

    if RJ_SKIP_IF_N_KNOWN and len(_rj_warm_best) == N_PLANETS:
        print(f"\n{'='*66}")
        print(f"  [{_label}] RJ_SKIP_IF_N_KNOWN=True — "
              f"using peel posteriors for N={N_PLANETS}")
        print(f"{'='*66}")
        _min_n = min(len(pr.samples) for pr in _rj_warm_results)
        _joint_samples = np.hstack([pr.samples[:_min_n] for pr in _rj_warm_results])
        _n_dist = np.zeros(N_PLANETS + 2, dtype=int)
        _n_dist[N_PLANETS] = _min_n
        _peel_logz = (sum(getattr(pr, 'logz', None) or 0 for pr in _rj_warm_results)
                      if all(getattr(pr, 'logz', None) is not None
                             for pr in _rj_warm_results)
                      else None)
        result_joint = SimpleNamespace(
            samples                = _joint_samples,
            log_probs              = None,
            best_params            = np.concatenate(_rj_warm_best),
            chain                  = None,
            chain_burn             = None,
            logprob_burn           = None,
            acceptance_fraction    = None,
            rj_acceptance_fraction = None,
            tau                    = None,
            tau_err                = None,
            nsteps_prod            = None,
            logz                   = _peel_logz,
            logzerr                = None,
            n_chain                = None,
            most_probable_n        = N_PLANETS,
            n_distribution         = _n_dist,
        )

    else:
        print(f"\n{'='*66}")
        print(f"  Joint {_label}: N_min=1, N_max={solver.N_MAX}, "
              f"{ndim_per} params/planet")
        print(f"{'='*66}")

        def _to_log_rj(v):
            v2 = v.copy()
            for j in _LUF_LOCAL_IDX:
                v2[j] = np.log(v[j])
            return v2

        _peel_best_rj = [_to_log_rj(pb) for pb in _rj_warm_best]

        def _samples_to_log_rj(_samps):
            s_log = _samps.copy()
            for j in _LUF_LOCAL_IDX:
                s_log[:, j] = np.log(np.abs(_samps[:, j]))
            return s_log

        _peel_results_rj = [SimpleNamespace(samples=_samples_to_log_rj(pr.samples))
                            for pr in _rj_warm_results]

        # Birth-proposal components: the detected-planet posteriors PLUS the
        # sub-threshold survey-probe peel (strongest unexplained residual peak),
        # so a birth can be proposed where an as-yet-undetected planet would
        # sit.  The probe peel is NOT added to _peel_results_rj -- it must not
        # pollute the in-model covariance / KDE / flow training.
        _birth_results_rj = list(_peel_results_rj)
        if ('_probe_peel_result' in globals() and _probe_peel_result is not None
                and getattr(_probe_peel_result, 'samples', None) is not None):
            _birth_results_rj.append(
                SimpleNamespace(samples=_samples_to_log_rj(_probe_peel_result.samples)))

        _param_lo_rj = [
            (np.log(PARAM_LO[gi]) if gi in (0,) else PARAM_LO[gi])
            for gi in FIT_IDX
        ]
        _param_hi_rj = [
            (np.log(PARAM_HI[gi]) if gi in (0,) else PARAM_HI[gi])
            for gi in FIT_IDX
        ]

        _extra_joint = dict(
            n_max           = solver.N_MAX,
            n_min           = 1,
            peel_best       = _peel_best_rj,
            peel_results    = _peel_results_rj if _peel_results_rj else None,
            birth_results   = _birth_results_rj if _birth_results_rj else None,
            param_lo        = _param_lo_rj,
            param_hi        = _param_hi_rj,
            log_uniform_idx = (),
            fit_idx         = FIT_IDX,
        )
        result_joint = solver.run(
            log_like_rj, None, None,
            args=(along_scan_obs, sigma_total),
            ndim=ndim_per,
            **_extra_joint,
        )

        # Log-uniform period columns come back in log space.  A dormant/unused
        # RJ leaf can carry an unconstrained placeholder there, so clamp the
        # exponent to the period prior before exp() -- outside [P_MIN, P_MAX] the
        # value is meaningless anyway -- to avoid a spurious exp overflow warning.
        _log_p_lo, _log_p_hi = np.log(P_MIN), np.log(P_MAX)
        n_inferred_tmp = result_joint.most_probable_n
        # best_params is a row of samples: copy it, or the conversion of the
        # samples below converts it too and it is then converted twice
        result_joint.best_params = np.array(result_joint.best_params, dtype=float)
        if result_joint.samples.size > 0:
            for _pi in range(n_inferred_tmp):
                for _j in _LUF_LOCAL_IDX:
                    _k = _pi * ndim_per + _j
                    result_joint.samples[:, _k] = np.exp(
                        np.clip(result_joint.samples[:, _k], _log_p_lo, _log_p_hi))
        bp_log  = result_joint.best_params
        bp_phys = bp_log.copy()
        for _pi in range(n_inferred_tmp):
            for _j in _LUF_LOCAL_IDX:
                _k = _pi * ndim_per + _j
                bp_phys[_k] = np.exp(np.clip(bp_log[_k], _log_p_lo, _log_p_hi))
        result_joint.best_params = bp_phys

# ── Fixed-N sampler paths ──────────────────────────────────────────────────────
else:
    # Hybrid: SOLVER='nessai' peels with nessai, but the joint stage runs as for
    # SOLVER='emcee' (see NESSAI_JOINT_USE_EMCEE).
    #
    # Forced on with RVs: nessai's own joint stage narrows its box from the
    # astrometry-only peels, and if the RVs prefer a period outside that box it
    # converges to a narrow, wrong answer without warning.
    # Not in nested-evidence mode: its box comes from the joint emcee fit with
    # the RVs, not from the astrometry-only peels.
    if SOLVER == 'nessai' and RV_ENABLED and not NESSAI_JOINT_USE_EMCEE and not NESTED_EVIDENCE:
        print("  NESSAI_JOINT_USE_EMCEE=0 overridden to 1: RV_ENABLED makes "
              "nessai's own astrometry-only joint search box a genuine risk "
              "of silently missing an RV-driven period degeneracy -- see "
              "the override comment in ExoPlanetLab.py.")
        NESSAI_JOINT_USE_EMCEE = True
    _nessai_joint_hybrid = (SOLVER == 'nessai' and NESSAI_JOINT_USE_EMCEE)
    _emcee_style_joint   = (SOLVER == 'emcee' or _nessai_joint_hybrid)
    # Parallel tempering for the joint fit. With RVs taken years after the Gaia
    # epochs the posterior is multimodal in phase (solver_ptemcee.py), which
    # the plain stretch move mixes slowly; EXOPLANETLAB_PT_NTEMPS=1 forces
    # plain emcee. Astrometry-only fits are tempered too (EXOPLANETLAB_PT_ASTRO,
    # default 1; 0 = plain emcee): weak signals, near-circular orbits and
    # periods near or beyond the baseline converge with it, in less total
    # sampling time despite the higher cost per step.
    _use_pt = (_emcee_style_joint and (RV_ENABLED or PT_ASTRO)
               and solver_ptemcee.NTEMPS > 1)
    _joint_solver = (solver_ptemcee if _use_pt
                     else solver_emcee if _emcee_style_joint else solver)

    p0_joint = None
    if SOLVER == 'emcee' or _nessai_joint_hybrid:
        nwalkers_joint = max(2 * ndim_total + 2, solver_emcee.NWALKERS)
        # (Relative astrometry: the same argument -- imaging epochs years after
        # the Gaia baseline turn a small period error into a large phase error.)
        if RV_ENABLED or RELAST_IDX:
            # Start the joint RV fit at the nearby posterior maximum, not at
            # the astrometry-only peel: a small period error is a large phase
            # error over a long RV baseline, and with free noise terms the
            # walkers can escape the misfit by inflating the jitters instead of
            # moving the period. A short optimisation, tried with both signs of
            # K and kept only if it improves, removes the problem.
            #
            # Only the orbit, K and the offsets gamma are optimised. The noise
            # terms and the prior-only M_star / parallax stay at their initial
            # values: the posterior density of a free noise term peaks at its
            # lower bound, a narrow spike away from the probability mass.
            try:
                from scipy.optimize import minimize as _minimize

                _k0 = N_PLANETS * ndim_per
                _free = np.ones(ndim_total, dtype=bool)
                if RV_ENABLED:
                    _jit0 = _k0 + N_PLANETS + N_RV_INST
                    _free[_jit0:_jit0 + N_RV_INST] = False          # RV jitters
                if RV_ENABLED and FIT_MSTAR_PARALLAX:
                    _free[_jit0 + N_RV_INST:_jit0 + N_RV_INST + 2] = False
                if ASTRO_JITTER:
                    _free[ASTRO_JIT_IDX] = False
                if RELAST_JITTER:
                    _free[RELAST_JIT_IDX] = False
                if ASTRO_NOISE == 'student':
                    _free[[NU_IDX, F_IDX] + ([NUH_IDX, FH_IDX] if HIPPARCOS else [])] = False
                if RV_COUPLED:
                    # K follows the orbit (with the sign of the start) instead
                    # of being optimised: its density is too narrow to search
                    _free[_k0] = False
                    initial_joint[_k0] = (np.where(initial_joint[_k0] < 0.0, -1.0, 1.0)
                                          * _rv_coupled_k(initial_joint)[0])

                def _full(xf, _base):
                    x = _base.copy()
                    x[_free] = xf
                    if RV_COUPLED:
                        try:
                            _kc = float(_rv_coupled_k(x)[0])
                        except Exception:
                            _kc = np.nan
                        if np.isfinite(_kc):
                            x[_k0] = (-1.0 if _base[_k0] < 0.0 else 1.0) * _kc
                    return x

                def _neg_lpost(x):
                    lp = log_prior_multi(x)
                    if not np.isfinite(lp):
                        return 1e30
                    val = -(lp + log_likelihood_multi(x, along_scan_obs, sigma_total))
                    return val if np.isfinite(val) else 1e30

                _best_x, _best_f = initial_joint.copy(), _neg_lpost(initial_joint)
                _f0 = _best_f
                _nfree = int(_free.sum())
                for _sign in ((1.0, -1.0) if RV_ENABLED else (1.0,)):
                    _base = initial_joint.copy()
                    if RV_ENABLED:
                        _base[_k0:_k0 + N_PLANETS] *= _sign
                    _x0 = _base[_free]
                    for _method in ('Nelder-Mead', 'Powell'):
                        _res = _minimize(lambda xf, _b=_base: _neg_lpost(_full(xf, _b)), _x0,
                                         method=_method,
                                         options=dict(maxiter=3000 * _nfree,
                                                      maxfev=3000 * _nfree,
                                                      xatol=1e-8, fatol=1e-6))
                        if np.isfinite(_res.fun) and _res.fun < _best_f:
                            _best_x, _best_f = _full(np.asarray(_res.x, dtype=float), _base), float(_res.fun)
                        _x0 = np.asarray(_res.x, dtype=float)   # chain the two methods
                if _best_f < _f0:
                    print(f"  Joint start optimised (RV / relative astrometry): "
                          f"ln(posterior) {-_f0:.1f} -> {-_best_f:.1f}")
                    for _j, _nm in enumerate(active_names_joint):
                        if abs(_best_x[_j] - initial_joint[_j]) > 1e-9 * max(1.0, abs(initial_joint[_j])):
                            print(f"    {_nm:16s} {initial_joint[_j]:14.6g} -> {_best_x[_j]:14.6g}")
                    initial_joint = _best_x
            except Exception as _ex:
                print(f"  WARNING: joint start optimisation skipped ({_ex!r}).")
        p0_joint = initial_joint + solver_emcee.EPSILON * np.random.randn(nwalkers_joint, ndim_total)
        for planet_j in range(N_PLANETS):
            for local_j, gi in enumerate(FIT_IDX):
                col = planet_j * ndim_per + local_j
                p0_joint[:, col] = np.clip(p0_joint[:, col], PARAM_LO[gi], PARAM_HI[gi])
        # Clip eccentricities (global index 1) for all planets
        if 1 in FIT_IDX:
            e_col = FIT_IDX.index(1)
            for planet_j in range(N_PLANETS):
                col = planet_j * ndim_per + e_col
                p0_joint[:, col] = np.clip(p0_joint[:, col], EPSILON, E_MAX)
        if RV_ENABLED:
            _gamma_start = N_PLANETS * ndim_per + N_PLANETS
            _jit_start   = _gamma_start + N_RV_INST
            for _j in range(N_PLANETS):
                _c = N_PLANETS * ndim_per + _j
                # Seed both RV branches (see K_MIN): flip the sign of K for
                # half the walkers, so that the ensemble starts in both wells.
                _flip = np.random.rand(nwalkers_joint) < 0.5
                p0_joint[_flip, _c] *= -1.0
                if RV_COUPLED:
                    # each walker's K on the coupling of its own orbit (the
                    # density in K is ~0.05-0.5% wide: a free start is far off)
                    p0_joint[:, _c] = np.where(p0_joint[:, _c] < 0.0, -1.0, 1.0) * _rv_coupled_k(p0_joint)
                p0_joint[:, _c] = np.clip(p0_joint[:, _c], K_MIN, K_MAX)
            for _j in range(N_RV_INST):
                p0_joint[:, _gamma_start + _j] = np.clip(
                    p0_joint[:, _gamma_start + _j], GAMMA_MIN, GAMMA_MAX)
                # Spread the jitter walkers by ~10% (multiplicatively; the
                # generic EPSILON spread is absolute, 1e-4 m/s, i.e. none at
                # all on a jitter of ~1000 m/s), strictly inside the bounds.
                p0_joint[:, _jit_start + _j] = np.clip(
                    initial_joint[_jit_start + _j]
                    * np.exp(0.1 * np.random.randn(nwalkers_joint)),
                    1.01 * JITTER_MIN, 0.99 * JITTER_MAX)
            if FIT_MSTAR_PARALLAX:
                _mstar_idx = _jit_start + N_RV_INST
                # Gaussian prior is unbounded but unphysical below 0.
                p0_joint[:, _mstar_idx]     = np.clip(p0_joint[:, _mstar_idx], 1e-6, None)
                p0_joint[:, _mstar_idx + 1] = np.clip(p0_joint[:, _mstar_idx + 1], 1e-6, None)
        if ASTRO_JITTER:
            # Spread the jitter walkers by +/-10% (the generic EPSILON spread is
            # absolute and negligible here), inside the bounds.
            p0_joint[:, ASTRO_JIT_IDX] = np.clip(
                initial_joint[ASTRO_JIT_IDX] * (1.0 + 0.1 * np.random.randn(nwalkers_joint)),
                1.01 * ASTRO_JITTER_MIN, 0.99 * ASTRO_JITTER_MAX)
        for _ix in RELAST_IDX:
            p0_joint[:, _ix] = np.clip(
                initial_joint[_ix] * np.exp(0.01 * np.random.randn(nwalkers_joint)),
                1.001 * RELAST_S_MIN, 0.999 * RELAST_S_MAX)
        if RELAST_JITTER:
            p0_joint[:, RELAST_JIT_IDX] = np.clip(
                initial_joint[RELAST_JIT_IDX] * np.exp(0.3 * np.random.randn(nwalkers_joint)),
                1.01 * RELAST_JIT_MIN, 0.99 * RELAST_JIT_MAX)
        if ASTRO_NOISE == 'student':
            for _ix, _lo, _hi in ((NU_IDX, NU_MIN, NU_MAX), (F_IDX, F_MIN, F_MAX)) + (
                    ((NUH_IDX, NU_MIN, NU_MAX), (FH_IDX, F_MIN, F_MAX)) if HIPPARCOS else ()):
                p0_joint[:, _ix] = np.clip(
                    initial_joint[_ix] * (1.0 + 0.1 * np.random.randn(nwalkers_joint)),
                    1.01 * _lo, 0.99 * _hi)
        print(f"\n{'='*66}")
        _joint_label = ('emcee (nessai-peel hybrid)' if _nessai_joint_hybrid
                         else 'emcee')
        print(f"  Joint {_joint_label}: {ndim_total} parameters, {nwalkers_joint} walkers")
        print(f"{'='*66}")
    else:
        print(f"\n{'='*66}")
        print(f"  Joint {SOLVER}: {ndim_total} parameters")
        print(f"{'='*66}")

    _extra_joint = {}
    if not RV_ENABLED and (SOLVER == 'emcee' or _nessai_joint_hybrid):
        # Astrometry-only joint fit: a generous upper limit with the adaptive
        # stop (100 tau), as for the RV fit below. Weak signals have tau of
        # several hundred to a thousand steps; fast-mixing runs stop early.
        _extra_joint = dict(
            nsteps_burn=int(os.environ.get('EXOPLANETLAB_JOINT_BURN_ASTRO', '5000')),
            nsteps_prod=int(os.environ.get('EXOPLANETLAB_JOINT_PROD_ASTRO', '200000')))
        if _use_pt:
            print(f"  Using the parallel-tempered joint fit for astrometry only "
                  f"({solver_ptemcee.NTEMPS} temperatures; EXOPLANETLAB_PT_ASTRO=1).")
    if RV_ENABLED and (SOLVER == 'emcee' or _nessai_joint_hybrid):
        # With RVs the posterior can be multimodal in (P, T0) (see _use_pt
        # above), and some noise terms mix slowly. With adaptive stopping
        # (solver_emcee.ADAPTIVE_STOP) nsteps_prod is an upper limit, so it is
        # generous: fast-mixing runs stop early, slow ones (Gaia BH3 + RV, tau
        # of thousands) use it. EXOPLANETLAB_JOINT_BURN / _JOINT_PROD override.
        _extra_joint = dict(
            nsteps_burn=int(os.environ.get('EXOPLANETLAB_JOINT_BURN', '10000')),
            nsteps_prod=int(os.environ.get('EXOPLANETLAB_JOINT_PROD', '200000')))
        if _use_pt:
            print(f"  Using parallel-tempered joint fit "
                  f"({solver_ptemcee.NTEMPS} temperatures) -- see "
                  f"solver_ptemcee.py. Set EXOPLANETLAB_PT_NTEMPS=1 to "
                  f"force plain emcee instead.")
    if EVIDENCE:
        if _use_pt:
            _extra_joint['evidence'] = True
        elif not NESTED_EVIDENCE:
            print("WARNING: EXOPLANETLAB_EVIDENCE=1 needs the tempered joint fit "
                  "(emcee or the nessai hybrid with tempering); no evidence computed.")
    if SOLVER == 'dynesty':
        # Single full-dimensional nested job (solver_dynesty.py).
        _extra_joint = dict(nlive=solver.NLIVE)

    if SOLVER == 'nessai' and not _nessai_joint_hybrid:
        # The tight-prior warm start (below) keeps the search volume small, so
        # 200 live points suffice. stopping_criterion / max_iteration: see
        # STOPPING_JOINT / JOINT_MAX_ITERATION.
        _extra_joint = dict(output=os.path.join(outdir, 'nessai_flow'),
                            nlive=200,
                            stopping_criterion=STOPPING_JOINT,
                            max_iteration=JOINT_MAX_ITERATION)

    if SOLVER == 'nessai' and not _nessai_joint_hybrid and NESTED_EVIDENCE:
        # the evidence needs every live point drawn from the box below, and no
        # iteration cap (a capped run's ln Z is not the integral)
        _extra_joint = dict(output=os.path.join(outdir, 'nessai_flow'), nlive=500,
                            stopping_criterion=0.1, max_iteration=None)
    if SOLVER == 'nessai' and not _nessai_joint_hybrid and not NESTED_EVIDENCE:
        from tight_prior import _tight_transform, _tight_log_prior_factory, _tight_stats
        _ndim_ti   = N_PLANETS * ndim_per   # TI params only; RV params use flat prior
        _samples_l = [peel_results[k // ndim_per].samples[:, k % ndim_per]
                      for k in range(_ndim_ti)]
        _lo_l = [float(_PRIOR_TRANSFORMS_MULTI[k](1e-9))    for k in range(_ndim_ti)]
        _hi_l = [float(_PRIOR_TRANSFORMS_MULTI[k](1 - 1e-9)) for k in range(_ndim_ti)]

        print("\nWarm-start prior widths (peel σ as % of full prior range):")
        for k in range(_ndim_ti):
            mu, sig, lo, hi, a, b = _tight_stats(_samples_l[k], _lo_l[k], _hi_l[k])
            frac = sig / (_hi_l[k] - _lo_l[k]) * 100
            name = active_names_joint[k]
            flag = " ← wide" if frac > 15 else ""
            print(f"  {name:14s}: σ = {sig:.3g}  ({frac:.1f}% of prior range){flag}")

        _tight_txs = [_tight_transform(s, lo, hi)
                      for s, lo, hi in zip(_samples_l, _lo_l, _hi_l)]
        if RV_ENABLED:
            _tight_txs.extend(_RV_TRANSFORMS)
            _tight_txs.extend(_MSTAR_PLX_TRANSFORMS)
        _pt_joint = lambda u, _t=_tight_txs: np.array([t(u[i]) for i, t in enumerate(_t)])
        _lp_ti    = _tight_log_prior_factory(_samples_l, _lo_l, _hi_l)
        if RV_ENABLED:
            def _lp_joint(params, _f=_lp_ti):
                lp = _f(params)
                if not np.isfinite(lp):
                    return lp
                _k_start     = N_PLANETS * ndim_per
                _gamma_start = _k_start + N_PLANETS
                _jit_start   = _gamma_start + N_RV_INST
                _mstar_idx   = _jit_start + N_RV_INST
                k_p     = params[_k_start:_gamma_start]
                gammas  = params[_gamma_start:_jit_start]
                jitters = np.asarray(params[_jit_start:_jit_start + N_RV_INST])
                if np.any(np.asarray(k_p) < K_MIN) or np.any(np.asarray(k_p) > K_MAX):
                    return -np.inf
                if np.any(np.asarray(gammas) < GAMMA_MIN) or np.any(np.asarray(gammas) > GAMMA_MAX):
                    return -np.inf
                if np.any(jitters < JITTER_MIN) or np.any(jitters > JITTER_MAX):
                    return -np.inf
                lp = lp - np.sum(np.log(jitters))
                if FIT_MSTAR_PARALLAX:
                    m_star, plx = params[_mstar_idx], params[_mstar_idx + 1]
                    if m_star <= 0.0 or plx <= 0.0:
                        return -np.inf
                    lp += (-0.5 * ((m_star - M_STAR_MSUN) / M_STAR_SIGMA) ** 2
                           -0.5 * ((plx - PARALLAX_MAS) / PARALLAX_SIGMA) ** 2)
                return lp
        else:
            _lp_joint = _lp_ti
        if ASTRO_JITTER or ASTRO_NOISE == 'student' or RELAST_ENABLED:
            # Jitter (or nu, f) and the relative-astrometry scales last:
            # flat-in-log transforms and prior.
            _tight_txs.extend(_ASTRO_JIT_TRANSFORMS + _ASTRO_NOISE_TRANSFORMS
                              + _RELAST_TRANSFORMS)
            _pt_joint = lambda u, _t=_tight_txs: np.array([t(u[i]) for i, t in enumerate(_t)])
            _lp_no_jit = _lp_joint
            def _lp_joint(params, _f=_lp_no_jit):
                lp = _f(params)
                return lp + _log_prior_astro_jitter(params) if np.isfinite(lp) else lp
    elif (SOLVER == 'dynesty' or NESTED_EVIDENCE) and len(peel_best) == N_PLANETS:
        # dynesty samples a flat per-parameter box: a single nested run over the
        # full log-uniform-period prior cannot find a strong, narrow peak, and a
        # box from the raw peels is contaminated at a 2:1 period ratio. The box
        # is placed from an unbiased source (emcee joint fit or the screen),
        # wide enough that its flat prior does not narrow the posterior. See
        # DYNESTY_BOX_SOURCE.
        _ndim_ti = N_PLANETS * ndim_per
        _lo_l = np.array([float(_PRIOR_TRANSFORMS_MULTI[k](1e-9))
                          for k in range(_ndim_ti)])
        _hi_l = np.array([float(_PRIOR_TRANSFORMS_MULTI[k](1 - 1e-9))
                          for k in range(_ndim_ti)])

        _box_lo = _box_hi = None
        _box_samples = None
        _box_desc = ''

        if DYNESTY_BOX_SOURCE == 'periodogram':
            _scr = (list(_detect_result.params)
                    if ('_detect_result' in globals()
                        and _detect_result.n_detected >= N_PLANETS) else None)
            if _scr is None:
                print("  dynesty: periodogram box unavailable (screen off or "
                      "under-detected) -- using the emcee-joint box instead.")
            else:
                _scr = sorted(_scr, key=lambda p: float(p[0]))[:N_PLANETS]
                _tref = float(t_obs.min())
                _blo, _bhi = _lo_l.copy(), _hi_l.copy()
                for _pi in range(N_PLANETS):
                    _sp = np.asarray(_scr[_pi], dtype=float)  # [P,e,T0,A,B,F,G]
                    for _lj, _gi in enumerate(FIT_IDX):
                        _k = _pi * ndim_per + _lj
                        if _gi == 0:                                  # P: +/- frac
                            _blo[_k] = _sp[0] * (1.0 - DYNESTY_BOX_P_FRAC)
                            _bhi[_k] = _sp[0] * (1.0 + DYNESTY_BOX_P_FRAC)
                        elif _gi == 1:                                # e: full prior
                            pass
                        elif _gi == 2:                                # T0: ~one period, centred on the (de-aliased) screen T0
                            _P = _sp[0]
                            _t0f = _tref + (float(_sp[2]) - _tref) % _P
                            _blo[_k], _bhi[_k] = _t0f - 0.6 * _P, _t0f + 0.6 * _P
                        else:                                        # A,B,F,G: +/- max(PAD, 3*|val|)
                            _pad = max(DYNESTY_BOX_TI_PAD, 3.0 * abs(float(_sp[_gi])))
                            _blo[_k] = float(_sp[_gi]) - _pad
                            _bhi[_k] = float(_sp[_gi]) + _pad
                _box_lo = np.maximum(_blo, _lo_l)
                _box_hi = np.minimum(_bhi, _hi_l)
                _box_desc = (f"periodogram screen (P +/-{DYNESTY_BOX_P_FRAC:.0%}, "
                             f"T0 one period, A..G +/-{DYNESTY_BOX_TI_PAD:g} mas)")

        if DYNESTY_BOX_SOURCE.startswith('file:'):
            # posterior samples of an earlier run of the same model (its
            # posterior_samples.npz): for a model whose short emcee fit is not
            # reliable, e.g. an unconstrained second orbit (NESTED_EVIDENCE)
            _bf = np.load(DYNESTY_BOX_SOURCE[5:], allow_pickle=True)
            if list(_bf['names'][:_ndim_ti]) != list(active_names_joint[:_ndim_ti]):
                sys.exit(f"\nWARNING: {DYNESTY_BOX_SOURCE[5:]} holds other parameters "
                         f"({list(_bf['names'][:_ndim_ti])}).\n")
            _box_samples = np.asarray(_bf['samples'][:, :_ndim_ti], dtype=float)
            _q   = np.percentile(_box_samples, [16, 50, 84], axis=0)
            _sig = np.maximum(0.5 * (_q[2] - _q[0]), 1e-12)
            _wid = np.maximum(DYNESTY_BOX_NSIG * _sig, 1e-4 * (_hi_l - _lo_l))
            _box_lo = np.maximum(_q[1] - _wid, _lo_l)
            _box_hi = np.minimum(_q[1] + _wid, _hi_l)
            _box_desc = (f"posterior of {os.path.basename(os.path.dirname(DYNESTY_BOX_SOURCE[5:]))} "
                         f"(+/-{DYNESTY_BOX_NSIG:g} sigma)")

        if _box_lo is None:                                          # 'emcee' or fallback
            _pre_result = _emcee_joint_prerefine('search-box source for dynesty')
            _box_samples = np.asarray(_pre_result.samples[:, :_ndim_ti], dtype=float)
            _q   = np.percentile(_box_samples, [16, 50, 84], axis=0)
            _med = _q[1]
            _sig = np.maximum(0.5 * (_q[2] - _q[0]), 1e-12)
            _wid = np.maximum(DYNESTY_BOX_NSIG * _sig, 1e-4 * (_hi_l - _lo_l))
            _box_lo = np.maximum(_med - _wid, _lo_l)
            _box_hi = np.minimum(_med + _wid, _hi_l)
            _box_desc = f"emcee joint fit (+/-{DYNESTY_BOX_NSIG:g} sigma)"

        # Evidence mode: T0 as an orbital phase. The likelihood is periodic in
        # T0 with period P, so the posterior has one copy per period of the T0
        # prior range ((T0_HI - T0_LO) / P ~ 200 for P = 84 d), and a box around
        # one copy misses all the others. Sampling u = (T0 - anchor) / P mod 1
        # instead integrates over all of them: the uniform T0 prior over the
        # range is, for a periodic likelihood, a uniform prior on u with density
        # 1 (to a fraction ~P / (T0_HI - T0_LO) of one period at the range ends).
        _t0_phase = {}                      # T0 column -> (P column, anchor)
        if NESTED_EVIDENCE and _box_samples is not None and 0 in FIT_IDX and 2 in FIT_IDX:
            _tref = float(t_obs.min())
            for _pi in range(N_PLANETS):
                _kP = _pi * ndim_per + FIT_IDX.index(0)
                _kT = _pi * ndim_per + FIT_IDX.index(2)
                _Ps, _Ts = _box_samples[:, _kP], _box_samples[:, _kT]
                _ph = ((_Ts - _tref) / _Ps) % 1.0
                _phm = float(np.angle(np.mean(np.exp(2j * np.pi * _ph))) / (2 * np.pi)) % 1.0
                _anc = _tref + (_phm - 0.5) * float(np.median(_Ps))
                _u = ((_Ts - _anc) / _Ps) % 1.0
                _uq = np.percentile(_u, [16, 50, 84])
                _uw = DYNESTY_BOX_NSIG * 0.5 * (_uq[2] - _uq[0])
                _box_lo[_kT], _box_hi[_kT] = max(0.0, _uq[1] - _uw), min(1.0, _uq[1] + _uw)
                _t0_phase[_kT] = (_kP, _anc)
        _T0_PHASE = dict(_t0_phase)
        print(f"\n{SOLVER.capitalize()} search box (flat), from the {_box_desc}:"
              + ("  (T0 as the phase (T0 - anchor) / P)" if _t0_phase else ""))
        for k in range(_ndim_ti):
            frac = ((_box_hi[k] - _box_lo[k]) if k in _t0_phase else
                    (_box_hi[k] - _box_lo[k]) / (_hi_l[k] - _lo_l[k])) * 100
            print(f"  {active_names_joint[k] + (' (phase)' if k in _t0_phase else ''):14s}: "
                  f"[{_box_lo[k]:.5g}, {_box_hi[k]:.5g}]  ({frac:.2f}% of prior range)")
        # The flat box narrows the search over the orbital elements only. Every
        # other parameter keeps its real prior through the unit-hypercube
        # transforms defined above: uniform K and gamma, log-uniform jitter,
        # Gaussian M_star and parallax. (dynesty samples through prior_transform
        # only and never evaluates a log-prior, so a density term could not
        # supply them.)
        _box_span = _box_hi - _box_lo
        def _pt_joint(u, _lo=_box_lo, _sp=_box_span, _n=_ndim_ti):
            u = np.asarray(u, dtype=float)
            out = np.empty(ndim_total)
            out[:_n] = _lo + u[:_n] * _sp
            for _k in range(_n, ndim_total):
                out[_k] = _ALL_TRANSFORMS[_k](u[_k])
            return out
        _lp_joint = log_prior_multi
        if NESTED_EVIDENCE:
            # the integrand is L x prior(orbits) x V_box over the flat box (see
            # NESTED_EVIDENCE); nessai samples by its log_prior: flat over the
            # box, the noise parameters' log-uniform priors
            # (T0 sampled as a phase: its prior density is 1, and the T0 term of
            # log_prior_astro is only the range check)
            _LOGNORM_ORBITS = N_PLANETS * (
                -np.log(np.log(P_MAX / P_MIN)) - np.log(1.0 - E_MIN)
                - (0.0 if _t0_phase else np.log(T0_HI - T0_LO))
                - (_iso_log_norm() if ORIENT_PRIOR == 'isotropic' else 4.0 * np.log(2.0 * TI_MAX))
            ) - _sep_log_norm()
            _LN_VBOX = float(np.sum(np.log(_box_span)))

            def _ll_nested(p, *a, _lv=_LN_VBOX):
                # the samplers carry the T0 phase (both samplers see the same
                # coordinates: nessai works in them directly)
                p = _phase_to_t0(p)
                lp = _log_prior_orbits_normalised(p)
                if not np.isfinite(lp):
                    return -1e300
                return log_likelihood_multi(p, *a) + lp + _lv
            def _lp_joint(p):
                # nessai samples by log_prior: flat over the orbital box, the
                # real priors of the RV and noise parameters (log_prior_multi
                # without its orbital part); dynesty uses its transforms
                q = _phase_to_t0(p)
                a = log_prior_astro(q)
                return log_prior_multi(q) - a if np.isfinite(a) else -np.inf
            _JOINT_LL = _ll_nested
            print(f"  Evidence on the normalised prior (NESTED_EVIDENCE): integrand L x "
                  f"prior(orbits) x V_box, ln V_box = {_LN_VBOX:.3f}, ln prior normalisation "
                  f"{_LOGNORM_ORBITS:.3f} ({ORIENT_PRIOR} orientation"
                  + (f", ordering / separation {_sep_log_norm():.4f}" if N_PLANETS > 1 else "")
                  + ")")
    else:
        if SOLVER == 'dynesty':
            print("  WARNING: peel found < N_PLANETS candidates; dynesty falls "
                  "back to the full prior and will likely not converge.")
        _pt_joint = prior_transform_multi
        _lp_joint = log_prior_multi

    # Vectorised likelihood for the emcee-style joint fit (see VECTORIZE). Used
    # only if it reproduces the scalar posterior on the starting walkers
    # (plus perturbed copies) to 1e-8 relative -- otherwise warn and fall back.
    if (VECTORIZE and p0_joint is not None and _lp_joint is log_prior_multi
            and _joint_solver in (solver_emcee, solver_ptemcee)):
        try:
            _Xc = np.vstack([p0_joint, p0_joint * (1.0 + 1e-3 * np.random.randn(*p0_joint.shape))])
            _sc = np.array([log_prior_multi(x) + (log_likelihood_multi(x, along_scan_obs, sigma_total)
                                                  if np.isfinite(log_prior_multi(x)) else 0.0)
                            for x in _Xc])
            _vc = log_posterior_multi_vec(_Xc, along_scan_obs, sigma_total)
            _fin = np.isfinite(_sc)
            _same = (np.array_equal(_fin, np.isfinite(_vc))
                     and np.allclose(_vc[_fin], _sc[_fin], rtol=1e-8, atol=1e-6))
            if _same:
                if _joint_solver is solver_emcee:
                    _extra_joint['log_posterior_vec'] = log_posterior_multi_vec
                else:
                    _extra_joint['log_likelihood_vec'] = log_likelihood_multi_vec
                    _extra_joint['log_prior_vec'] = log_prior_multi_vec
                print(f"  Vectorised likelihood verified against the scalar one on "
                      f"{len(_Xc)} walkers (max |diff| "
                      f"{np.max(np.abs(_vc[_fin] - _sc[_fin])) if _fin.any() else 0:.1e}).")
            else:
                print("  WARNING: vectorised likelihood disagrees with the scalar "
                      "one on the starting walkers -- using the scalar path.")
        except Exception as _ex:
            print(f"  WARNING: vectorised likelihood check failed ({_ex!r}) -- "
                  f"using the scalar path.")

    if (LOG_JITTER and JITTER_IDX and p0_joint is not None
            and _joint_solver in (solver_emcee, solver_ptemcee)):
        _extra_joint['log_idx'] = JITTER_IDX   # see LOG_JITTER

    _stage(f'joint fit sampling ({_joint_solver.__name__})')
    if FIXED_IDX:
        # Sample only the free parameters (see RV_JITTER_MIN_N); the result is
        # expanded back to the full vector below, so nothing downstream changes.
        _LP_JOINT_FULL, _PT_JOINT_FULL = _lp_joint, _pt_joint
        _swap = {'log_posterior_vec': _lpost_vec_free, 'log_likelihood_vec': _ll_vec_free,
                 'log_prior_vec': _lp_vec_free}
        _extra_red = {k: _swap.get(k, v) for k, v in _extra_joint.items()}
        if 'log_idx' in _extra_red:
            _extra_red['log_idx'] = [_FREE_IDX.index(k) for k in _extra_red['log_idx']
                                     if k in _FREE_IDX]
        result_joint = _joint_solver.run(
            _ll_free, _lp_free, _pt_free,
            args=(along_scan_obs, sigma_total),
            p0=None if p0_joint is None else np.asarray(p0_joint)[:, _FREE_IDX],
            ndim=len(_FREE_IDX),
            **_extra_red,
        )
        result_joint = _expand_result(result_joint)
    else:
        result_joint = _joint_solver.run(
            _ll_nested if (NESTED_EVIDENCE and '_ll_nested' in globals()) else log_likelihood_multi,
            _lp_joint,
            _pt_joint,
            args=(along_scan_obs, sigma_total),
            p0=p0_joint,
            ndim=ndim_total,
            **_extra_joint,
        )

if NESTED_EVIDENCE and _T0_PHASE:
    # the samplers carried the T0 phase: back to T0
    for _fld in ('samples', 'best_params', 'weighted_samples'):
        if getattr(result_joint, _fld, None) is not None:
            setattr(result_joint, _fld, _phase_to_t0(getattr(result_joint, _fld)))

_joint_solver.print_diagnostics(result_joint, active_names_joint)


def _log_evidence_rv0():
    """ln Z of the RVs without a companion (0 without RVs): per instrument a
    constant gamma (uniform prior, integrated analytically -- its bounds are
    far outside the data) and a jitter (log-uniform prior, on a grid; fixed
    for the instruments of FIXED_JITTER_INST)."""
    if not RV_ENABLED:
        return 0.0
    if RV_TREND:
        return None                 # not implemented (the slope couples the instruments)
    from scipy.special import logsumexp
    tot = 0.0
    for k in range(N_RV_INST):
        m = np.asarray(rv_inst_idx) == k
        y, s2 = np.asarray(rv_obs, float)[m], np.asarray(rv_sigma, float)[m] ** 2
        fixed = k in FIXED_JITTER_INST
        lj = (np.array([np.log(FIXED_VAL)]) if fixed
              else np.linspace(np.log(JITTER_MIN), np.log(JITTER_MAX), 4000))
        v = s2[None, :] + np.exp(2.0 * lj)[:, None]
        W = np.sum(1.0 / v, axis=1)
        g = np.sum(y[None, :] / v, axis=1) / W
        ll = (-0.5 * np.sum((y[None, :] - g[:, None]) ** 2 / v + np.log(v), axis=1)
              + 0.5 * np.log(2.0 * np.pi / W) - np.log(GAMMA_MAX - GAMMA_MIN))
        if fixed:
            tot += float(ll[0])
        else:
            w = np.ones(len(lj)); w[[0, -1]] = 0.5
            tot += float(logsumexp(ll + np.log(w * (lj[1] - lj[0]))) - np.log(lj[-1] - lj[0]))
    return tot


def _log_evidence_n0(model=None):
    """ln Z without a companion, with the same data, noise model and likelihood
    normalisation as the joint fit: the single star (N = 0), or with
    model = 'accel7' / 'accel9' the acceleration models (accel.py).
    Only the noise parameters remain and are integrated directly over their
    normalised priors -- nu and f (Student-t, 2-D), the jitter (Gaussian, 1-D),
    or nothing (Gaussian without jitter); with RVs, plus _log_evidence_rv0."""
    from scipy.special import logsumexp
    if RELAST_ENABLED or HIPPARCOS:
        return None
    if model is not None and _PROJ is None:
        return None
    if model is not None:
        import accel
        _ssn0 = list(_dr4_epoch['proj_param_names'])
        _cols = (np.asarray(t_obs, float) / 365.25, _ssn0.index('pmra'), _ssn0.index('pmdec'))
    rv0 = _log_evidence_rv0()
    if rv0 is None:
        return None
    if ASTRO_NOISE == 'student':
        # adaptive grid in (ln nu, ln f) (robust.StudentTSingleStar.log_evidence)
        if model is not None:
            return rv0 + accel.log_evidence(ROBUST_BASE, *_cols, model,
                                            (NU_MIN, NU_MAX), (F_MIN, F_MAX), psi=scan_angles)
        return rv0 + ROBUST_BASE.log_evidence(np.zeros(N_epochs), (NU_MIN, NU_MAX),
                                              (F_MIN, F_MAX))
    r = np.asarray(along_scan_obs, float)
    s2 = np.asarray(sigma_total, float) ** 2
    if model is not None:
        _ga = accel.GaussAccel(accel.projected_columns(
            _dr4_epoch['proj_design'], *_cols, model, _PROJ, psi=scan_angles), model)

    def _ll(v):          # ln L for variances v (K, N)
        if model is not None:
            return _ga.loglike(np.broadcast_to(r, v.shape), v)
        return -0.5 * np.sum(r[None, :] ** 2 / v + np.log(v), axis=1)
    if ASTRO_JITTER:
        ls = np.linspace(np.log(ASTRO_JITTER_MIN), np.log(ASTRO_JITTER_MAX), 2000)
        ll = _ll(s2[None, :] + np.exp(2 * ls)[:, None])
        w = np.ones(len(ls)); w[[0, -1]] = 0.5
        return rv0 + float(logsumexp(ll + np.log(w * np.diff(ls)[0]))
                           - np.log(np.log(ASTRO_JITTER_MAX / ASTRO_JITTER_MIN)))
    return rv0 + float(_ll(s2[None, :])[0])


def _accel_estimate(model):
    """Fitted acceleration terms without a companion, flat prior, at the best
    noise parameters: dict(names, units, value, error, significance)."""
    import accel
    _ssn0 = list(_dr4_epoch['proj_param_names'])
    _cols = (np.asarray(t_obs, float) / 365.25, _ssn0.index('pmra'), _ssn0.index('pmdec'))
    if ASTRO_NOISE == 'student':
        return accel.estimate(ROBUST_BASE, *_cols, model, (NU_MIN, NU_MAX), (F_MIN, F_MAX),
                              psi=scan_angles)
    Y = accel.projected_columns(_dr4_epoch['proj_design'], *_cols, model, _PROJ, psi=scan_angles)
    r = np.asarray(along_scan_obs, float)[None, :]
    s2 = np.asarray(sigma_total, float) ** 2
    if ASTRO_JITTER:     # the jitter that maximises the likelihood with the terms
        ls = np.linspace(np.log(ASTRO_JITTER_MIN), np.log(ASTRO_JITTER_MAX), 400)
        v = s2[None, :] + np.exp(2 * ls)[:, None]
        rr = accel.gauss_terms(np.broadcast_to(r, v.shape), v, Y)[2]
        s2 = v[np.argmax(-0.5 * np.sum(rr * rr / v + np.log(v), axis=1))]
    b, C, _ = accel.gauss_terms(r, s2, Y)
    return dict(names=accel.NAMES[model], units=accel.UNITS[model],
                value=b[0], error=np.sqrt(np.diag(C[0])),
                significance=float(np.sqrt(b[0] @ np.linalg.solve(C[0], b[0]))))


LOGZ0 = None
if EVIDENCE:
    try:
        LOGZ0 = _log_evidence_n0()
        if LOGZ0 is not None:
            print(f"Log evidence for N = 0 (no companion, same data and noise model): "
                  f"ln Z0 = {LOGZ0:.2f}")
    except Exception as err:
        print(f"WARNING: N = 0 evidence failed: {err!r}")

# Acceleration models (accel.py): constant acceleration ('accel7')
# and acceleration + jerk ('accel9') of the single-star motion -- the simpler
# hypotheses an orbit with P >> baseline has to beat. Direct integrals with the
# normalisation of ln Z0 and of the tempered fit's ln Z. DR4 epoch data (real
# or injected), both noise models; with RVs the RV part is that of N = 0 (no
# RV trend).
LOGZ_ACCEL, ACCEL_FIT = {}, {}
if LOGZ0 is not None and _PROJ is not None:
    try:
        import accel
        for _m in ('accel7', 'accel9', 'harm2', 'harm23'):
            LOGZ_ACCEL[_m] = _log_evidence_n0(_m)
            ACCEL_FIT[_m] = _accel_estimate(_m)
            print(f"Log evidence for {_m} (no companion, {accel.DESCRIPTION[_m]})"
                  + f": ln Z = {LOGZ_ACCEL[_m]:.2f}; ln B vs single star = "
                  f"{LOGZ_ACCEL[_m] - LOGZ0:+.2f}; significance "
                  f"{ACCEL_FIT[_m]['significance']:.1f}")
            for _n, _u, _v, _s in zip(*(ACCEL_FIT[_m][k] for k in
                                        ('names', 'units', 'value', 'error'))):
                print(f"    {_n:10s} = {_v:+9.4f} +/- {_s:.4f} {_u}")
    except Exception as err:
        print(f"WARNING: acceleration evidences failed: {err!r}")


# ============================================================
# Per-planet results, plots, and summaries
# ============================================================

_stage('results: posterior save, orbit-corrected single-star parameters')
n_inferred   = getattr(result_joint, 'most_probable_n', N_PLANETS)
n_for_plots  = n_inferred

if SOLVER in ('rjmcmc', 'rjflow'):
    if result_joint.n_chain is not None:
        plot_orbit.plot_n_posterior(result_joint.n_chain, N_PLANETS, outdir=outdir)
    else:
        print(f"  [N-posterior plot skipped — RJ_SKIP_IF_N_KNOWN=True]")
    print(f"\nInferred N = {n_inferred}  (true N = {N_PLANETS})")
    if n_inferred != N_PLANETS:
        print(f"  WARNING: inferred N differs from true N.")
    # Source of the reported parameters, plots and best fit.
    #   chain  : the cold chain of the reversible-jump stage at the most
    #            probable N (planets ordered by period).
    #   warmup : the emcee joint pre-refinement (RJ_WARM_START_USE_EMCEE), a
    #            fit at the known number of planets.
    # 'auto' takes the chain when its in-model acceptance is at least
    # RJ_REPORT_MIN_ACC; below that it has moved too little to give intervals.
    # Without a chain or a pre-refinement the peels are used: they are
    # contaminated by crosstalk between the planets (worst at a small-integer
    # period ratio).
    _rj_has_pre = (SOLVER in ('rjmcmc', 'rjflow')
                   and '_pre_result' in globals()
                   and getattr(_pre_result, 'chain', None) is not None)
    _acc_rj = getattr(result_joint, 'acceptance_fraction', None)
    _rj_has_chain = (SOLVER in ('rjmcmc', 'rjflow') and _acc_rj is not None
                     and result_joint.samples.shape[0] > 0)
    _rj_use_chain = _rj_has_chain and (
        RJ_REPORT == 'chain'
        or (RJ_REPORT != 'warmup' and _acc_rj >= RJ_REPORT_MIN_ACC))
    _rj_use_pre = _rj_has_pre and not _rj_use_chain
    if _rj_use_chain:
        print(f"  Parameters below are from the reversible-jump chain: "
              f"{result_joint.samples.shape[0]} cold-chain samples at N = {n_inferred} "
              f"(in-model acceptance {_acc_rj:.3f}).")
    elif _rj_use_pre:
        print("  Parameters below are from the emcee pre-refinement at "
              f"N = {N_PLANETS}"
              + (f" (in-model acceptance of the reversible-jump chain {_acc_rj:.3f}, "
                 f"below {RJ_REPORT_MIN_ACC})" if _acc_rj is not None
                 and _acc_rj < RJ_REPORT_MIN_ACC else "") + ".")
    _h_parts = []
    for _i in range(n_inferred):
        if _rj_use_pre and _i < N_PLANETS:
            _bp_i = _pre_result.best_params[_i * ndim_per:(_i + 1) * ndim_per]
            _h_parts.append(orbit.orbital_model_AL(t_obs, _bp_i, scan_angles, projector=_PROJ))
        elif _i < len(peel_best) and not _rj_use_chain:
            _h_parts.append(orbit.orbital_model_AL(t_obs, peel_best[_i], scan_angles, projector=_PROJ))
        else:
            _bp_i = result_joint.best_params[_i * ndim_per:(_i + 1) * ndim_per]
            _h_parts.append(orbit.orbital_model_AL(t_obs, _bp_i, scan_angles, projector=_PROJ))
    h_best         = np.sum(_h_parts, axis=0) if _h_parts else np.zeros(N_epochs)
    residual_joint = along_scan_obs - h_best
else:
    _rj_use_pre = False
    _rj_use_chain = False
    residual_joint = along_scan_obs - model_all(result_joint.best_params)

# Match inferred planets to true planets by period
if 0 in FIT_IDX and N_PLANETS > 0:
    _p_col   = FIT_IDX.index(0)
    _inf_P   = np.array([
        _pre_result.best_params[i * ndim_per + _p_col]
        if (_rj_use_pre and i < N_PLANETS)
        else peel_best[i][_p_col] if (i < len(peel_best) and not _rj_use_chain)
        else result_joint.best_params[i * ndim_per + _p_col]
        for i in range(n_inferred)
    ])
    _true_P   = true_params_all[:, 0]
    _truth_for = [None] * n_inferred
    _matched   = set()
    for _kt in range(N_TRUE):
        _dists = np.array([
            abs(_inf_P[i] - _true_P[_kt]) / _true_P[_kt] if i not in _matched else np.inf
            for i in range(n_inferred)
        ])
        _ibest = int(np.argmin(_dists))
        # The 20% relative-period cutoff decides which true planet an inferred
        # one corresponds to. With one planet there is nothing to decide, so
        # always match: a poor period match is then part of the difference the
        # overlay is there to show.
        if (N_PLANETS == 1 and N_TRUE == 1) or _dists[_ibest] < 0.20:
            _truth_for[_ibest] = _kt
            _matched.add(_ibest)
else:
    _truth_for = [k if k < N_TRUE else None for k in range(N_PLANETS)]

if not HAVE_TRUTH:
    # Blind fit -- true_params_all is an inert placeholder (see its
    # construction above), never a real answer to display or overlay.
    _truth_for = [None] * len(_truth_for)

# ── Posterior samples to disk, and single-star parameters corrected for the orbit ──
# The posterior is saved (<= POSTERIOR_SAVE_MAX equal-weight draws) so that
# later post-processing never needs a re-run. For DR4 pre-release sources the
# single-star parameters beta (position offsets, parallax, proper motion,
# pseudo-colour) that the projection (I - H) removes are then recovered for
# each draw: beta = N^-1 D^T W (d - s(theta)) with s the unprojected orbit
# model, plus a draw from its conditional covariance N^-1 (user guide,
# "Projected versus fitted single-star terms"). The weights are those of the
# orbit likelihood for that draw, not those of the single-star fit, which
# include the AGIS excess noise that the unmodelled orbit inflates. Wrapped in
# try/except: this runs after the sampling and must never lose a finished run.
POSTERIOR_SAVE_MAX = 20000
SINGLE_STAR_DRAWS  = 2000
single_star_q = None

# Parallax used for the companion mass (EXOPLANETLAB_MASS_PARALLAX):
#   'orbit'  (default) the orbit-corrected parallax recovered from the epoch data
#            for each posterior draw (_single_star_draws), so each draw's a0 is
#            converted to AU with the parallax consistent with its own orbit.
#            Needs the DR4 raw data (not available for injections: the prior
#            is used there). No parallax zero-point correction is applied: the
#            zero-point of the DR4 pre-release astrometry is not established.
#   'prior'  the external value (RV-file header / settings) with its Gaussian
#            uncertainty.
MASS_PARALLAX = os.environ.get('EXOPLANETLAB_MASS_PARALLAX', 'orbit').strip().lower()
_SINGLE_STAR_OK = (DR4_EPOCH and '_dr4_epoch' in globals()
                   and 'proj_data' in _dr4_epoch)


def _single_star_draws(rows, rng):
    """Orbit-corrected single-star parameters for posterior rows (k, ndim):
    beta = N^-1 D^T W (d - s(theta)) plus a draw from N^-1, per row; returns
    (k, p) in the order of _dr4_epoch['proj_param_names'] (see the comment
    above POSTERIOR_SAVE_MAX)."""
    _Dm  = np.asarray(_dr4_epoch['proj_design'], dtype=float)
    _dv  = np.asarray(_dr4_epoch['proj_data'], dtype=float)
    _var = np.asarray(sigma_total, dtype=float) ** 2
    _lam = np.diag(np.asarray(_dr4_epoch['proj_prior_prec'], dtype=float))
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    if ASTRO_NOISE == 'student' and rows.shape[1] > F_IDX:
        # Student-t: the robust single-star fit of the likelihood itself.
        _b = ROBUST.beta_draws(_robust_models(rows), _noise_nu(rows), _noise_f(rows), rng)
        if RV_COUPLED:
            # the parallax of a draw is the one its K stands for; the other
            # terms keep their draw (their dependence on it is neglected)
            _rv_coupled_setup()
            _b[:, _RV_COUPLED_J] = _rv_coupled_scale(rows) / np.abs(rows[:, N_PLANETS * ndim_per])
        return _b
    _jit_col = (ASTRO_JIT_IDX if (ASTRO_JITTER and rows.shape[1] > ASTRO_JIT_IDX) else None)
    out = np.empty((rows.shape[0], _Dm.shape[1]))
    for _k, _row in enumerate(rows):
        _s_raw = np.zeros_like(_dv)
        for _pi in range(N_PLANETS):
            _s_raw += orbit.orbital_model_AL(
                t_obs, unpack_i(_row[_pi * ndim_per:(_pi + 1) * ndim_per], _pi),
                scan_angles, projector=None)
        _w  = 1.0 / (_var + (_row[_jit_col] ** 2 if _jit_col is not None else 0.0))
        _Nm = (_Dm.T * _w) @ _Dm + _lam
        _bh = np.linalg.solve(_Nm, (_Dm.T * _w) @ (_dv - _s_raw))
        _L  = np.linalg.cholesky(np.linalg.inv(_Nm))
        out[_k] = _bh + _L @ rng.standard_normal(_Dm.shape[1])
    return out

try:
    _post_src = _pre_result if _rj_use_pre else result_joint
    _post = np.asarray(getattr(_post_src, 'samples', None), dtype=float)
    if _post.ndim == 2 and _post.shape[0] > 0:
        _rng_post = np.random.default_rng(12345)
        if _post.shape[0] > POSTERIOR_SAVE_MAX:
            _post = _post[_rng_post.choice(_post.shape[0], POSTERIOR_SAVE_MAX, replace=False)]
        _post_names = list(active_names_joint[:_post.shape[1]])
        _save = dict(samples=_post, names=np.array(_post_names), solver=SOLVER,
                     n_fit=N_PLANETS, astro_noise=ASTRO_NOISE,
                     logz0=np.nan if LOGZ0 is None else LOGZ0,
                     **{'logz_' + _m: LOGZ_ACCEL.get(_m, np.nan)
                        for _m in ('accel7', 'accel9', 'harm2', 'harm23')},
                     astro_accel=ASTRO_ACCEL,
                     logz=np.nan if getattr(result_joint, 'logz', None) is None else result_joint.logz,
                     logzerr=np.nan if getattr(result_joint, 'logzerr', None) is None else result_joint.logzerr,
                     # False: the halves of the production chain disagree (solver_ptemcee
                     # EVIDENCE_TOL); ln Z is then from the second half and not converged
                     logz_stable=bool(getattr(result_joint, 'logz_stable', True) is not False),
                     source_mode=SOURCE_MODE, source_id=SOURCE_ID, seed=RUN_SEED)

        if _SINGLE_STAR_OK:
            _ss_names = list(_dr4_epoch['proj_param_names'])
            _n_ss = min(SINGLE_STAR_DRAWS, _post.shape[0])
            _ss = _single_star_draws(_post[:_n_ss], _rng_post)
            _save.update(single_star=_ss, single_star_names=np.array(_ss_names))
            _units = {'ra_offset': 'mas', 'dec_offset': 'mas', 'parallax': 'mas',
                      'pmra': 'mas/yr', 'pmdec': 'mas/yr', 'pseudocolour': '',
                      'hip_frame_ra': 'mas', 'hip_frame_dec': 'mas',
                      'hip_spin_ra': 'mas/yr', 'hip_spin_dec': 'mas/yr',
                      'accel_ra': 'mas/yr^2', 'accel_dec': 'mas/yr^2',
                      'harm2_cos': 'mas', 'harm2_sin': 'mas', 'harm3_cos': 'mas',
                      'harm3_sin': 'mas',
                      'jerk_ra': 'mas/yr^3', 'jerk_dec': 'mas/yr^3'}
            single_star_q = []
            # injected single-star truth (INJECT_SS): the star sits at the
            # reference position at J2017.5, all frame terms are zero
            _ss_true = ({} if INJECT_SS is None else
                        dict(ra_offset=0.0, dec_offset=0.0, parallax=INJECT_SS[0],
                             pmra=INJECT_SS[1], pmdec=INJECT_SS[2], hip_frame_ra=0.0,
                             hip_frame_dec=0.0, hip_spin_ra=0.0, hip_spin_dec=0.0))
            if INJECT_ACCEL is not None:
                _ss_true.update(zip(('accel_ra', 'accel_dec', 'jerk_ra', 'jerk_dec'),
                                    INJECT_ACCEL))
            if INJECT_HARM is not None:
                _ss_true.update(zip(('harm2_cos', 'harm2_sin', 'harm3_cos', 'harm3_sin'),
                                    INJECT_HARM))
            print("\n--- Single-star parameters corrected for the orbit "
                  "(posterior median [16%, 84%]) ---")
            for _j, _nm in enumerate(_ss_names):
                _q16, _q50, _q84 = np.percentile(_ss[:, _j], [16, 50, 84])
                single_star_q.append((_nm, float(_q16), float(_q50), float(_q84),
                                      _units.get(_nm, '')))
                print(f"  {_nm:12s} {_units.get(_nm, ''):7s} {_q50:12.4f} "
                      f"[{_q16:.4f}, {_q84:.4f}]"
                      + (f"   true {_ss_true[_nm]:.4f}" if _nm in _ss_true else ""))
            print("  (position offsets are relative to the reference position of "
                  "the epoch data)")
        if ASTRO_ACCEL and (ACC_GAUSS is not None or not _SINGLE_STAR_OK):
            # the acceleration terms of the fit where the list above does not
            # hold them: the Gaussian model, and Student-t injections without
            # single-star motion
            _acc_names = accel.NAMES[ASTRO_ACCEL]
            _rows_acc = _post[:min(SINGLE_STAR_DRAWS, _post.shape[0])]
            if ACC_GAUSS is not None:
                _m_acc = _astro_models_raw(_rows_acc)
                _m_acc = _m_acc - (_PROJ.X @ (_PROJ.K @ _m_acc.T)).T
                _s2_acc = np.broadcast_to(np.asarray(sigma_total, float) ** 2, _m_acc.shape)
                if ASTRO_JITTER:
                    _s2_acc = _s2_acc + (_rows_acc[:, ASTRO_JIT_IDX] ** 2)[:, None]
                _acc = ACC_GAUSS.draws(np.asarray(along_scan_obs, float)[None, :] - _m_acc,
                                       _s2_acc, _rng_post)
            else:
                _acc = ROBUST.beta_draws(_robust_models(_rows_acc), _noise_nu(_rows_acc),
                                         _noise_f(_rows_acc), _rng_post)[:, ROBUST_BASE.p:]
            _save.update(accel_terms=_acc, accel_names=np.array(_acc_names))
            _acc_true = ({} if INJECTION is None or INJECT_OUTER is not None else
                         dict(zip(('harm2_cos', 'harm2_sin', 'harm3_cos', 'harm3_sin'),
                                  INJECT_HARM or (0.0,) * 4)) if ASTRO_ACCEL in accel.HARMONICS
                         else dict(zip(accel.TERM_NAMES, INJECT_ACCEL or (0.0,) * 4)))
            single_star_q = single_star_q or []
            print(f"\n--- {accel.DESCRIPTION[ASTRO_ACCEL].capitalize()} terms corrected for the orbit "
                  "(posterior median [16%, 84%]) ---")
            for _j, (_nm, _un) in enumerate(zip(_acc_names, accel.UNITS[ASTRO_ACCEL])):
                _q16, _q50, _q84 = np.percentile(_acc[:, _j], [16, 50, 84])
                single_star_q.append((_nm, float(_q16), float(_q50), float(_q84), _un))
                print(f"  {_nm:12s} {_un:8s} {_q50:12.4f} [{_q16:.4f}, {_q84:.4f}]"
                      + (f"   true {_acc_true[_nm]:.4f}" if _nm in _acc_true else ""))

        if RV_TREND:
            # flat-prior estimate of the common RV slope for each posterior draw
            _rows_t = _post[:min(SINGLE_STAR_DRAWS, _post.shape[0])]
            _rr, _se2 = _rv_resid_rows(_rows_t)
            _xt = _rv_trend_x()[None, :]
            _At = np.sum(_xt * _xt / _se2, axis=1)
            _bt = (np.sum(_xt * _rr / _se2, axis=1) / _At
                   + _rng_post.standard_normal(len(_At)) / np.sqrt(_At))
            _save.update(rv_trend=_bt)
            _q16, _q50, _q84 = np.percentile(_bt, [16, 50, 84])
            print(f"\n--- RV trend common to all instruments (flat-prior estimate) ---\n"
                  f"  slope   m/s/yr  {_q50:12.3f} [{_q16:.3f}, {_q84:.3f}]")
        np.savez_compressed(os.path.join(outdir, 'posterior_samples.npz'), **_save)
        print(f"Saved {os.path.join(outdir, 'posterior_samples.npz')} "
              f"({_post.shape[0]} draws x {_post.shape[1]} parameters)")
except Exception as _ex:
    print(f"  WARNING: posterior saving / single-star recovery skipped ({_ex!r}).")

print("\n--- Fitted Parameters ---")
print(f"{'Plt':5s} {'Parameter':10s} {'Unit':6s} {'Median':>12s} "
      f"{'16th':>10s} {'84th':>10s} {'True':>12s}")
print("-" * 68)

# Astrometric jitter: one per source, so printed once here (and repeated in
# each planet's summary.txt), with the median formal error for comparison.
astro_jitter_q = None
if ASTRO_JITTER and getattr(result_joint, 'samples', None) is not None:
    _sj16, _sj50, _sj84 = np.percentile(
        np.asarray(result_joint.samples)[:, ASTRO_JIT_IDX], [16, 50, 84])
    astro_jitter_q = (float(_sj16), float(_sj50), float(_sj84),
                      float(np.median(sigma_total)))
    print(f"  all  {'jitter_ast':10s} {'mas':6s} {_sj50:12.4f} "
          f"{_sj16:10.4f} {_sj84:10.4f} {'N/A':>12s}")
    print(f"         (added in quadrature to the along-scan errors; median "
          f"formal error {np.median(sigma_total):.4f} mas)")
astro_noise_q = None
if ASTRO_NOISE == 'student' and getattr(result_joint, 'samples', None) is not None:
    _smp = np.asarray(result_joint.samples)
    astro_noise_q = dict(nu=tuple(float(v) for v in np.percentile(_smp[:, NU_IDX], [16, 50, 84])),
                         f=tuple(float(v) for v in np.percentile(_smp[:, F_IDX], [16, 50, 84])),
                         sigma_med=float(np.median(sigma_total)))
    if HIPPARCOS:
        astro_noise_q.update(
            nu_hip=tuple(float(v) for v in np.percentile(_smp[:, NUH_IDX], [16, 50, 84])),
            f_hip=tuple(float(v) for v in np.percentile(_smp[:, FH_IDX], [16, 50, 84])),
            sigma_med_hip=float(np.median(HIP['c']['sigma'])))
    for _nm, _q in (('nu_ast', astro_noise_q['nu']), ('f_ast', astro_noise_q['f'])) + (
            (('nu_hip', astro_noise_q['nu_hip']), ('f_hip', astro_noise_q['f_hip']))
            if HIPPARCOS else ()):
        print(f"  all  {_nm:10s} {'':6s} {_q[1]:12.4f} {_q[0]:10.4f} {_q[2]:10.4f} {'N/A':>12s}")
    print("         (Student-t along-scan errors t_nu(0, f^2 sigma^2); nu -> 100 is Gaussian)")

_all_best_params_j = []   # collected for the joint orbit-sky plot below
_all_true_params_j  = []

# T0 display fold: only when both P and T0 are being fitted (need a period to
# fold by).  _t0_ref is the first observation epoch.
_t0_loc  = FIT_IDX.index(2) if 2 in FIT_IDX else None
_p_loc   = FIT_IDX.index(0) if 0 in FIT_IDX else None
_do_fold = FOLD_T0 and _t0_loc is not None and _p_loc is not None
_t0_ref  = float(t_obs.min())
_t0_raw_notes = []   # (planet_j, raw_q16, raw_q50, raw_q84, k, P0) for printing
_t0_fold_par  = {}   # planet_j -> (P0, anchor) used to fold, reused for joint corner


def _fold_result_t0(r, p_loc, t0_loc, t_ref):
    """Return a shallow copy of result ``r`` with its T0 column folded, plus a
    tuple (raw_q16, raw_q50, raw_q84, k, P0, anchor): the raw (unfolded) T0
    quantiles, k = #periods the raw median sits above the folded one, the median
    period P0, and the fold anchor (so the T0 truth can be folded to match)."""
    P0  = float(np.median(np.asarray(r.samples, dtype=float)[:, p_loc]))
    col = np.asarray(r.samples, dtype=float)[:, t0_loc]
    anc = _t0_anchor(col, P0, t_ref)
    raw_q16, raw_q50, raw_q84 = np.percentile(col, [16, 50, 84])

    # Each sample (chain entry, best fit) is folded with its OWN period (see
    # _fold_t0); P0 only sets the anchor and is reported.
    upd = {}
    s = np.array(r.samples, dtype=float, copy=True)
    s[:, t0_loc] = _fold_t0(s[:, t0_loc], s[:, p_loc], anc); upd['samples'] = s
    ch = getattr(r, 'chain', None)
    if ch is not None and np.asarray(ch).ndim == 3 and np.asarray(ch).shape[2] > max(t0_loc, p_loc):
        ch = np.array(ch, dtype=float, copy=True)
        ch[:, :, t0_loc] = _fold_t0(ch[:, :, t0_loc], ch[:, :, p_loc], anc)
        upd['chain'] = ch
    ws = getattr(r, 'weighted_samples', None)
    if ws is not None and np.asarray(ws).ndim == 2 and np.asarray(ws).shape[1] > max(t0_loc, p_loc):
        ws = np.array(ws, dtype=float, copy=True)
        ws[:, t0_loc] = _fold_t0(ws[:, t0_loc], ws[:, p_loc], anc)
        upd['weighted_samples'] = ws
    bp = getattr(r, 'best_params', None)
    if bp is not None:
        bp = np.array(bp, dtype=float, copy=True)
        if bp.ndim == 1 and bp.shape[0] > max(t0_loc, p_loc):
            bp[t0_loc] = float(_fold_t0(bp[t0_loc], bp[p_loc], anc)); upd['best_params'] = bp
    folded = SimpleNamespace(**{**vars(r), **upd})
    fold_med = float(np.median(s[:, t0_loc]))
    k = int(round((raw_q50 - fold_med) / P0)) if P0 else 0
    return folded, (raw_q16, raw_q50, raw_q84, k, P0, anc)


_stage('per-planet mass, plots (corner, traces, sky) and summaries')
for planet_j in range(n_for_plots):
    subdir_j = outdir if n_for_plots == 1 else os.path.join(outdir, f'planet_{planet_j + 1}')
    os.makedirs(subdir_j, exist_ok=True)

    # For rjmcmc/rjflow each planet comes from the same source as the joint
    # corner (see RJ_REPORT): the reversible-jump chain, or the emcee joint
    # pre-refinement, or the raw peel if there is neither.  A planet beyond
    # the peeled count always comes from the chain.
    if _rj_use_pre and planet_j < N_PLANETS:
        result_j = extract_planet_result(_pre_result, planet_j)
    elif (SOLVER in ('rjmcmc', 'rjflow') and planet_j < len(peel_results)
          and not _rj_use_chain):
        result_j = peel_results[planet_j]
    else:
        result_j = extract_planet_result(result_joint, planet_j)

    active_names_j  = [PARAM_NAMES[gi] for gi in FIT_IDX]
    active_units_j  = [PARAM_UNITS[gi] for gi in FIT_IDX]
    active_scales_j = PARAM_SCALES[FIT_IDX]

    k_true          = _truth_for[planet_j]
    has_truth       = k_true is not None
    active_truths_j = (np.array(true_params_all[k_true][FIT_IDX], dtype=float)
                       if has_truth else None)

    # Fold T0 (and the T0 truth) into the first-epoch period cycle for display.
    if _do_fold and getattr(result_j, 'samples', None) is not None \
            and np.asarray(result_j.samples).ndim == 2 \
            and np.asarray(result_j.samples).shape[1] > _t0_loc \
            and np.asarray(result_j.samples).shape[0] > 1:
        result_j, _raw = _fold_result_t0(result_j, _p_loc, _t0_loc, _t0_ref)
        # _raw = (raw_q16, raw_q50, raw_q84, k, P0, anchor)
        _t0_raw_notes.append((planet_j, *_raw[:5]))
        _t0_fold_par[planet_j] = (_raw[4], _raw[5])   # (P0, anchor)
        if has_truth:
            _Pt = float(active_truths_j[_p_loc]) if active_truths_j[_p_loc] is not None else np.nan
            active_truths_j[_t0_loc] = float(_fold_t0(
                active_truths_j[_t0_loc], _Pt if np.isfinite(_Pt) and _Pt > 0 else _raw[4],
                _raw[5]))

    for j, (name, unit, scale) in enumerate(
            zip(active_names_j, active_units_j, active_scales_j)):
        vals           = result_j.samples[:, j] * scale
        q16, q50, q84  = np.percentile(vals, [16, 50, 84])
        truth_str      = (f"{active_truths_j[j] * scale:12.4f}"
                          if has_truth else f"{'N/A':>12s}")
        print(f"  p{planet_j+1}   {name:10s} {unit:6s} {q50:12.4f} "
              f"{q16:10.4f} {q84:10.4f} {truth_str}")

    # RV semi-amplitude of this planet, from the joint chain. Printed for every
    # RV run -- the physical-parameter block of summary.txt only shows K when a
    # stellar mass and parallax are supplied, which simulated sources lack.
    # |K|: a negative K is the same orbit on the omega+180 branch (see the
    # K_MIN comment), so report the conventional K > 0 magnitude.
    rv_K_j = None
    if RV_ENABLED and SOLVER not in ('rjmcmc', 'rjflow'):
        _K_samp = np.abs(np.asarray(result_joint.samples)[:, N_PLANETS * ndim_per + planet_j])
        _K16, _K50, _K84 = np.percentile(_K_samp, [16, 50, 84])
        _K_true = (float(K_TRUE[k_true])
                   if RV_MODE == 'simulated' and has_truth and k_true < len(K_TRUE)
                   else None)
        rv_K_j = (float(_K16), float(_K50), float(_K84), _K_true)
        _K_true_str = f"{_K_true:12.4f}" if _K_true is not None else f"{'N/A':>12s}"
        print(f"  p{planet_j+1}   {'K':10s} {'m/s':6s} {_K50:12.4f} "
              f"{_K16:10.4f} {_K84:10.4f} {_K_true_str}")

    _t0_raw_arg = None
    if _do_fold and _t0_raw_notes and _t0_raw_notes[-1][0] == planet_j:
        _, _r16, _r50, _r84, _k, _P0 = _t0_raw_notes[-1]
        _sc  = active_scales_j[_t0_loc]
        _un  = active_units_j[_t0_loc]
        _t0_raw_arg = (_t0_loc, _un, _sc, _r16, _r50, _r84, _k, _P0)
        print(f"  p{planet_j+1}   {'T0(raw)':10s} {_un:6s} {_r50*_sc:12.4f} "
              f"{_r16*_sc:10.4f} {_r84*_sc:10.4f} {'':>12s}")
        print(f"         (T0 is degenerate mod P; value above is folded to the "
              f"first-epoch cycle, raw = folded {_k:+d}*P, P~{_P0:.4g} d)")
    print()

    if result_j.chain_burn is not None:
        plot_orbit.plot_burnin(
            result_j.chain_burn, result_j.logprob_burn,
            active_names_j, active_units_j, active_scales_j, active_truths_j,
            outdir=subdir_j,
        )
    if result_j.chain is not None:
        plot_orbit.plot_traces(
            result_j.chain,
            active_names_j, active_units_j, active_scales_j, active_truths_j,
            result_j.nsteps_prod, tau=result_j.tau,
            outdir=subdir_j,
        )

    _cd_j, _cw_j = corner_input(result_j, len(active_names_j))
    plot_orbit.plot_corner(
        _cd_j, active_names_j, active_units_j,
        active_scales_j, active_truths_j,
        outdir=subdir_j, weights=_cw_j,
    )

    best_params_full_j = (result_j.best_params
                          if SOLVER in ('rjmcmc', 'rjflow')
                          else unpack_i(result_j.best_params, planet_j))
    true_params_j      = true_params_all[k_true] if has_truth else None
    _all_best_params_j.append(best_params_full_j)
    _all_true_params_j.append(true_params_j)

    plot_orbit.plot_orbit_sky(
        t_obs, best_params_full_j, true_params_j, scan_angles,
        source_id=(SOURCE_ID if DR4_EPOCH
                   else f'planet_{planet_j+1}'),
        outdir=subdir_j,
    )

    # Per-planet residual: data minus this planet's best fit
    residual_k = along_scan_obs - orbit.orbital_model_AL(
        t_obs, best_params_full_j, scan_angles, projector=_PROJ)
    plot_orbit.plot_residuals(t_obs, residual_k, sigma_total, outdir=subdir_j)

    snr_j          = (np.std(orbit.orbital_model_AL(t_obs, true_params_all[k_true], scan_angles, projector=_PROJ))
                      / np.median(sigma_total) if has_truth else np.nan)
    true_params_plot = true_params_all[k_true] if has_truth else np.zeros(7)

    mass_info_j = None
    if M_STAR_MSUN is not None and PARALLAX_MAS is not None:
        _P_j, _e_j, _T0_j, _A_j, _B_j, _F_j, _G_j = best_params_full_j
        _samp = getattr(result_joint, 'samples', None)
        if SOLVER in ('rjmcmc', 'rjflow') or _samp is None or len(_samp) == 0:
            # The RJ solvers' leaf-based posterior does not map onto the flat
            # joint vector used below: use the best-fit point (uncertainty not
            # propagated).
            K_best_j = None
            if RV_ENABLED and SOLVER not in ('rjmcmc', 'rjflow'):
                K_best_j = float(result_joint.best_params[N_PLANETS * ndim_per + planet_j])
            mass_info_j = mass_estimate.estimate_planet_mass(
                P_days=_P_j, e=_e_j, A=_A_j, B=_B_j, F=_F_j, G=_G_j,
                parallax_mas=PARALLAX_MAS, m_star_msun=M_STAR_MSUN,
                K_ms=K_best_j,
            )
            mass_info_j['is_posterior'] = False
        else:
            # Full posterior propagation: the mass is computed per draw of the
            # joint chain (mass_estimate.estimate_planet_mass_posterior), which
            # carries the spread of the orbit and, with
            # PROPAGATE_MSTAR_PARALLAX, of M_star and the parallax.
            _n_draw  = min(500, len(_samp))
            _idx     = np.random.choice(len(_samp), _n_draw, replace=False)
            _draws   = _samp[_idx]
            _ti_cols = slice(planet_j * ndim_per, (planet_j + 1) * ndim_per)
            _unpacked = np.array([unpack_i(row, planet_j) for row in _draws[:, _ti_cols]])
            _P_arr, _e_arr = _unpacked[:, 0], _unpacked[:, 1]
            _A_arr, _B_arr, _F_arr, _G_arr = (_unpacked[:, 3], _unpacked[:, 4],
                                               _unpacked[:, 5], _unpacked[:, 6])

            _K_arr = _draws[:, N_PLANETS * ndim_per + planet_j] if RV_ENABLED else None

            if FIT_MSTAR_PARALLAX:
                _mstar_idx = N_PLANETS * ndim_per + N_PLANETS + 2 * N_RV_INST
                _mstar_arr = _draws[:, _mstar_idx]
                _plx_arr   = _draws[:, _mstar_idx + 1]
            elif PROPAGATE_MSTAR_PARALLAX:
                # Not sampled (see MSTAR_PLX_IN_CHAIN): their posterior is their
                # prior, independent of the orbit, so draw them independently.
                _mstar_arr = _draw_positive_normal(M_STAR_MSUN, M_STAR_SIGMA, _n_draw)
                _plx_arr   = _draw_positive_normal(PARALLAX_MAS, PARALLAX_SIGMA, _n_draw)
            else:
                _mstar_arr = M_STAR_MSUN
                _plx_arr   = PARALLAX_MAS
            _plx_source = 'external prior'
            if MASS_PARALLAX == 'orbit' and _SINGLE_STAR_OK:
                try:
                    _ssd = _single_star_draws(_draws, np.random.default_rng(2024 + planet_j))
                    _plx_arr = _ssd[:, list(_dr4_epoch['proj_param_names']).index('parallax')]
                    _plx_source = 'orbit-corrected (this fit, no zero-point correction)'
                except Exception as _ex:
                    print(f"  WARNING: orbit-corrected parallax unavailable ({_ex!r}); "
                          f"using the external prior for the mass.")
            elif MASS_PARALLAX == 'orbit':
                print("  NOTE: no orbit-corrected parallax for this data mode; the mass "
                      "uses the external parallax prior.")

            mass_info_j = mass_estimate.estimate_planet_mass_posterior(
                P_days=_P_arr, e=_e_arr, A=_A_arr, B=_B_arr, F=_F_arr, G=_G_arr,
                parallax_mas=_plx_arr, m_star_msun=_mstar_arr, K_ms=_K_arr,
            )
            mass_info_j['is_posterior'] = True
            mass_info_j['n_draw']       = _n_draw
            mass_info_j['parallax_source'] = _plx_source

    rv_inst_info_j = None
    if RV_ENABLED and SOLVER not in ('rjmcmc', 'rjflow'):
        _gamma_start = N_PLANETS * ndim_per + N_PLANETS
        _jit_start   = _gamma_start + N_RV_INST
        # Posterior medians with 16/84 percentile bounds (not the single
        # best-fit sample): per-instrument gamma/jitter are nuisance
        # parameters and, for instruments with 1-2 points, prior dominated,
        # so a single max-posterior/max-likelihood sample is not
        # representative.
        _rv_samp = np.asarray(result_joint.samples)
        def _q(_col):
            _lo, _md, _hi = np.percentile(_rv_samp[:, _col], [16, 50, 84])
            return float(_lo), float(_md), float(_hi)
        rv_inst_info_j = []
        for _i, _name in enumerate(RV_INSTRUMENTS):
            _g = _q(_gamma_start + _i)
            _j = _q(_jit_start + _i)
            rv_inst_info_j.append(dict(
                name=_name,
                gamma_ms=_g[1], gamma_lo=_g[0], gamma_hi=_g[2],
                jitter_ms=_j[1], jitter_lo=_j[0], jitter_hi=_j[2],
                jitter_fixed=_i in FIXED_JITTER_INST,
                n_obs=int(np.sum(rv_inst_idx == _i))))

    # Relative astrometry of this companion: a_rel = s a0 and, with a
    # parallax, the dynamical total mass M_tot = (a_rel / plx)^3 / P_yr^2 and,
    # for a dark companion (f_flux = 0, so s = 1 / f_mass), M2 = M_tot / s.
    relast_info_j = None
    if planet_j in RELAST_POS and getattr(result_joint, 'samples', None) is not None:
        from generate_sources import campbell_elements as _campbell
        _rs = np.asarray(result_joint.samples)
        _rs = _rs[np.random.default_rng(3030 + planet_j).choice(
            len(_rs), min(2000, len(_rs)), replace=False)]
        _rr = _planet_rows(_rs, planet_j)
        _a0r = _campbell(_rr[:, 3], _rr[:, 4], _rr[:, 5], _rr[:, 6])[0]
        _sr = _rs[:, RELAST_POS[planet_j]]
        _q3 = lambda v: tuple(float(x) for x in np.percentile(v, [16, 50, 84]))
        relast_info_j = dict(n_obs=int(np.sum(rel_comp == planet_j)), s=_q3(_sr),
                             a_rel=_q3(_sr * _a0r), true_s=RELAST_TRUE_SCALE)
        if RELAST_JITTER:
            relast_info_j.update(
                jitter=_q3(_rs[:, RELAST_JIT_IDX]),
                sigma_med=float(np.median(np.concatenate([rel_sx, rel_sy]))))
        _plx_r, _plx_src = None, None
        if MASS_PARALLAX == 'orbit' and _SINGLE_STAR_OK:
            try:
                _plx_r = _single_star_draws(_rs, np.random.default_rng(4040 + planet_j))[
                    :, list(_dr4_epoch['proj_param_names']).index('parallax')]
                _plx_src = 'orbit-corrected (this fit)'
            except Exception as _ex:
                print(f"  WARNING: orbit-corrected parallax unavailable for a_rel ({_ex!r})")
        if _plx_r is None and PARALLAX_MAS is not None:
            _plx_r = (_draw_positive_normal(PARALLAX_MAS, PARALLAX_SIGMA, len(_rs))
                      if PARALLAX_SIGMA else np.full(len(_rs), float(PARALLAX_MAS)))
            _plx_src = 'external'
        if _plx_r is not None:
            _arel_au = _sr * _a0r / _plx_r
            _mtot = _arel_au ** 3 / (_rr[:, 0] / 365.25) ** 2
            relast_info_j.update(a_rel_au=_q3(_arel_au), m_tot=_q3(_mtot),
                                 m2_dark=_q3(_mtot / _sr), plx_source=_plx_src)
            if RELAST_FLUX_RATIO is not None:
                # luminous companion: beta = r / (1 + r), f = 1 / s + beta
                _fr = (_draw_positive_normal(RELAST_FLUX_RATIO[0], RELAST_FLUX_RATIO[1], len(_rs))
                       if RELAST_FLUX_RATIO[1] > 0 else np.full(len(_rs), RELAST_FLUX_RATIO[0]))
                _fm = 1.0 / _sr + _fr / (1.0 + _fr)
                _okf = _fm < 1.0
                relast_info_j.update(flux_ratio=RELAST_FLUX_RATIO,
                                     f_unphysical=float(np.mean(~_okf)))
                if np.any(_okf):
                    relast_info_j.update(mass_fraction=_q3(_fm[_okf]),
                                         m2=_q3((_fm * _mtot)[_okf]),
                                         m1=_q3(((1.0 - _fm) * _mtot)[_okf]))
        print(f"  Relative astrometry, companion {planet_j + 1}: s = a_rel/a0 = "
              f"{relast_info_j['s'][1]:.4g} [{relast_info_j['s'][0]:.4g}, "
              f"{relast_info_j['s'][2]:.4g}], a_rel = {relast_info_j['a_rel'][1]:.4g} mas"
              + (f", M_tot = {relast_info_j['m_tot'][1]:.4g} Msun" if 'm_tot' in relast_info_j else ''))
        _rdraw = _rs[:min(100, len(_rs))]
        plot_orbit.plot_relative_orbit(
            rel_t[rel_comp == planet_j], rel_x[rel_comp == planet_j], rel_y[rel_comp == planet_j],
            rel_sx[rel_comp == planet_j], rel_sy[rel_comp == planet_j],
            _planet_rows(_rdraw, planet_j), _rdraw[:, RELAST_POS[planet_j]],
            t_obs, outdir=subdir_j)

    plot_orbit.write_summary(
        outdir=subdir_j,
        solver_name=SOLVER,
        n_transits=N_epochs,
        t_total_days=_obs_span_days,
        obs_years=_obs_span_days / 365.25,
        snr=snr_j,
        sigma_noise=SUMMARY_SIGMA,
        sigma_label=SUMMARY_SIGMA_LABEL,
        param_names=PARAM_NAMES,
        param_units=PARAM_UNITS,
        param_scales=PARAM_SCALES,
        true_params=true_params_plot,
        fit_flags=FIT_FLAGS,
        fit_idx=FIT_IDX,
        active_names=active_names_j,
        active_units=active_units_j,
        active_scales=active_scales_j,
        active_truths=active_truths_j,
        result=result_j,
        residual=residual_joint,
        mass_info=mass_info_j,
        rv_instruments=rv_inst_info_j,
        rv_K=rv_K_j,
        astro_jitter=astro_jitter_q,
        astro_noise=astro_noise_q,
        rel_astrometry=relast_info_j,
        hipparcos=(None if HIP is None else dict(
            warning=HIP_WARNING, hip_id=HIP['hip_id'], n=HIP['n'],
            isol_n=HIP['hdr']['isol_n'], notes=HIP['notes'], rv=HIP['rv'],
            persp=HIP['persp'], offset=HIP['offset'],
            synthetic=INJECTION is not None)),
        single_star=single_star_q,
        t0_raw=_t0_raw_arg,
        truth_is_literature=(DR4_EPOCH and HAVE_TRUTH),
        # result_j.tau is this planet's own 7-orbital-element slice of the
        # shared joint chain (see extract_planet_result) -- pass the full,
        # unsliced joint diagnostics too so the saved summary's convergence
        # verdict reflects the whole chain (RV K/gamma/jitter, M_star/
        # parallax included), not just this planet's own elements.
        full_tau=getattr(result_joint, 'tau', None),
        full_active_names=active_names_joint,
        full_nsteps_prod=getattr(result_joint, 'nsteps_prod', None),
        full_tau_err=getattr(result_joint, 'tau_err', None),
    )

print("----------------------------------------------------------------------")

_stage('joint plots (multi-planet corner, RV fit)')

# Hipparcos IAD at the joint best fit: the robust single-star and
# frame terms refitted for that orbit (as in the likelihood), then removed
if HIPPARCOS and getattr(result_joint, 'best_params', None) is not None:
    try:
        _Xb = np.atleast_2d(np.asarray(result_joint.best_params, dtype=float))
        _mb = _robust_models(_Xb)
        _s2b = ROBUST._per_row(_noise_f(_Xb), 1) ** 2 * ROBUST.sig2[None, :]
        _bb, _rb, _ = ROBUST.fit_beta(ROBUST.d[None, :] - _mb, _s2b,
                                      ROBUST._per_row(_noise_nu(_Xb), 1))
        _h = slice(len(ROBUST.d) - HIP['n'], None)
        plot_orbit.plot_hipparcos(
            1991.25 + HIP['c']['t'], (ROBUST.d - _bb[0] @ ROBUST.X.T)[_h],
            np.sqrt(_s2b[0, _h]), _mb[0, _h], outdir=outdir,
            synthetic=INJECTION is not None)
    except Exception as _ex:
        print(f"  WARNING: Hipparcos plot skipped ({_ex!r})")

if n_for_plots > 1:
    _js = result_joint.samples
    if n_for_plots <= 2 and _js.size > 0 and _js.shape[1] >= n_for_plots * ndim_per:
        _nd_j  = n_for_plots * ndim_per            # planet params only (RV K/gamma trail)
        _ws_j  = getattr(result_joint, 'weighted_samples', None)
        _chn_j = getattr(result_joint, 'chain', None)
        _joint_shim = SimpleNamespace(
            samples          = _js[:, :_nd_j],
            weighted_samples = _ws_j[:, :_nd_j] if _ws_j is not None else None,
            weights          = getattr(result_joint, 'weights', None),
            chain            = (_chn_j[:, :, :_nd_j]
                                if _chn_j is not None and _chn_j.shape[2] >= _nd_j
                                else None),
            log_probs        = getattr(result_joint, 'log_probs', None),
            logp_df          = _nd_j,
        )
        _jd, _jw = corner_input(_joint_shim, _nd_j)

        # For rjmcmc/rjflow reported from the warm-up (see RJ_REPORT): draw
        # the joint corner from the emcee joint pre-refinement chain, through
        # the same corner_samples() chi-square cut + head-discard as the
        # standalone emcee/nessai joint corners.
        if _rj_use_pre and n_for_plots == N_PLANETS:
            _pre_prod = getattr(solver_emcee, 'NSTEPS_PROD', 25000)
            _jd = corner_samples(_pre_result, N_PLANETS * ndim_per,
                                 discard=min(PLOT_JOINT_DISCARD, _pre_prod // 5))
            _jw = None

        # Fold each planet's joint-corner T0 column (and its T0 truth) to match
        # the per-planet plots.  Reuse the (P0, anchor) computed per planet above.
        _tp_disp = true_params_all
        if _do_fold and _t0_fold_par:
            _jd = np.array(_jd, dtype=float, copy=True)
            _tp_disp = np.array(true_params_all, dtype=float, copy=True)
            for _pi, (_P0f, _ancf) in _t0_fold_par.items():
                _c = _pi * ndim_per + _t0_loc
                _cp = _pi * ndim_per + _p_loc
                if _c < _jd.shape[1] and _cp < _jd.shape[1]:
                    _jd[:, _c] = _fold_t0(_jd[:, _c], _jd[:, _cp], _ancf)
                _kt = _truth_for[_pi] if _pi < len(_truth_for) else None
                if _kt is not None:
                    _Pt = _tp_disp[_kt, 0]
                    _tp_disp[_kt, 2] = _fold_t0(_tp_disp[_kt, 2],
                                                _Pt if np.isfinite(_Pt) and _Pt > 0 else _P0f,
                                                _ancf)

        plot_orbit.plot_joint_corner(
            _jd,
            n_for_plots,
            [PARAM_NAMES[gi] for gi in FIT_IDX],
            [PARAM_UNITS[gi] for gi in FIT_IDX],
            PARAM_SCALES[FIT_IDX],
            _truth_for,
            _tp_disp,
            FIT_IDX,
            outdir=outdir,
            weights=_jw,
        )
    elif n_for_plots > 2:
        print(f"  [joint corner plot skipped — {n_for_plots}×{ndim_per} parameters]")

    plot_orbit.plot_joint_orbit_sky(
        _all_best_params_j, _all_true_params_j, t_span=(float(t_obs.min()), float(t_obs.max())),
        source_id=(SOURCE_ID if DR4_EPOCH else ''),
        outdir=outdir,
    )

    plot_orbit.plot_residuals(t_obs, residual_joint, sigma_total, outdir=outdir)

if RV_ENABLED and t_rv is not None and result_joint.samples is not None:
    _k_true_plot  = (K_TRUE[:N_PLANETS] if RV_MODE == 'simulated' else None)
    _gam_true_plt = (GAMMA_TRUE         if RV_MODE == 'simulated' else None)
    plot_orbit.plot_rv(
        t_rv, rv_obs, rv_sigma, rv_inst_idx, RV_INSTRUMENTS,
        result_joint.best_params,
        result_joint.samples,
        N_PLANETS, ndim_per,
        true_params_all=(true_params_all if SOURCE_MODE == 'simulated' else None),
        K_true=_k_true_plot,
        gamma_true=_gam_true_plt,
        outdir=outdir,
    )

_t_end_stages = datetime.datetime.now()
_timing_lines = _stage_table(_t_end_stages)
print()
print('\n'.join(_timing_lines))
try:
    import glob as _glob
    for _sf in _glob.glob(os.path.join(outdir, '**', 'summary.txt'), recursive=True):
        with open(_sf, 'a') as _fh:
            _fh.write('\n' + '\n'.join(_timing_lines) + '\n')
except Exception as _ex:
    print(f"  WARNING: could not append the stage timing to summary.txt ({_ex!r}).")
_t_end      = datetime.datetime.now()
_duration   = _t_end - _t_start
print(f"\nRun started:  {_t_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Run finished: {_t_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration:     {str(_duration).split('.')[0]}")
print(f"SOURCE_ID    = {SOURCE_ID}")
print(f"RV data used = {RV_ENABLED}"
      + (f"  (RV_MODE={RV_MODE!r}, RV_FILE={RV_FILE!r})" if RV_MODE == 'external'
         else f"  (RV_MODE={RV_MODE!r})" if RV_ENABLED else ""))
print(f"Astrometric jitter fitted = {ASTRO_JITTER}")
print(f"Astrometric noise model    = {ASTRO_NOISE}")
print(f"Jitters sampled in log    = {bool(LOG_JITTER and JITTER_IDX)}")
print(f"Orientation prior         = {ORIENT_PRIOR}")
print(f"Tempered joint fit        = {bool(_use_pt) if '_use_pt' in globals() else False}")
_fit_label = f"N={N_PLANETS}" + (f"+{ASTRO_ACCEL}" if ASTRO_ACCEL else "")
if getattr(result_joint, 'logz', None) is not None:
    print(f"Log evidence ({_fit_label})   = {result_joint.logz:.2f} +/- {result_joint.logzerr:.2f} "
          f"(bridge; thermodynamic integration "
          f"{getattr(result_joint, 'logz_ti', float('nan')):.2f})"
          + ("" if getattr(result_joint, 'logz_stable', True) is not False
             else "  -- NOT STABLE: run longer"))
if LOGZ0 is not None:
    print(f"Log evidence (N = 0)   = {LOGZ0:.2f}; ln B({_fit_label} vs 0) = "
          f"{result_joint.logz - LOGZ0:.2f}" if getattr(result_joint, 'logz', None) is not None
          else f"Log evidence (N = 0)   = {LOGZ0:.2f}")
print(f"RV trend in fit           = {RV_TREND}")
print(f"K coupled to the orbit    = {RV_COUPLED}")
print(f"Seed (EXOPLANETLAB_SEED)  = {RUN_SEED}" + ("" if _SEED_ENV else "  (drawn for this run)"))
try:
    _run_info = _write_run_info(finished=True)
    print("Versions                  = " + ", ".join(
        f"{k} {v}" for k, v in _run_info['versions'].items()) + f"; git {_run_info['git']}")
except Exception as _ex:
    print(f"  WARNING: run_info.json not written ({_ex!r})")
print(f"Acceleration terms in fit = "
      f"{ASTRO_ACCEL or ('accel7 (Hipparcos + Gaia)' if HIP_ACCEL else 'none')}")
for _m, _z in LOGZ_ACCEL.items():
    print(f"Log evidence ({_m}) = {_z:.2f}; ln B({_m} vs single star) = {_z - LOGZ0:+.2f}"
          + (f"; ln B({_fit_label} vs {_m}) = {result_joint.logz - _z:+.2f}"
             if getattr(result_joint, 'logz', None) is not None else ""))
print(f"M_star/parallax in chain  = {FIT_MSTAR_PARALLAX}"
      + ("  (drawn from their priors for the mass instead)"
         if PROPAGATE_MSTAR_PARALLAX and not FIT_MSTAR_PARALLAX else ""))
print("\nDone.")
