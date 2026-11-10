# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""detect.py: pieces of the signal screen (the full screen takes ~20 s per pass
and is exercised by Tools/false_positive_test.py)."""

import numpy as np
import pytest

import detect
import orbit

RNG = np.random.default_rng(4)
N = 300
T = np.sort(RNG.uniform(-900.0, 900.0, N))
PSI = RNG.uniform(0.0, 2.0 * np.pi, N)
SIG = RNG.uniform(0.05, 0.2, N)


@pytest.fixture(autouse=True)
def _no_projector():
    detect._PROJ = None
    yield
    detect._PROJ = None


def test_period_grid_covers_the_range():
    g = detect._make_period_grid(1800.0, 10.0, 3600.0, 4)
    assert g.max() == pytest.approx(3600.0) and 10.0 <= g.min() < 10.1
    assert np.all(np.diff(1.0 / g) > 0)


@pytest.mark.parametrize('nu, f', [(3.0, 1.0), (2.0, 0.8), (8.0, 1.5)])
def test_student_fit_recovers_the_noise_parameters(nu, f):
    sig = np.random.default_rng(7).uniform(0.05, 0.2, 4000)
    y = f * sig * np.random.default_rng(8).standard_t(nu, 4000)
    nu_hat, f_hat = detect._t_fit(y, sig)
    assert nu_hat == pytest.approx(nu, rel=0.25) and f_hat == pytest.approx(f, rel=0.08)


def test_score_transform_leaves_gaussian_data_nearly_unchanged():
    y = SIG * RNG.standard_normal(N)
    score, clip, s, nu, f = detect._robustify(y, SIG)
    assert nu > 20 and f == pytest.approx(1.0, abs=0.15)
    assert np.corrcoef(score, y)[0, 1] > 0.98


def test_score_transform_bounds_outliers():
    y = SIG * RNG.standard_t(3.0, N)
    y[11] = 500 * SIG[11]
    score, clip, s, nu, f = detect._robustify(y, SIG)
    assert np.max(np.abs(score / s)) <= (nu + 1) / (2 * np.sqrt(nu)) + 1e-9   # max of psi
    assert abs(score[11]) < abs(np.median(np.abs(score))) * 5
    assert np.max(np.abs(clip / s)) <= detect.ROBUST_CLIP + 1e-12


def test_score_transform_stays_linear_when_a_signal_dominates():
    y = SIG * RNG.standard_normal(N) + 20.0 * np.sin(T / 60.0)
    score, clip, s, nu, f = detect._robustify(y, SIG)
    assert np.corrcoef(score, y)[0, 1] > 0.99


def test_periodogram_peaks_at_the_injected_period():
    par = [240.0, 0.2, 30.0, *orbit.ti_from_campbell(0.5, 1.0, 0.7, 1.0)]
    y = orbit.orbital_model_AL(T, par, PSI) + SIG * RNG.standard_normal(N)
    cp, sp = np.cos(PSI), np.sin(PSI)
    periods = np.array([60.0, 120.0, 200.0, 240.0, 290.0, 480.0, 900.0])
    power = [detect._profile_power_at_period(T, y, SIG, P, cp, sp, np.linspace(0, 0.6, 4), 16)[0]
             for P in periods]
    assert periods[int(np.argmax(power))] == 240.0
    assert max(power) > 0.5


def test_linear_amplitudes_are_recovered_at_the_true_nonlinear_parameters():
    P, e, T0 = 240.0, 0.2, 30.0
    ti = orbit.ti_from_campbell(0.5, 1.0, 0.7, 1.0)
    y = orbit.orbital_model_AL(T, [P, e, T0, *ti], PSI)
    E = orbit.kepler_eq(2 * np.pi * (T - T0) / P, e)
    chi2, coeffs = detect._chi2_reduction(y, 1.0 / SIG ** 2, np.cos(E) - e,
                                          np.sqrt(1 - e ** 2) * np.sin(E), np.cos(PSI), np.sin(PSI))
    assert chi2 < 1e-12 and np.allclose(coeffs, ti, atol=1e-8)
