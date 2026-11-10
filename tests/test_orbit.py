# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""orbit.py: Kepler solver, Thiele-Innes / Campbell conversion, along-scan
model, RV model, single-star projection."""

import numpy as np
import pytest

import orbit
from generate_sources import campbell_elements

RNG = np.random.default_rng(1)
T = np.sort(RNG.uniform(-900.0, 900.0, 200))
PSI = RNG.uniform(0.0, 2.0 * np.pi, 200)


@pytest.mark.parametrize('e', [0.0, 0.1, 0.5, 0.9, 0.99, 0.9999])
def test_kepler_equation_is_solved(e):
    M = np.linspace(-20.0, 20.0, 4001)
    E = orbit.kepler_eq(M, e)
    assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-9


def test_kepler_sin_cos_variant_agrees():
    M = np.linspace(0.0, 2.0 * np.pi, 1001)
    for e in (0.0, 0.3, 0.8):
        E, sE, cE = orbit.kepler_eq_sc(M, e)
        E0 = orbit.kepler_eq(M, e)
        assert np.max(np.abs(np.sin(E0) - sE)) < 1e-9
        assert np.max(np.abs(np.cos(E0) - cE)) < 1e-9


def test_thiele_innes_campbell_round_trip():
    """a0, i and the (omega, Omega) pair with Omega in [0, pi) come back, and
    the pair reproduces the Thiele-Innes constants (not their negatives)."""
    for _ in range(500):
        a0 = 10 ** RNG.uniform(-2, 2)
        om, Om = RNG.uniform(0, 2 * np.pi), RNG.uniform(0, np.pi)
        inc = np.arccos(RNG.uniform(-0.99, 0.99))
        ti = np.array(orbit.ti_from_campbell(a0, om, Om, inc))
        a1, om1, Om1, inc1 = campbell_elements(*ti)
        assert a1 == pytest.approx(a0, rel=1e-9)
        assert inc1 == pytest.approx(inc, abs=1e-7)
        assert 0.0 <= Om1 < np.pi and 0.0 <= om1 < 2 * np.pi
        assert np.cos(om1 - om) == pytest.approx(1.0, abs=1e-9)
        assert np.cos(Om1 - Om) == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(orbit.ti_from_campbell(a1, om1, Om1, inc1), ti, rtol=1e-8,
                           atol=1e-10)


def test_campbell_elements_work_on_arrays():
    om, Om = RNG.uniform(0, 2 * np.pi, 300), RNG.uniform(0, np.pi, 300)
    inc = np.arccos(RNG.uniform(-0.99, 0.99, 300))
    ti = orbit.ti_from_campbell(1.0, om, Om, inc)
    a1, om1, Om1, inc1 = campbell_elements(*ti)
    assert np.allclose(np.cos(om1 - om), 1.0) and np.allclose(np.cos(Om1 - Om), 1.0)


def test_omega_from_ti_matches_campbell():
    for _ in range(500):
        ti = orbit.ti_from_campbell(1.3, RNG.uniform(0, 2 * np.pi), RNG.uniform(0, np.pi),
                                    np.arccos(RNG.uniform(-0.99, 0.99)))
        assert np.cos(orbit.omega_from_ti(*ti) - campbell_elements(*ti)[1]) == pytest.approx(
            1.0, abs=1e-9)


def test_the_other_node_gives_the_same_orbit_and_the_opposite_velocity():
    """(omega + pi, Omega + pi) has the same Thiele-Innes constants; the RV
    with omega + pi is the negative: only RVs tell the two apart."""
    a = np.array(orbit.ti_from_campbell(1.0, 0.8, 0.5, 1.1))
    b = np.array(orbit.ti_from_campbell(1.0, 0.8 + np.pi, 0.5 + np.pi, 1.1))
    assert np.allclose(a, b, atol=1e-12)


def test_orbit_is_periodic_in_time_and_in_T0():
    P, e, T0 = 137.0, 0.4, 12.0
    ti = orbit.ti_from_campbell(2.0, 0.3, 1.0, 1.2)
    a = orbit.astrometric_orbit(T, P, e, T0, *ti)
    b = orbit.astrometric_orbit(T + 3 * P, P, e, T0, *ti)
    c = orbit.astrometric_orbit(T, P, e, T0 - 5 * P, *ti)
    assert np.allclose(a, b, atol=1e-9) and np.allclose(a, c, atol=1e-9)


def test_orbit_at_periastron_and_size():
    """At T0 the star is at X = 1 - e, Y = 0; the orbit never exceeds a0 (1 + e)."""
    P, e, T0, a0 = 300.0, 0.6, 50.0, 3.0
    A, B, F, G = orbit.ti_from_campbell(a0, 0.4, 0.9, 0.5)
    dra, ddec = orbit.astrometric_orbit(np.array([T0]), P, e, T0, A, B, F, G)
    assert dra[0] == pytest.approx(B * (1 - e)) and ddec[0] == pytest.approx(A * (1 - e))
    dra, ddec = orbit.astrometric_orbit(np.linspace(0, P, 2000), P, e, T0, A, B, F, G)
    assert np.max(np.hypot(dra, ddec)) <= a0 * (1 + e) + 1e-9


def test_along_scan_model_is_the_projection_of_the_sky_orbit():
    par = [211.0, 0.35, -40.0, *orbit.ti_from_campbell(1.5, 2.0, 0.4, 1.0)]
    dra, ddec = orbit.astrometric_orbit(T, *par)
    al = orbit.orbital_model_AL(T, par, PSI)
    assert np.allclose(al, dra * np.sin(PSI) + ddec * np.cos(PSI), atol=1e-12)


def test_vectorised_models_equal_the_scalar_ones():
    rows = np.array([[211.0, 0.35, -40.0, *orbit.ti_from_campbell(1.5, 2.0, 0.4, 1.0)],
                     [33.0, 0.0001, 5.0, *orbit.ti_from_campbell(0.2, 0.1, 2.4, 2.9)],
                     [5000.0, 0.9, 800.0, *orbit.ti_from_campbell(20.0, 4.0, 1.4, 0.2)]])
    vec = orbit.orbital_model_AL_vec(T, rows, PSI)
    for j, r in enumerate(rows):
        assert np.allclose(vec[j], orbit.orbital_model_AL(T, r, PSI), atol=1e-9)
    dra, ddec = orbit.astrometric_orbit_vec(T, rows)
    for j, r in enumerate(rows):
        a, b = orbit.astrometric_orbit(T, *r)
        assert np.allclose(dra[j], a, atol=1e-9) and np.allclose(ddec[j], b, atol=1e-9)


def test_rv_models_agree_and_have_the_right_amplitude():
    P, e, T0, K, om = 80.0, 0.3, 4.0, 500.0, 1.2
    one = orbit.rv_model_single(T, P, e, T0, K, om)
    vec = orbit.rv_model_vec(T, [P], [e], [T0], [K], [om])[0]
    assert np.allclose(one, vec, atol=1e-8)
    full = orbit.rv_model_single(np.linspace(0, P, 20001), P, e, T0, K, om)
    assert full.max() - full.min() == pytest.approx(2 * K, rel=1e-4)
    # omega -> omega + pi reverses the velocity
    assert np.allclose(orbit.rv_model_single(T, P, e, T0, K, om + np.pi), -one, atol=1e-8)


def test_relative_orbit_is_the_scaled_opposite_of_the_photocentre():
    rows = np.array([[211.0, 0.35, -40.0, *orbit.ti_from_campbell(1.5, 2.0, 0.4, 1.0)]])
    dra, ddec = orbit.astrometric_orbit_vec(T, rows)
    xr, yr = orbit.relative_orbit_vec(T, rows, [50.0])
    assert np.allclose(xr, -50.0 * dra) and np.allclose(yr, -50.0 * ddec)


def _projector(n=200, lam_last=0.0):
    tau = T / 365.25
    X = np.column_stack([np.sin(PSI), np.cos(PSI), RNG.uniform(-1, 1, n),
                         tau * np.sin(PSI), tau * np.cos(PSI)])
    w = 1.0 / RNG.uniform(0.05, 0.2, n) ** 2
    L = np.zeros(5)
    L[-1] = lam_last
    gain = np.linalg.solve(X.T @ (w[:, None] * X) + np.diag(L), X.T * w[None, :])
    return X, w, orbit.SingleStarProjector(X, gain)


def test_relative_astrometry_likelihood_with_jitter():
    """The likelihood of the relative positions is a product of 2-D Gaussians
    whose covariance gains jitter^2 on the diagonal only."""
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(11)
    n = 7
    sx, sy = rng.uniform(0.5, 3.0, n), rng.uniform(0.5, 3.0, n)
    rho = rng.uniform(-0.8, 0.8, n)
    dx, dy = rng.normal(0, 3, (3, n)), rng.normal(0, 3, (3, n))
    jit = np.array([0.0, 0.7, 4.0])
    got = orbit.relative_astrometry_lnlike(dx, dy, sx, sy, rho, jit)
    for k in range(3):
        want = sum(multivariate_normal.logpdf(
            [dx[k, i], dy[k, i]], mean=[0.0, 0.0],
            cov=[[sx[i] ** 2 + jit[k] ** 2, rho[i] * sx[i] * sy[i]],
                 [rho[i] * sx[i] * sy[i], sy[i] ** 2 + jit[k] ** 2]]) for i in range(n))
        assert got[k] == pytest.approx(want, rel=1e-12)
    # one set of residuals and a scalar jitter; no jitter is the default
    assert orbit.relative_astrometry_lnlike(dx[1], dy[1], sx, sy, rho, 0.7) == pytest.approx(got[1])
    assert orbit.relative_astrometry_lnlike(dx[0], dy[0], sx, sy, rho) == pytest.approx(got[0])


def test_relative_astrometry_jitter_is_recovered_at_its_maximum():
    """Residuals with errors larger than quoted: the likelihood peaks at the
    missing jitter, sqrt(true^2 - quoted^2)."""
    rng = np.random.default_rng(12)
    n, quoted, true = 4000, 0.5, 2.0
    dx, dy = rng.normal(0, true, n), rng.normal(0, true, n)
    s = np.full(n, quoted)
    grid = np.linspace(1.5, 2.4, 181)
    ll = orbit.relative_astrometry_lnlike(np.tile(dx, (len(grid), 1)), np.tile(dy, (len(grid), 1)),
                                          s, s, np.zeros(n), grid)
    assert grid[np.argmax(ll)] == pytest.approx(np.sqrt(true ** 2 - quoted ** 2), abs=0.05)


def test_projector_removes_the_single_star_terms_and_is_idempotent():
    X, w, proj = _projector()
    beta = RNG.normal(size=5)
    assert np.max(np.abs(proj.apply(X @ beta))) < 1e-9
    v = RNG.normal(size=len(T))
    r = proj.apply(v)
    assert np.allclose(proj.apply(r), r, atol=1e-9)
    # the residual is orthogonal (in the fit's weights) to every design column
    assert np.max(np.abs(X.T @ (w * r))) < 1e-7
    # matrices are projected column by column
    M = RNG.normal(size=(len(T), 3))
    assert np.allclose(proj.apply(M)[:, 1], proj.apply(M[:, 1]))


def test_projector_rejects_other_lengths():
    _, _, proj = _projector()
    with pytest.raises(ValueError):
        proj.apply(np.zeros(len(T) - 1))


def test_projected_model_equals_fit_residual_of_the_model():
    X, w, proj = _projector()
    par = [400.0, 0.2, 10.0, *orbit.ti_from_campbell(1.0, 0.5, 1.5, 1.0)]
    m = orbit.orbital_model_AL(T, par, PSI)
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * m))
    assert np.allclose(orbit.orbital_model_AL(T, par, PSI, projector=proj), m - X @ beta,
                       atol=1e-9)


def test_coupled_k_density_is_normalised_and_peaks_at_the_implied_k():
    """coupled_k_lnterm: p(K) * k_range integrates to k_range over both signs
    of K, and sits at K = c / parallax."""
    c = orbit.implied_rv_scale(576.0, 0.33, 0.25, np.radians(122.0))
    plx, sd, k_range = 13.6, 0.05, 1.0e4
    K = np.linspace(-5000.0, 5000.0, 2_000_001)
    dens = np.exp(orbit.coupled_k_lnterm(K, c, plx, sd, k_range))
    assert np.isclose(np.sum(dens) * (K[1] - K[0]), k_range, rtol=1e-4)
    assert np.isclose(abs(K[np.argmax(dens)]), c / plx, rtol=1e-3)
    assert orbit.coupled_k_lnterm(0.0, c, plx, sd, k_range) == -np.inf
    # the scale itself: a 1 mas orbit at 10 mas parallax, edge-on, P = 1 yr, e = 0
    # moves the star on a 0.1 AU circle: K = 2 pi 0.1 AU / yr = 2.98 km/s
    assert np.isclose(orbit.implied_rv_scale(365.25, 0.0, 1.0, np.pi / 2) / 10.0, 2978.5, rtol=1e-3)
