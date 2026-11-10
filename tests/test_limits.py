"""Upper limits on undetected companions (limits.py)."""

import numpy as np
import pytest
from scipy.stats import norm

import limits
import orbit
from generate_sources import campbell_elements


def _geometry(n=300, seed=3):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(-900.0, 900.0, n))
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    sig = rng.uniform(0.05, 0.15, n)
    tau = t / 365.25
    # position, parallax-like and proper-motion columns
    pf = np.sin(2 * np.pi * tau) * np.sin(psi) + 0.4 * np.cos(2 * np.pi * tau) * np.cos(psi)
    X = np.column_stack([np.sin(psi), np.cos(psi), pf, tau * np.sin(psi), tau * np.cos(psi)])
    return t, psi, sig, X, rng


def test_columns_reproduce_the_orbit_model():
    t, psi, _, _, _ = _geometry()
    p = np.array([310.0, 0.35, 42.0, 0.2, -0.1, 0.05, 0.3])
    model = orbit.orbital_model_AL(t, p, psi, projector=None)
    assert np.allclose(limits.ti_columns(t, psi, p[0], p[1], p[2]) @ p[3:], model, atol=1e-12)


def test_unit_orbits_have_unit_semi_major_axis_and_are_isotropic():
    u = limits.unit_orbits(4000, np.random.default_rng(0))
    a0, _, _, inc = campbell_elements(*u.T)
    assert np.allclose(a0, 1.0, atol=1e-9)
    assert abs(np.mean(np.cos(inc))) < 0.05          # cos i uniform on [-1, 1]
    assert abs(np.mean(np.cos(inc) ** 2) - 1.0 / 3.0) < 0.03


def test_limit_without_signal_is_the_half_normal_quantile():
    # one orientation, constants m = 0: the posterior of a0 is half-normal
    u = np.array([[1.0, 0.0, 0.0, 1.0]]) / 1.0
    C = np.diag([0.01, 0.02, 0.03, 0.04])
    q = float(u[0] @ np.linalg.inv(C) @ u[0])
    for level in (0.68, 0.95):
        r = limits.limit_at_period([np.zeros(4)], [C], u, level=level)
        assert r['a0_limit'] == pytest.approx(norm.ppf(0.5 * (1 + level)) / np.sqrt(q), rel=1e-6)
    assert r['a0_sens'] == pytest.approx(8.0 / np.sqrt(q))
    assert r['s_best'] == 0.0


def test_strong_signal_limit_is_above_the_amplitude():
    u = np.array([[1.0, 0.0, 0.0, 1.0]])
    C = np.diag([1e-4] * 4)
    r = limits.limit_at_period([np.array([0.5, 0.0, 0.0, 0.5])], [C], u, level=0.95)
    # truncated Gaussian centred on 0.5 with sigma = 1 / sqrt(q)
    assert r['a0_limit'] == pytest.approx(0.5 + norm.ppf(0.95) / np.sqrt(2e4), rel=1e-4)
    assert r['ln_b'] > 100


def test_ti_fit_recovers_an_injected_orbit_in_both_noise_models():
    t, psi, sig, X, rng = _geometry()
    p = np.array([280.0, 0.0, 0.0, 0.3, -0.2, 0.1, 0.25])
    y = (orbit.orbital_model_AL(t, p, psi, projector=None) + X @ rng.normal(0, 1, 5)
         + sig * rng.standard_normal(len(t)))
    cols = limits.ti_columns(t, psi, 280.0)
    for noise in ('gauss', 'student'):
        nu, f = limits.noise_fit(y - cols @ p[3:], X, sig, noise=noise)
        m, C = limits.ti_fit(y, X, sig, cols, nu, f)
        z = (m - p[3:]) / np.sqrt(np.diag(C))
        assert np.all(np.abs(z) < 4.0), (noise, z)
        assert 0.8 < f < 1.25


def test_limits_scale_with_the_noise_and_cover_weak_injections():
    t, psi, sig, X, rng = _geometry()
    periods = np.array([100.0, 300.0, 700.0])
    y = sig * rng.standard_normal(len(t))
    r1 = limits.upper_limits(y, X, sig, t, psi, periods, noise='gauss', n_orient=500)
    r2 = limits.upper_limits(3 * y, X, 3 * sig, t, psi, periods, noise='gauss', n_orient=500)
    assert np.allclose(r2['a0_limit'], 3 * r1['a0_limit'], rtol=1e-6)
    assert np.all(r1['a0_limit'] < r1['a0_sens'])       # no signal: limit below S = 8
    # coverage: a0 uniform, circular orbits -- about 5% of truths above the 95% limit
    above, n = 0, 120
    for _ in range(n):
        P, a0 = float(np.exp(rng.uniform(np.log(50), np.log(800)))), rng.uniform(0, 0.1)
        ti = orbit.ti_from_campbell(a0, rng.uniform(0, 6.28), rng.uniform(0, 6.28),
                                    np.arccos(rng.uniform(-1, 1)))
        yy = (orbit.orbital_model_AL(t, np.array([P, 0.0, 0.0, *ti]), psi, projector=None)
              + sig * rng.standard_normal(len(t)))
        above += a0 > limits.upper_limits(yy, X, sig, t, psi, [P], noise='gauss',
                                          n_orient=300)['a0_limit'][0]
    assert above <= 15                                   # 5% expected (6 of 120)


def test_companion_mass_inverts_the_mass_function():
    m_star, plx, P = 1.0, 20.0, 400.0
    for m in (1e-3, 0.05, 0.5):
        a_au = (m ** 3 / (m_star + m) ** 2 * (P / 365.25) ** 2) ** (1.0 / 3.0)
        assert limits.companion_mass(a_au * plx, P, plx, m_star) == pytest.approx(m, rel=1e-8)
