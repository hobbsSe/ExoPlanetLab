"""Priors of the orbital elements (priors.py)."""

import numpy as np
import pytest

import orbit
import priors

A0_MIN, FLOOR = 1e-3, np.sin(np.radians(2.0)) ** 2
BOUNDS = dict(p_min=1.0, p_max=7305.0, e_min=1e-6, t0_lo=-8000.0, t0_hi=8000.0, ti_max=500.0,
              a0_min=A0_MIN, sin2_floor=FLOOR)


def _ti(a0, inc_deg, Om=0.7, om=1.9):
    return orbit.ti_from_campbell(a0, om, Om, np.radians(inc_deg))


def test_ti_prior_is_flat():
    assert priors.orient_log_prior(*_ti(0.3, 50.0), 'ti', A0_MIN, FLOOR) == 0.0
    out = priors.orient_log_prior(*(np.array([v, v]) for v in _ti(0.3, 50.0)), 'ti', A0_MIN, FLOOR)
    assert np.all(out == 0.0) and out.shape == (2,)


@pytest.mark.parametrize('a0, inc', [(0.01, 30.0), (0.5, 60.0), (3.0, 90.0), (20.0, 135.0)])
def test_isotropic_density_is_a0_to_minus_4_over_sin2_i(a0, inc):
    lp = priors.orient_log_prior(*_ti(a0, inc), 'isotropic', A0_MIN, FLOOR)
    assert lp == pytest.approx(-4 * np.log(a0) - np.log(np.sin(np.radians(inc)) ** 2), abs=1e-9)


def test_isotropic_density_depends_on_a0_and_i_only():
    ref = priors.orient_log_prior(*_ti(0.7, 40.0, 0.1, 0.2), 'isotropic', A0_MIN, FLOOR)
    for Om, om in ((1.0, 2.0), (3.0, 5.5), (6.0, 0.4)):
        assert priors.orient_log_prior(*_ti(0.7, 40.0, Om, om), 'isotropic', A0_MIN,
                                       FLOOR) == pytest.approx(ref, abs=1e-9)


def test_isotropic_floor_and_lower_bound():
    face_on = priors.orient_log_prior(*_ti(1.0, 0.0), 'isotropic', A0_MIN, FLOOR)
    assert face_on == pytest.approx(-np.log(FLOOR), abs=1e-9)            # finite, floored
    assert priors.orient_log_prior(*_ti(0.5 * A0_MIN, 60.0), 'isotropic', A0_MIN, FLOOR) == -np.inf
    arr = priors.orient_log_prior(*(np.array([x, y]) for x, y in
                                    zip(_ti(0.5 * A0_MIN, 60.0), _ti(1.0, 60.0))),
                                  'isotropic', A0_MIN, FLOOR)
    assert arr[0] == -np.inf and np.isfinite(arr[1])


def test_isotropic_normalisation_against_direct_integration():
    # A small box and a large a0_min, so that plain Monte Carlo in (A, B, F, G)
    # converges: the integral of the density over the box must equal iso_log_norm.
    ti_max, a0_min, floor = 1.0, 0.3, 0.05
    rng = np.random.default_rng(1)
    x = rng.uniform(-ti_max, ti_max, (4, 2_000_000))
    lp = priors.orient_log_prior(*x, 'isotropic', a0_min, floor)
    direct = np.log((2 * ti_max) ** 4 * np.mean(np.where(np.isfinite(lp), np.exp(lp), 0.0)))
    assert priors.iso_log_norm(ti_max, a0_min, floor, n=1_000_000) == pytest.approx(direct, abs=0.02)


def test_orbits_separated():
    assert priors.orbits_separated(100.0, 0.0, 119.0, 0.0, 'ratio', 1.2) == False   # noqa: E712
    assert priors.orbits_separated(100.0, 0.0, 121.0, 0.0, 'ratio', 1.2) == True    # noqa: E712
    assert priors.orbits_separated(100.0, 0.9, 101.0, 0.9, 'none', 1.2) == True     # noqa: E712
    # non-crossing: a_b (1 - e_b) > a_a (1 + e_a), a ~ P^(2/3)
    assert priors.orbits_separated(100.0, 0.1, 100.0 * 1.3 ** 1.5, 0.1, 'noncrossing', 1.2) == True   # noqa: E712
    assert priors.orbits_separated(100.0, 0.5, 200.0, 0.5, 'noncrossing', 1.2) == False               # noqa: E712
    out = priors.orbits_separated(np.array([100.0, 100.0]), 0.0, np.array([110.0, 300.0]), 0.0,
                                  'ratio', 1.2)
    assert list(out) == [False, True]


def test_separation_normalisation_matches_the_analytic_values():
    assert priors.sep_log_norm(1, 1.0, 7305.0, 1e-6, 'ratio', 1.2) == 0.0
    # ordering only: half of the pairs
    assert priors.sep_log_norm(2, 1.0, 7305.0, 1e-6, 'none', 1.2, n=400_000) == pytest.approx(
        np.log(0.5), abs=0.01)
    # ratio r: ln P uniform on a range L -> probability (L - ln r)^2 / (2 L^2)
    L, r = np.log(7305.0), 1.2
    assert priors.sep_log_norm(2, 1.0, 7305.0, 1e-6, 'ratio', r, n=400_000) == pytest.approx(
        np.log((L - np.log(r)) ** 2 / (2 * L * L)), abs=0.01)
    # three orbits, ordering only: 1 / 3!
    assert priors.sep_log_norm(3, 1.0, 7305.0, 1e-6, 'none', 1.2, n=400_000) == pytest.approx(
        np.log(1.0 / 6.0), abs=0.02)


def test_log_prior_orbit_bounds_and_period_prior():
    ti = _ti(0.3, 50.0)
    good = np.array([300.0, 0.3, 10.0, *ti])
    lp = priors.log_prior_orbit(good, orient_prior='ti', **BOUNDS)
    assert lp == pytest.approx(-np.log(300.0))
    assert priors.log_prior_orbit(good, orient_prior='ti', jeffreys_p=False, **BOUNDS) == 0.0
    iso = priors.log_prior_orbit(good, orient_prior='isotropic', **BOUNDS)
    assert iso == pytest.approx(-np.log(300.0) - 4 * np.log(0.3) - np.log(np.sin(np.radians(50)) ** 2))
    for k, bad in ((0, 0.5), (0, 8000.0), (1, 1.0), (1, -0.1), (2, 9000.0), (3, 600.0), (6, -600.0)):
        p = good.copy()
        p[k] = bad
        assert priors.log_prior_orbit(p, orient_prior='ti', **BOUNDS) == -np.inf


def test_single_orbit_normalisation():
    # 'ti': product of the one-dimensional integrals
    ln = priors.log_norm_orbit(orient_prior='ti', **BOUNDS)
    assert ln == pytest.approx(np.log(np.log(7305.0)) + np.log(1 - 1e-6) + np.log(16000.0)
                               + 4 * np.log(1000.0))
    # the period part by quadrature of exp(log_prior_orbit)
    P = np.exp(np.linspace(np.log(1.0), np.log(7305.0), 20001)[1:-1])
    ti = _ti(0.3, 50.0)
    dens = np.exp([priors.log_prior_orbit(np.array([p, 0.3, 0.0, *ti]), orient_prior='ti', **BOUNDS)
                   for p in P])
    assert np.trapezoid(dens * P, np.log(P)) == pytest.approx(np.log(7305.0), rel=2e-4)
