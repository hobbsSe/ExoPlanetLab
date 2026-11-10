# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""accel.py: acceleration and scan-angle harmonic terms and their evidences."""

import numpy as np
import pytest
from scipy.special import logsumexp

import accel
import orbit
import robust

RNG = np.random.default_rng(3)
N = 240
T = np.sort(RNG.uniform(-900.0, 900.0, N))
TAU = T / 365.25
PSI = RNG.uniform(0.0, 2.0 * np.pi, N)
SIG = RNG.uniform(0.05, 0.2, N)
X = np.column_stack([np.sin(PSI), np.cos(PSI), RNG.uniform(-1, 1, N),
                     TAU * np.sin(PSI), TAU * np.cos(PSI)])
IPA, IPD = 3, 4
PM = X[:, [IPA, IPD]]


def test_model_registry_is_consistent():
    for m, pairs in accel.MODELS.items():
        assert len(accel.NAMES[m]) == len(accel.UNITS[m]) == 2 * pairs
        assert len(accel.SCALES[m]) == pairs
        assert accel.columns(m, X, TAU, IPA, IPD, PSI).shape == (N, 2 * pairs)


def test_columns():
    c = accel.columns('accel9', X, TAU, IPA, IPD)
    assert np.allclose(c[:, :2], PM * TAU[:, None] / 2) and np.allclose(c[:, 2:], PM * TAU[:, None] ** 2 / 6)
    h = accel.columns('harm23', X, TAU, IPA, IPD, PSI)
    assert np.allclose(h, np.column_stack([np.cos(2 * PSI), np.sin(2 * PSI),
                                           np.cos(3 * PSI), np.sin(3 * PSI)]))
    with pytest.raises(ValueError):
        accel.columns('harm2', X, TAU, IPA, IPD)


def test_iso2_against_numpy():
    C = RNG.normal(size=(6, 2, 2))
    C = C @ C.transpose(0, 2, 1) + 0.1 * np.eye(2)
    b = RNG.normal(size=(6, 2))
    var = np.array([1e-4, 0.3, 50.0])
    logdet, q = accel._iso2(C, b, var)
    for k in range(6):
        for j, v in enumerate(var):
            M = C[k] + v * np.eye(2)
            assert logdet[k, j] == pytest.approx(np.linalg.slogdet(M)[1], rel=1e-10, abs=1e-10)
            assert q[k, j] == pytest.approx(b[k] @ np.linalg.solve(M, b[k]), rel=1e-9)


@pytest.mark.parametrize('model', sorted(accel.MODELS))
def test_gaussian_limit_equals_a_direct_gaussian_prior(model):
    """For fixed scales the analytic prior term equals putting the Gaussian
    prior in the fit (exactly, in the Gaussian limit)."""
    d = SIG * RNG.standard_normal(N) + (PM * TAU[:, None] / 2) @ [0.04, -0.03]
    rob = robust.StudentTSingleStar(d, X, SIG, np.zeros(5))
    aug = accel.augment(rob, TAU, IPA, IPD, model, psi=PSI)
    k = aug.p - rob.p
    z, nu, f = np.zeros((1, N)), np.array([1e9]), np.array([1.0])
    l, beta, H = aug._terms(z, nu, f)
    C, b = np.linalg.inv(H[0])[rob.p:, rob.p:], beta[0, rob.p:]
    for s in (0.003, 0.05, 2.0):
        sv = np.array([s, s, 2 * s, 2 * s])[:k]
        M = C + np.diag(sv ** 2)
        mine = l[0] - 0.5 * np.linalg.slogdet(M)[1] - 0.5 * b @ np.linalg.solve(M, b)
        ref = robust.StudentTSingleStar(d, aug.X, SIG, np.r_[rob.lam, 1 / sv ** 2]).loglike(
            z, nu, f)[0] + 0.5 * np.sum(np.log(1 / sv ** 2))
        assert mine == pytest.approx(ref, abs=1e-6)


def test_two_pair_mixture_against_brute_force():
    C = RNG.normal(size=(3, 4, 4))
    C = C @ C.transpose(0, 2, 1) * 1e-3 + 1e-4 * np.eye(4)
    b = RNG.normal(0, 0.05, size=(3, 4))
    mix = accel._mix('accel9')(C, b)
    va, lwa = accel._scale_grid(*accel.SCALES['accel9'][0])
    vj, lwj = accel._scale_grid(*accel.SCALES['accel9'][1])
    for k in range(3):
        terms = []
        for a, la in zip(va, lwa):
            for j, lj in zip(vj, lwj):
                M = C[k] + np.diag([a, a, j, j])
                terms.append(-0.5 * np.linalg.slogdet(M)[1] - 0.5 * b[k] @ np.linalg.solve(M, b[k])
                             + la + lj)
        assert mix[k] == pytest.approx(logsumexp(terms), abs=1e-8)


def test_injected_acceleration_is_recovered_and_preferred():
    d = SIG * RNG.standard_normal(N) + (PM * TAU[:, None] / 2) @ [0.08, -0.05]
    rob = robust.StudentTSingleStar(d, X, SIG, np.zeros(5))
    est = accel.estimate(rob, TAU, IPA, IPD, 'accel7', (1.0, 100.0), (0.1, 10.0))
    assert np.all(np.abs(est['value'] - [0.08, -0.05]) < 4 * est['error'])
    lz0 = rob.log_evidence(np.zeros(N), (1.0, 100.0), (0.1, 10.0))
    lz = accel.log_evidence(rob, TAU, IPA, IPD, 'accel7', (1.0, 100.0), (0.1, 10.0))
    assert lz - lz0 > 5


def test_no_acceleration_is_not_preferred_and_costs_little():
    d = SIG * RNG.standard_normal(N)
    rob = robust.StudentTSingleStar(d, X, SIG, np.zeros(5))
    lz0 = rob.log_evidence(np.zeros(N), (1.0, 100.0), (0.1, 10.0))
    for model in ('accel7', 'harm2'):
        dz = accel.log_evidence(rob, TAU, IPA, IPD, model, (1.0, 100.0), (0.1, 10.0),
                                psi=PSI) - lz0
        assert -4.0 < dz < 3.0


def test_gauss_accel_against_brute_force_integration():
    gain = np.linalg.solve(X.T @ (X / SIG[:, None] ** 2), X.T / SIG[None, :] ** 2)
    proj = orbit.SingleStarProjector(X, gain)
    Y = accel.projected_columns(X, TAU, IPA, IPD, 'accel7', proj)
    r = proj.apply(SIG * RNG.standard_normal(N) + (PM * TAU[:, None] / 2) @ [0.03, -0.02])
    mine = accel.GaussAccel(Y, 'accel7').loglike(r[None], SIG ** 2)[0]
    b, C, _ = accel.gauss_terms(r[None], SIG ** 2, Y)
    e = np.sqrt(np.diag(C[0]))
    g = np.linspace(-8, 8, 121)
    A1, A2 = np.meshgrid(b[0, 0] + g * e[0], b[0, 1] + g * e[1], indexing='ij')
    aa = np.column_stack([A1.ravel(), A2.ravel()])
    ll = -0.5 * np.sum((r[None, :] - aa @ Y.T) ** 2 / SIG ** 2 + np.log(SIG ** 2), axis=1)
    da = (g[1] - g[0]) ** 2 * e[0] * e[1]
    va, lwa = accel._scale_grid(*accel.SCALES['accel7'][0])
    tot = [logsumexp(ll - 0.5 * np.sum(aa ** 2, axis=1) / v - np.log(2 * np.pi * v)) + np.log(da)
           for v in va]
    assert mine == pytest.approx(logsumexp(np.array(tot) + lwa), abs=1e-3)
