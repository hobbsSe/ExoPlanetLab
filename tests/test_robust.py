# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""robust.py: the Student-t likelihood with marginalised single-star terms."""

import numpy as np
import pytest
from scipy.special import gammaln

import robust

RNG = np.random.default_rng(2)
N = 150
SIG = RNG.uniform(0.05, 0.2, N)
X = np.column_stack([RNG.normal(size=N), RNG.normal(size=N), RNG.uniform(-1, 1, N)])
LAM = np.array([0.0, 0.0, 4.0])
D = X @ [0.3, -0.2, 0.1] + SIG * RNG.standard_normal(N)


def _gauss_marginal(d, f=1.0):
    """Exact ln of the Gaussian likelihood integrated over beta (flat prior, or
    Gaussian of precision LAM), with the constants robust.loglike keeps."""
    s2 = (f * SIG) ** 2
    A = X.T @ (X / s2[:, None]) + np.diag(LAM)
    b = np.linalg.solve(A, X.T @ (d / s2))
    r = d - X @ b
    return (-0.5 * np.sum(r * r / s2) - 0.5 * np.sum(np.log(2 * np.pi * s2))
            - 0.5 * b @ (LAM * b) - 0.5 * np.linalg.slogdet(A)[1])


def test_gaussian_limit_is_the_exact_marginal_likelihood():
    rob = robust.StudentTSingleStar(D, X, SIG, LAM)
    m = RNG.normal(0, 0.05, (4, N))
    for f in (1.0, 1.3):
        ll = rob.loglike(m, np.full(4, 1e9), np.full(4, f))
        ref = np.array([_gauss_marginal(D - mi, f) for mi in m])
        assert np.allclose(ll, ref, atol=2e-3)


def test_single_star_terms_in_the_model_do_not_change_the_likelihood():
    """A model that is a combination of unconstrained design columns is
    absorbed by the refitted single-star terms."""
    rob = robust.StudentTSingleStar(D, X, SIG, LAM)
    nu, f = np.array([3.0]), np.array([1.1])
    base = rob.loglike(np.zeros((1, N)), nu, f)
    shifted = rob.loglike((X[:, :2] @ [5.0, -3.0])[None, :], nu, f)
    assert shifted[0] == pytest.approx(base[0], abs=1e-6)


def test_without_single_star_terms_it_is_the_plain_student_t_density():
    rob = robust.StudentTSingleStar(D, np.zeros((N, 0)), SIG)
    nu, f = 4.0, 0.9
    s2 = (f * SIG) ** 2
    ref = np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * s2)
                 - (nu + 1) / 2 * np.log1p(D ** 2 / (nu * s2)))
    assert rob.loglike(np.zeros((1, N)), np.array([nu]), np.array([f]))[0] == pytest.approx(ref)


def test_outlier_costs_less_than_in_the_gaussian_model():
    d = D.copy()
    d[7] += 40 * SIG[7]
    rob0, rob1 = (robust.StudentTSingleStar(v, X, SIG, LAM) for v in (D, d))
    z = np.zeros((1, N))
    drop_t = (rob0.loglike(z, [3.0], [1.0]) - rob1.loglike(z, [3.0], [1.0]))[0]
    drop_g = (rob0.loglike(z, [1e9], [1.0]) - rob1.loglike(z, [1e9], [1.0]))[0]
    assert 0 < drop_t < 0.05 * drop_g


def test_rows_are_independent():
    rob = robust.StudentTSingleStar(D, X, SIG, LAM)
    m = RNG.normal(0, 0.05, (5, N))
    nu, f = RNG.uniform(2, 20, 5), RNG.uniform(0.8, 1.2, 5)
    all_ = rob.loglike(m, nu, f)
    for j in range(5):
        assert rob.loglike(m[j:j + 1], nu[j:j + 1], f[j:j + 1])[0] == pytest.approx(all_[j], abs=1e-8)


def test_log_evidence_against_a_fine_grid():
    rob = robust.StudentTSingleStar(D, X, SIG, LAM)
    lz = rob.log_evidence(np.zeros(N), (1.0, 100.0), (0.1, 10.0))
    g0, g1 = np.linspace(0, np.log(100), 220), np.linspace(np.log(0.6), np.log(1.8), 220)
    A, B = np.meshgrid(np.exp(g0), np.exp(g1), indexing='ij')
    ll = np.concatenate([rob.loglike(np.zeros((len(a), N)), a, b)
                         for a, b in zip(np.array_split(A.ravel(), 50),
                                         np.array_split(B.ravel(), 50))]).reshape(A.shape)
    w0, w1 = np.gradient(g0), np.gradient(g1)
    w0[[0, -1]] *= 0.5
    w1[[0, -1]] *= 0.5
    ref = (np.log(np.sum(np.exp(ll - ll.max()) * np.outer(w0, w1))) + ll.max()
           - np.log(np.log(100.0)) - np.log(np.log(100.0)))
    assert lz == pytest.approx(ref, abs=0.02)


def test_beta_draws_scatter_about_the_fit():
    rob = robust.StudentTSingleStar(D, X, SIG, LAM)
    m = np.zeros((4000, N))
    draws = rob.beta_draws(m, np.full(4000, 1e9), np.ones(4000), np.random.default_rng(0))
    s2 = SIG ** 2
    A = X.T @ (X / s2[:, None]) + np.diag(LAM)
    b = np.linalg.solve(A, X.T @ (D / s2))
    assert np.allclose(draws.mean(0), b, atol=4 * np.sqrt(np.diag(np.linalg.inv(A)) / 4000) + 1e-4)
    assert np.allclose(draws.std(0), np.sqrt(np.diag(np.linalg.inv(A))), rtol=0.08)


def test_two_noise_groups_reduce_to_one_when_equal():
    g = np.r_[np.zeros(N // 2, int), np.ones(N - N // 2, int)]
    one = robust.StudentTSingleStar(D, X, SIG, LAM)
    two = robust.StudentTSingleStar(D, X, SIG, LAM, groups=g)
    z = np.zeros((1, N))
    assert two.loglike(z, np.array([[3.0, 3.0]]), np.array([[1.1, 1.1]]))[0] == pytest.approx(
        one.loglike(z, [3.0], [1.1])[0], abs=1e-8)
