# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""solver_ptemcee: the evidence estimators, on a problem with a known ln Z.

One parameter, prior N(0, P^2), likelihood exp(-x^2 / 2 s^2): the tempered
posterior at inverse temperature beta is Gaussian, so exact samples of every
temperature can be drawn, and Z = s / sqrt(s^2 + P^2)."""

import numpy as np
import pytest

pytest.importorskip('eryn')
import solver_ptemcee as sp      # noqa: E402

S, P = 0.1, 10.0
EXACT = np.log(S / np.sqrt(S ** 2 + P ** 2))


def _tempered(betas, n=3000, w=64, seed=1):
    rng = np.random.default_rng(seed)
    logl = np.empty((n, len(betas), w))
    for k, b in enumerate(betas):
        var = 1.0 / (b / S ** 2 + 1.0 / P ** 2)
        logl[:, k, :] = -0.5 * rng.normal(0, np.sqrt(var), (n, w)) ** 2 / S ** 2
    return logl


FINE = np.r_[np.geomspace(1, 1e-6, 15), 0.0]
COARSE = np.r_[np.geomspace(1, 1e-6, 4), 0.0]


def test_bridge_and_stepping_stone_recover_the_evidence():
    L = _tempered(FINE)
    z, err, blocks = sp.evidence_bridge(FINE, L)
    assert z == pytest.approx(EXACT, abs=0.03) and err < 0.02 and len(blocks) == 8
    assert sp.evidence_estimates(FINE, L, repeats=5)[0] == pytest.approx(EXACT, abs=0.03)
    d = sp.evidence_diagnostics(FINE, L)
    assert d['forward'] == pytest.approx(d['reverse'], abs=0.03)
    assert len(d['rungs']) == len(FINE) - 1


def test_bridge_survives_poorly_overlapping_rungs():
    """With five rungs the reverse stepping-stone is off by more than 1; the
    bridge estimate, which uses both temperatures of a rung, is not."""
    L = _tempered(COARSE)
    d = sp.evidence_diagnostics(COARSE, L)
    assert abs(d['reverse'] - EXACT) > 0.5
    assert sp.evidence_bridge(COARSE, L)[0] == pytest.approx(EXACT, abs=0.05)


def test_the_order_of_the_temperatures_does_not_matter():
    L = _tempered(FINE)
    o = np.random.default_rng(0).permutation(len(FINE))
    assert sp.evidence_bridge(FINE[o], L[:, o, :])[0] == pytest.approx(
        sp.evidence_bridge(FINE, L)[0], abs=1e-9)


def test_halves_agree_for_stationary_chains_and_not_for_drifting_ones():
    L = _tempered(FINE)
    h1, h2 = sp.evidence_halves(FINE, L)
    assert abs(h2 - h1) < sp.EVIDENCE_TOL
    # a temperature that has not equilibrated: its first half sits in a state of
    # much lower likelihood
    D = L.copy()
    D[:len(D) // 2, 5, :] -= 400.0
    h1, h2 = sp.evidence_halves(FINE, D)
    assert abs(h2 - h1) > sp.EVIDENCE_TOL


def test_store_thinning_keeps_the_store_under_its_limit():
    T = sp._store_thin(210000, 32, 64, 17)
    assert 210000 / T * 32 * 64 * 20 * 8 / 1e9 <= sp.PT_STORE_MAX_GB
    assert sp._store_thin(1000, 4, 64, 9) == 1
