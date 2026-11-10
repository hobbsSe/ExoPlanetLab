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
RJFlow backend: Eryn trans-dimensional sampler with normalising-flow in-model proposals.

Extension of solver_rjmcmc.py that leads the in-model move with a per-source
RealNVP normalising flow trained on the nessai peel samples.  The flows learn
the posterior shape of each individual planet so that in-model proposals land
near the posterior mode with high probability, raising the in-model acceptance
rate above the near-zero values seen with a plain Gaussian move.  The
differential-evolution and Gaussian moves from solver_rjmcmc.py are mixed in
as a local random-walk component.

Architecture
------------
  * RealNVP with alternating coupling layers (6 layers, 64 hidden units per layer).
  * Scale networks output clamped via tanh * 2 to prevent numerical blow-up.
  * Flows are trained in standardised space (subtract mean, divide by std);
    the standardisation Jacobian cancels in the MH ratio.

Proposal mechanism (independence sampler)
-----------------------------------------
  x_prop  ~ flow(·)              (independent of x_curr)
  log_factor = log_flow_std(x_curr_std) - log_flow_std(x_prop_std)
  MH ratio = exp(log_lik(x_prop) + log_prior(x_prop)
                 - log_lik(x_curr) - log_prior(x_curr)
                 + log_factor)

When the flow accurately captures the posterior, log_factor ≈ 0 for typical
samples and the acceptance approaches the posterior ratio (≈ 1 for near-mode
proposals).  Leaves without a trained flow fall back to a symmetric Gaussian
random walk (log_factor = 0).

Public API
----------
run(log_likelihood, log_prior, prior_transform, args, ...) -> SimpleNamespace
    Identical return namespace to solver_rjmcmc.run.

print_diagnostics(result, active_names)
    Same as solver_rjmcmc.print_diagnostics.
"""

import os
import numpy as np
from types import SimpleNamespace
from config import N_MIN, N_MAX, N_POOL

import torch
import torch.nn as nn

from scipy.stats import truncnorm
from scipy.special import logsumexp
from tight_prior import _tight_stats, period_for

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform
from eryn.moves import DistributionGenerateRJ

try:
    from eryn.moves import MHMove
except ImportError:
    from eryn.moves.mhmove import MHMove

from solver_rjmcmc import (_PeriodicPerSourceMove, _PerSourceDEMove,
                           _regularise_cov, _source_covs, _parse_move_mix,
                           _wrap_reflect, _SourceConditioner,
                           _cold_chain_tau, _print_tau)

# Eryn hyperparameters — same defaults and env overrides as solver_rjmcmc.py,
# and for the same reason: burn-in needs enough steps for the Gaussian
# fallback move's adaptive tuning (_PeriodicPerSourceMove.rescale) to converge
# before it's frozen for production.
NWALKERS    = int(os.environ.get('EXOPLANETLAB_RJ_NWALKERS', 50))
NSTEPS_BURN = int(os.environ.get('EXOPLANETLAB_RJ_NSTEPS_BURN', 2000))
NSTEPS_PROD = int(os.environ.get('EXOPLANETLAB_RJ_NSTEPS_PROD', 8000))




# Flow hyperparameters
_N_LAYERS = 6
_HIDDEN   = 64
_N_EPOCHS = 800
_LR       = 3e-3
_BATCH    = 512
_PATIENCE = int(os.environ.get('EXOPLANETLAB_RJ_FLOW_PATIENCE', 40))   # epochs


# ── Normalising flow ──────────────────────────────────────────────────────────

class _CouplingLayer(nn.Module):
    """One RealNVP affine coupling layer with alternating split direction."""

    def __init__(self, ndim, hidden, even):
        super().__init__()
        self.d1   = ndim // 2
        self.d2   = ndim - self.d1
        self.even = even
        n_cond  = self.d1 if even else self.d2
        n_trans = self.d2 if even else self.d1
        self.s_net = nn.Sequential(
            nn.Linear(n_cond, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_trans),
        )
        self.t_net = nn.Sequential(
            nn.Linear(n_cond, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_trans),
        )

    def forward(self, x):
        """x -> (y, log_det_J).  Used for log_prob."""
        x1, x2 = x[:, :self.d1], x[:, self.d1:]
        if self.even:
            s = torch.tanh(self.s_net(x1)) * 2.0
            t = self.t_net(x1)
            y2 = x2 * torch.exp(s) + t
            return torch.cat([x1, y2], dim=1), s.sum(dim=1)
        else:
            s = torch.tanh(self.s_net(x2)) * 2.0
            t = self.t_net(x2)
            y1 = x1 * torch.exp(s) + t
            return torch.cat([y1, x2], dim=1), s.sum(dim=1)

    def inverse(self, y):
        """y -> x.  Used for sample()."""
        y1, y2 = y[:, :self.d1], y[:, self.d1:]
        if self.even:
            s = torch.tanh(self.s_net(y1)) * 2.0
            t = self.t_net(y1)
            x2 = (y2 - t) * torch.exp(-s)
            return torch.cat([y1, x2], dim=1)
        else:
            s = torch.tanh(self.s_net(y2)) * 2.0
            t = self.t_net(y2)
            x1 = (y1 - t) * torch.exp(-s)
            return torch.cat([x1, y2], dim=1)


class _RealNVP(nn.Module):
    """Stack of alternating coupling layers forming a RealNVP flow."""

    def __init__(self, ndim, n_layers=_N_LAYERS, hidden=_HIDDEN):
        super().__init__()
        self.ndim   = ndim
        self.layers = nn.ModuleList(
            [_CouplingLayer(ndim, hidden, even=(i % 2 == 0))
             for i in range(n_layers)]
        )
        self._log2pi = float(np.log(2.0 * np.pi))

    def log_prob(self, x):
        """Log-probability of x under the flow; x has shape (batch, ndim)."""
        z = x
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in self.layers:
            z, ld = layer(z)
            log_det += ld
        log_pz = -0.5 * (z ** 2 + self._log2pi).sum(dim=1)
        return log_pz + log_det

    def sample(self, n):
        """Draw n samples from the flow; returns tensor of shape (n, ndim)."""
        with torch.no_grad():
            z = torch.randn(n, self.ndim)
            x = z
            for layer in reversed(self.layers):
                x = layer.inverse(x)
        return x


def _to_std(x, mu, scale):
    """Physical -> standardised space; scale is a vector of spreads or the
    lower-triangular factor L of the covariance (x - mu = z L^T)."""
    if scale.ndim == 1:
        return (x - mu) / scale
    return np.linalg.solve(scale, (x - mu).T).T


def _from_std(z, mu, scale):
    """Standardised -> physical space (inverse of _to_std)."""
    if scale.ndim == 1:
        return z * scale + mu
    return z @ scale.T + mu


def _train_flow(samples, n_layers=_N_LAYERS, hidden=_HIDDEN,
                n_epochs=_N_EPOCHS, lr=_LR, batch_size=_BATCH,
                print_every=0):
    """Standardise samples, train a RealNVP, return (flow, mu, scale).

    The flow is trained in standardised space: the mean is subtracted and the
    linear correlations are removed with the factor L of the covariance
    (scale; x - mu = z L^T), so the network sees unit variances and no
    correlation and has only the departure from a Gaussian to learn.  Dividing
    by the spreads alone left correlations of 0.99 between the Thiele-Innes
    constants, and the loss was still falling after 800 epochs.  If the
    covariance cannot be factorised the spreads are used.  mu and scale
    convert between the two spaces (_to_std, _from_std); the transformation
    is linear, so its Jacobian cancels in the acceptance ratio.

    If print_every > 0, prints the mean NLL loss every print_every epochs.
    """
    mu    = samples.mean(axis=0)
    # no absolute floor: a tightly constrained parameter can have a spread
    # below any fixed one; a constant column gets 1
    sigma = samples.std(axis=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    scale = sigma
    try:
        # factorise the correlation matrix, which does not depend on the units
        _corr = np.cov((samples - mu) / sigma, rowvar=False)
        scale = sigma[:, None] * np.linalg.cholesky(_corr)
    except np.linalg.LinAlgError:
        pass
    x_std = torch.tensor(_to_std(samples, mu, scale), dtype=torch.float32)

    ndim = samples.shape[1]
    flow = _RealNVP(ndim, n_layers=n_layers, hidden=hidden)
    opt  = torch.optim.Adam(flow.parameters(), lr=lr)

    # A fifth of the samples is held out and the training stops when the loss
    # on it has not improved for _PATIENCE epochs; the weights of the best
    # epoch are kept.  With 2000 samples the network otherwise learns the
    # individual samples: the training loss keeps falling for all 800 epochs
    # while the loss on other samples of the same posterior rises from 10 to
    # 66, and proposals drawn from such a flow are rejected.
    n_all = len(x_std)
    perm0 = torch.randperm(n_all)
    n_val = n_all // 5 if n_all >= 100 else 0
    x_val = x_std[perm0[:n_val]]
    x_std = x_std[perm0[n_val:]]
    best_loss, best_epoch, best_state = np.inf, 0, None

    n  = len(x_std)
    bs = min(batch_size, n)
    for epoch in range(1, n_epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, n, bs):
            batch = x_std[perm[start:start + bs]]
            opt.zero_grad()
            loss = -flow.log_prob(batch).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches  += 1
        if print_every and epoch % print_every == 0:
            print(f"    epoch {epoch:4d}/{n_epochs}  loss = {epoch_loss / n_batches:.4f}",
                  flush=True)
        if n_val:
            with torch.no_grad():
                val_loss = float(-flow.log_prob(x_val).mean())
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            elif epoch - best_epoch >= _PATIENCE:
                break

    if best_state is not None:
        flow.load_state_dict(best_state)
        # loss of a unit Gaussian on the same held-out samples, for comparison
        gauss_loss = float((0.5 * (x_val ** 2).sum(dim=1)).mean()
                           + 0.5 * ndim * np.log(2.0 * np.pi))
        print(f"    stopped at epoch {epoch}; kept epoch {best_epoch}, held-out loss "
              f"{best_loss:.3f} (unit Gaussian: {gauss_loss:.3f})", flush=True)

    flow.eval()
    return flow, mu, scale


# ── Flow-based in-model move ──────────────────────────────────────────────────

class _FlowInModelMove(MHMove):
    """Independence-sampler in-model move using trained per-source flows.

    For each active leaf with an associated trained flow, the proposal is drawn
    directly from the flow (independent of the current position).  Leaves beyond
    the number of trained flows fall back to a symmetric Gaussian random walk
    with the same step size as solver_rjmcmc.py's GaussianMove (log_factor = 0).

    The MH correction for the independence sampler is:
        log_factor[t, w] = Σ_{active leaves k} [log_q(x_curr_k) - log_q(x_prop_k)]
    where log_q is evaluated in standardised space and the Jacobian of the
    standardisation transform cancels.
    """

    def __init__(self, flows, mus, sigmas, fallback_std=None,
                 conditioner=None, cond_flows=None):
        """
        flows        : list of _RealNVP, one per source (leaf 0, 1, …)
        mus          : list of ndarray (ndim,) — flow training means
        sigmas       : list of ndarray — flow training scales (see _to_std)
        fallback_std : ndarray (ndim,) — step size for leaves without a flow
        conditioner  : _SourceConditioner, or None
        cond_flows   : list of (flow, mu, scale) trained on the residuals of
                       the conditioner; used for leaf k in the walkers where
                       all the other detected planets are active
        """
        super().__init__()
        self.flows        = flows
        self.mus          = [np.asarray(m) for m in mus]
        self.sigmas       = [np.asarray(s) for s in sigmas]
        self.fallback_std = fallback_std
        self._cond        = conditioner if cond_flows else None
        self._cond_flows  = cond_flows or []

    def _draw(self, flow, mu, scale, x_curr, shift):
        """Independence proposal x_prop = shift + flow draw, with
        log[q(x_curr)/q(x_prop)] evaluated in standardised space."""
        with torch.no_grad():
            t_curr  = torch.tensor(_to_std(x_curr - shift, mu, scale),
                                   dtype=torch.float32)
            lp_curr = flow.log_prob(t_curr).numpy()
            t_prop  = flow.sample(len(x_curr))
            lp_prop = flow.log_prob(t_prop).numpy()
        return _from_std(t_prop.numpy().astype(float), mu, scale) + shift, lp_curr - lp_prop

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Propose new coordinates for one randomly selected active leaf per walker.

        Only one active leaf is updated per step (per-source proposal, matching
        _PeriodicPerSourceMove Rec 1 in solver_rjmcmc.py).  Updating all leaves
        jointly multiplies the per-leaf acceptance probabilities, collapsing the
        overall rate to near zero when any one flow is poorly trained.

        Returns
        -------
        new_branches : dict  same structure as branches_coords
        lf_total     : ndarray (ntemps, nwalkers)  log[q(x_curr)/q(x_prop)]
        """
        first = next(iter(branches_coords.values()))
        ntemps, nwalkers = first.shape[0], first.shape[1]
        lf_total   = np.zeros((ntemps, nwalkers))
        new_branches = {}

        for bname, coords in branches_coords.items():
            ntemps_b, nwalkers_b, nleaves_max, ndim = coords.shape
            new_coords = coords.copy()

            if branches_inds is not None and bname in branches_inds:
                inds = branches_inds[bname]
            else:
                inds = np.ones((ntemps_b, nwalkers_b, nleaves_max), dtype=bool)

            lf = np.zeros((ntemps_b, nwalkers_b))

            # Per-walker: pick ONE active leaf to update; restore all others.
            chosen_leaf = np.full((ntemps_b, nwalkers_b), -1, dtype=int)
            for t in range(ntemps_b):
                for w in range(nwalkers_b):
                    active = np.where(inds[t, w])[0]
                    if len(active) > 0:
                        chosen_leaf[t, w] = active[int(random.randint(len(active)))]

            for leaf in range(nleaves_max):
                # Only update walkers that chose this leaf.
                active_mask  = inds[:, :, leaf] & (chosen_leaf == leaf)
                t_idx, w_idx = np.where(active_mask)
                n_active     = len(t_idx)
                if n_active == 0:
                    continue

                x_curr = coords[t_idx, w_idx, leaf, :]   # (n_active, ndim)

                if leaf < len(self.flows):
                    # Flow-based independence sampler.  Walkers in which all
                    # the other detected planets are active use the flow of
                    # the residuals, shifted by the term in those planets.
                    x_prop = np.empty_like(x_curr)
                    dlf    = np.zeros(n_active)
                    is_cond = np.zeros(n_active, dtype=bool)
                    if self._cond is not None and leaf < len(self._cond_flows):
                        others  = self._cond.others(leaf)
                        is_cond = np.all(inds[t_idx, w_idx][:, others], axis=1)
                    if np.any(is_cond):
                        x_oth = coords[t_idx[is_cond], w_idx[is_cond]][:, others, :]
                        shift = self._cond.shift(leaf, x_oth.reshape(len(x_oth), -1))
                        x_prop[is_cond], dlf[is_cond] = self._draw(
                            *self._cond_flows[leaf], x_curr[is_cond], shift)
                    if np.any(~is_cond):
                        x_prop[~is_cond], dlf[~is_cond] = self._draw(
                            self.flows[leaf], self.mus[leaf], self.sigmas[leaf],
                            x_curr[~is_cond], 0.0)
                    lf[t_idx, w_idx] += dlf

                else:
                    # Symmetric Gaussian fallback — log_factor contribution is 0
                    std   = (self.fallback_std if self.fallback_std is not None
                             else np.ones(ndim) * 0.01)
                    x_prop = x_curr + std * np.random.randn(n_active, ndim)

                new_coords[t_idx, w_idx, leaf, :] = x_prop

            new_branches[bname] = new_coords
            lf_total += lf

        return new_branches, lf_total


# ── Prior helper (identical to solver_rjmcmc.py) ──────────────────────────────

def _build_prior(param_lo, param_hi, log_uniform_idx):
    dists = {}
    for i in range(len(param_lo)):
        lo, hi = float(param_lo[i]), float(param_hi[i])
        if i in log_uniform_idx:
            dists[i] = log_uniform(lo, hi)
        else:
            dists[i] = uniform_dist(lo, hi)
    return ProbDistContainer(dists)


class _1DMixture:
    """Equal-weight mixture of frozen 1D truncated-Gaussian components (one
    per peeled source), exposing the .rvs(size)/.logpdf(x) interface
    ProbDistContainer expects for a per-dimension sub-distribution.
    See solver_rjmcmc.py for the full docstring (identical implementation)."""
    def __init__(self, dists, lo_phys, period=None):
        self.dists   = dists   # list of (frozen_truncnorm, mu)
        self.k       = len(dists)
        self.lo_phys = lo_phys
        self.period  = period

    def rvs(self, size=1):
        out_shape = size if isinstance(size, tuple) else (size,)
        n = int(np.prod(out_shape)) if len(out_shape) else 1
        choice = np.random.randint(0, self.k, size=n)
        out = np.empty(n)
        for i in range(n):
            d, mu = self.dists[choice[i]]
            v = float(d.rvs(size=1)[0])
            if self.period is not None:
                v = np.mod(v + mu - self.lo_phys, self.period) + self.lo_phys
            out[i] = v
        return out.reshape(out_shape)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        logps = []
        for d, mu in self.dists:
            if self.period is not None:
                xw = np.mod(x - mu + self.period / 2.0, self.period) - self.period / 2.0
                logps.append(d.logpdf(xw))
            else:
                logps.append(d.logpdf(x))
        return logsumexp(np.stack(logps, axis=0), axis=0) - np.log(self.k)


def _build_birth_generate_dist(peel_results, param_lo, param_hi, ndim, fit_idx=None):
    """Tight per-source truncated Gaussians from the peel posteriors, mixed
    across sources per dimension.  Used as the RJ birth-proposal generator
    instead of the flat/wide physical prior (see solver_rjmcmc.py's fuller
    docstring for the Birth/death acceptance = 0.000 rationale)."""
    if fit_idx is None:
        fit_idx = list(range(ndim))
    dists = {}
    for j in range(ndim):
        period = period_for(fit_idx[j])
        per_source = []
        for pr in peel_results:
            mu, sig, lo, hi, a, b = _tight_stats(
                pr.samples[:, j], param_lo[j], param_hi[j], period=period)
            if period is not None:
                per_source.append((truncnorm(a, b, loc=0.0, scale=sig), mu))
            else:
                per_source.append((truncnorm(a, b, loc=mu, scale=sig), mu))
        dists[j] = _1DMixture(per_source, lo_phys=float(param_lo[j]), period=period)
    return ProbDistContainer(dists)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(log_likelihood, log_prior, prior_transform, args,
        p0=None, ndim=7, n_max=N_MAX, n_min=N_MIN,
        nwalkers=NWALKERS, nsteps_burn=NSTEPS_BURN, nsteps_prod=NSTEPS_PROD,
        peel_best=None, peel_results=None, birth_results=None,
        param_lo=None, param_hi=None,
        log_uniform_idx=(0,),
        fit_idx=None,
        n_pool=N_POOL,
        **kwargs):
    """Run Eryn RJMCMC with normalising-flow in-model proposals.

    Parameters and return value are identical to solver_rjmcmc.run (including
    the ``birth_results`` kwarg and the EXOPLANETLAB_RJ_NTEMPS /
    EXOPLANETLAB_RJ_INMODEL_MIX / EXOPLANETLAB_RJ_START environment overrides).
    The only behavioural difference is that in-model proposals come from
    RealNVP flows trained on the peel posteriors, mixed with the
    affine-invariant DE move and a Gaussian fallback.

    peel_results must be provided (ExoPlanetLab.py always supplies it).  If it
    is None or empty the function raises ValueError because the flows cannot be
    trained without posterior samples.
    """
    import multiprocessing as _mp

    if param_lo is None or param_hi is None:
        raise ValueError("param_lo and param_hi must be provided")
    if not peel_results:
        raise ValueError(
            "solver_rjflow requires peel_results to train the normalising flows. "
            "Ensure ExoPlanetLab.py runs the peel phase before calling this solver."
        )

    pool = None
    if n_pool > 1:
        if _mp.get_start_method(allow_none=True) != 'fork':
            _mp.set_start_method('fork', force=True)
        pool = _mp.Pool(n_pool)

    ntemps   = max(1, int(os.environ.get('EXOPLANETLAB_RJ_NTEMPS', '4')))
    start_mode = os.environ.get('EXOPLANETLAB_RJ_START', 'mixed').strip().lower()
    if start_mode not in ('mixed', 'nmin', 'npeel'):
        start_mode = 'mixed'
    nwalkers = max(nwalkers, 2 * ndim + 2)
    if nwalkers % 2 != 0:
        nwalkers += 1

    param_lo_arr = np.array([float(x) for x in param_lo])
    param_hi_arr = np.array([float(x) for x in param_hi])

    # ── Train one flow per peel source ───────────────────────────────────────
    print("Training normalising flows on peel posteriors ...")
    flows, mus, sigmas = [], [], []
    for k, pr in enumerate(peel_results):
        samples_k = pr.samples
        if len(samples_k) < 50:
            raise ValueError(
                f"Peel source {k} has only {len(samples_k)} samples — "
                "too few to train a reliable flow (need ≥ 50)."
            )
        print(f"  Flow {k+1}/{len(peel_results)}: "
              f"{len(samples_k)} samples, {ndim} dims, {_N_EPOCHS} epochs ...")
        fl, mu, sg = _train_flow(samples_k, print_every=max(1, _N_EPOCHS // 10))
        flows.append(fl)
        mus.append(mu)
        sigmas.append(sg)
    print("  Flows trained.")

    # One more flow per planet, on the residuals about its linear dependence
    # on the other planets (see solver_rjmcmc._SourceConditioner).
    conditioner = _SourceConditioner(peel_results, ndim)
    cond_flows = []
    if conditioner.ok and conditioner.n == len(flows):
        print("  Flows of each planet given the others. Volume of the conditional "
              f"region relative to the marginal one: {conditioner.describe()}")
        for k in range(conditioner.n):
            print(f"  Conditional flow {k+1}/{conditioner.n} ...")
            cond_flows.append(_train_flow(conditioner.residuals(k),
                                          print_every=max(1, _N_EPOCHS // 10)))

    # Fallback step size for leaves beyond the peel count (symmetric Gaussian)
    fallback_std = np.median([np.std(pr.samples, axis=0) for pr in peel_results],
                             axis=0)

    # ── Prior ────────────────────────────────────────────────────────────────
    model_prior = _build_prior(param_lo, param_hi, log_uniform_idx)
    priors   = {'model': model_prior}

    # ── Moves ────────────────────────────────────────────────────────────────
    # DistributionGenerateRJ: birth/death proposals from a tight, peel-informed
    # mixture rather than the flat physical prior (Birth/death acceptance =
    # 0.000 fix — see solver_rjmcmc.py's _build_birth_generate_dist docstring).
    # peel_results is always available here (required above), so no fallback.
    _fit_idx = fit_idx if fit_idx is not None else list(range(ndim))
    _birth_src = birth_results if birth_results else peel_results
    birth_generate_dist = _build_birth_generate_dist(
        _birth_src, param_lo_arr, param_hi_arr, ndim, fit_idx=_fit_idx)
    if birth_results and len(birth_results) > len(peel_results):
        print(f"  Birth proposal: {len(birth_results)} components "
              f"(detected sources + strongest unexplained residual peak).")
    flow_move = _FlowInModelMove(flows, mus, sigmas, fallback_std=fallback_std,
                                 conditioner=conditioner, cond_flows=cond_flows)

    # Gaussian random-walk fallback, mixed in alongside the flow independence
    # sampler (matches solver_rjmcmc.py's Gaussian(70%)/KDE(30%) pattern).
    # This is not just belt-and-suspenders: the flow is an *independence*
    # sampler trained only on each source's individual peel posterior, which
    # is itself biased by cross-source degeneracies/crosstalk. If the true
    # joint-fit mode lies outside the region the flow was trained on, the
    # flow can never propose a point there -- no number of steps fixes that,
    # since it has no local diffusive component. The Gaussian move gives the
    # chain a way to random-walk away from a biased flow mode, exactly as it
    # does for solver_rjmcmc.py. See that file's _PeriodicPerSourceMove for
    # the periodic/reflective boundary handling shared with this move.
    # Full empirical covariance, 10%-shrunk toward its diagonal for
    # positive-definiteness -- a diagonal proposal steps across the thin
    # correlated per-source ridge and is always rejected (see solver_rjmcmc.py).
    stds = np.median([np.std(pr.samples, axis=0) for pr in peel_results], axis=0)
    _covs = [np.cov(pr.samples, rowvar=False) for pr in peel_results
             if getattr(pr, 'samples', None) is not None
             and pr.samples.shape[0] > 2 * pr.samples.shape[1]]
    if _covs:
        _cov_mat, _cov_ok, _cov_cond = _regularise_cov(_covs, stds, ndim)
        if _cov_ok:
            print(f"  Gaussian-fallback proposal: full empirical covariance "
                  f"(cond ~ {_cov_cond:.1e}).")
        else:
            _cov_mat = None
    else:
        _cov_mat = None
    cov  = {'model': _cov_mat if _cov_mat is not None else np.diag(stds ** 2)}
    gauss_move = _PeriodicPerSourceMove(cov, param_lo_arr, param_hi_arr,
                                        source_covs=_source_covs(peel_results, ndim))

    # Affine-invariant DE move (see solver_rjmcmc.py) — tracks the correlated
    # ridge from the ensemble alone, complementing the flow independence sampler.
    de_move = _PerSourceDEMove(param_lo_arr, param_hi_arr, 1.0e-3 * np.asarray(stds))

    _move_by_name = {'flow': flow_move, 'de': de_move, 'gauss': gauss_move}
    _mix = _parse_move_mix(os.environ.get('EXOPLANETLAB_RJ_INMODEL_MIX'),
                           'flow:0.65,de:0.10,gauss:0.25')
    in_model_moves = [(_move_by_name[n], w) for n, w in _mix if n in _move_by_name]
    if not in_model_moves:
        in_model_moves = [(flow_move, 0.7), (gauss_move, 0.3)]

    rj_move   = DistributionGenerateRJ(
        {'model': birth_generate_dist},
        nleaves_max={'model': n_max},
        nleaves_min={'model': n_min},
    )

    # ── Sampler ──────────────────────────────────────────────────────────────
    _tempering_kwargs = (dict(ntemps=ntemps, Tmax=np.inf, adaptive=True)
                         if ntemps > 1 else {})
    sampler = EnsembleSampler(
        nwalkers,
        {'model': ndim},
        log_likelihood,
        priors,
        branch_names=['model'],
        nleaves_max={'model': n_max},
        nleaves_min={'model': n_min},
        args=args,
        moves=in_model_moves,
        rj_moves=[(rj_move, 1.0)],
        tempering_kwargs=_tempering_kwargs,
        pool=pool,
    )

    # ── Initial state ─────────────────────────────────────────────────────────
    # Draw walker starting positions directly from the trained flows.
    # This is the critical fix for zero in-model acceptance: the MH log-factor
    # is lp_curr - lp_prop = log_flow(x_curr) - log_flow(x_prop).  If x_curr
    # was seeded from peel best-fits or raw posterior samples, the flow may
    # assign very low probability to that specific point (especially near mode
    # boundaries), making lp_curr → -∞ and causing all proposals to be
    # rejected.  Seeding from the flow itself guarantees lp_curr is always a
    # typical value under the flow, so the log-factor never dominates the MH
    # ratio negatively.
    n_peel = len(peel_best) if peel_best else 1
    n_start_lo = max(n_min, 1)

    coords_gw = np.zeros((ntemps, nwalkers, n_max, ndim))
    inds_gw   = np.zeros((ntemps, nwalkers, n_max), dtype=bool)
    if start_mode == 'npeel':
        inds_gw[:, :, :n_peel] = True
    elif start_mode == 'nmin':
        inds_gw[:, :, :n_start_lo] = True
    else:  # mixed — a quarter start at n_min to probe the birth channel
        _lo_n = max(1, nwalkers // 4)
        inds_gw[:, :_lo_n, :n_start_lo] = True
        inds_gw[:, _lo_n:, :n_peel]     = True
    print(f"  Initial N per walker: start_mode='{start_mode}' "
          f"(n_min={n_min}, peel count={n_peel}); {ntemps} temperature(s).")

    print("Initialising walkers from trained flows ...")
    for j in range(n_peel):
        if j < len(flows):
            # Sample ntemps*nwalkers points from flow j, then un-standardise
            with torch.no_grad():
                x_std = flows[j].sample(ntemps * nwalkers).numpy()   # (N, ndim) std-space
            x_init = _from_std(x_std.astype(float), mus[j], sigmas[j])  # physical space
            coords_gw[:, :, j, :] = x_init.reshape(ntemps, nwalkers, ndim)
        else:
            # Safety fallback: flow not available (should not normally occur)
            pb    = np.asarray(peel_best[j], dtype=float)
            noise = fallback_std * np.random.randn(ntemps, nwalkers, ndim) * 0.1
            coords_gw[:, :, j, :] = pb[np.newaxis, np.newaxis, :]  + noise

    # Inactive slots: random draws from uniform prior bounds
    for j in range(n_peel, n_max):
        u = np.random.uniform(0.0, 1.0, (ntemps, nwalkers, ndim))
        coords_gw[:, :, j, :] = param_lo_arr + u * (param_hi_arr - param_lo_arr)

    coords_gw = np.clip(
        coords_gw,
        param_lo_arr[np.newaxis, np.newaxis, np.newaxis, :],
        param_hi_arr[np.newaxis, np.newaxis, np.newaxis, :],
    )

    # Verify that initial flow log-probs are well-defined (not -inf).
    # A healthy run should show median log_prob values around -ndim/2 * log(2π)
    # in standardised space (≈ -8.8 for ndim=7).  Very negative values (<< -50)
    # would indicate walkers escaped the flow support after clipping.
    for j in range(min(n_peel, len(flows))):
        x_j = coords_gw[0, :, j, :]                  # (nwalkers, ndim)
        x_j_std = _to_std(x_j, mus[j], sigmas[j])
        with torch.no_grad():
            lp_j = flows[j].log_prob(
                torch.tensor(x_j_std, dtype=torch.float32)
            ).numpy()
        print(f"  Initial flow {j+1} log_prob: "
              f"median={np.median(lp_j):.1f}, min={lp_j.min():.1f}, max={lp_j.max():.1f}")

    state = State({'model': coords_gw}, inds={'model': inds_gw})

    # ── Burn-in + production ─────────────────────────────────────────────────
    print(f"Running RJFlow RJMCMC: {nwalkers} walkers, "
          f"{ndim} params/source, N ∈ [{n_min}, {n_max}]")
    print(f"Burn-in: {nsteps_burn} steps (adaptive Gaussian-fallback tuning), "
          f"production: {nsteps_prod} steps (fixed proposal)")
    _mix_str = ', '.join(f"{n} {w:.2f}" for n, w in _mix if n in _move_by_name)
    print(f"In-model moves: {_mix_str}")
    if ntemps > 1:
        print(f"Parallel tempering: {ntemps} temperatures "
              f"(~{ntemps}x the single-chain cost).")
    print("----------------------------------------------------------------------")
    try:
        # Two-stage run, same rationale and mechanism as solver_rjmcmc.py:
        # tune the Gaussian fallback move's step size during (unstored)
        # burn-in in manual chunks, then freeze it for the stored production
        # phase. eryn's own tune=True cannot be used here -- see
        # _PeriodicPerSourceMove.rescale docstring in solver_rjmcmc.py for
        # why (the RJ birth/death move repurposes the `tune` name as a bool
        # config flag, crashing eryn's tune=True dispatch).
        if nsteps_burn > 0:
            n_chunks   = 10
            chunk_size = max(1, nsteps_burn // n_chunks)
            remaining  = nsteps_burn
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
                gauss_move.accepted, gauss_move.num_proposals = \
                    np.zeros((ntemps, nwalkers)), 0
                state = sampler.run_mcmc(
                    state, this_chunk, store=False, progress=False)
                rate = (np.mean(np.asarray(gauss_move.accepted, dtype=float), axis=1)
                        / max(gauss_move.num_proposals, 1))
                gauss_move.rescale(rate)
                remaining -= this_chunk
            _tf = gauss_move._temp_factor
            print(f"  Burn-in done. Gaussian fallback proposal step factor per temperature "
                  f"(cold first): {', '.join(f'{x:.3f}' for x in _tf)}")

        for mv, _ in in_model_moves:
            mv.accepted, mv.num_proposals = np.zeros((ntemps, nwalkers)), 0
        rj_move.accepted, rj_move.num_proposals = np.zeros((ntemps, nwalkers)), 0

        sampler.run_mcmc(state, nsteps_prod, progress=True)
    except KeyboardInterrupt:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    if pool is not None:
        pool.close()
        pool.join()

    # ── Extract results (identical to solver_rjmcmc.py) ─────────────────────────
    chain_coords = sampler.get_chain()['model']
    chain_inds   = sampler.get_inds()['model']
    n_chain_full = sampler.get_nleaves()['model']
    chain_logl   = sampler.get_log_like()

    n_chain    = n_chain_full[:, 0, :]
    logl_chain = chain_logl[:, 0, :]

    n_flat          = n_chain.flatten().astype(int)
    n_distribution  = np.bincount(n_flat, minlength=n_max + 1)
    most_probable_n = int(np.argmax(n_distribution))

    samples_list, logl_list = [], []
    for step in range(nsteps_prod):
        for w in range(nwalkers):
            if n_chain[step, w] == most_probable_n:
                active = chain_inds[step, 0, w]
                leaf_p = chain_coords[step, 0, w, active]
                order  = np.argsort(leaf_p[:, 0])
                samples_list.append(leaf_p[order].flatten())
                logl_list.append(logl_chain[step, w])

    if samples_list:
        samples     = np.array(samples_list)
        log_probs   = np.array(logl_list)
        best_params = samples[np.argmax(log_probs)]
    else:
        samples     = np.zeros((0, most_probable_n * ndim))
        log_probs   = np.zeros(0)
        best_params = np.zeros(most_probable_n * ndim)

    def _cold_mean(x):
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        return float(np.mean(x[0])) if x.ndim > 1 else float(np.mean(x))

    af_inmodel = _cold_mean(sampler.acceptance_fraction)
    af_rj      = _cold_mean(sampler.rj_acceptance_fraction)
    rj_tau, rj_tau_means, rj_tau_all, rj_tau_walkers = _cold_chain_tau(chain_coords, chain_inds, n_chain,
                                             most_probable_n)

    # Acceptance of each in-model move on the cold chain (production only).
    # The overall figure hides a move that is never accepted.
    for _mv, _w in in_model_moves:
        try:
            _acc = np.asarray(_mv.accepted, dtype=float)
            _np = max(int(_mv.num_proposals), 1)
            print(f"  Cold-chain acceptance of {type(_mv).__name__.lstrip('_'):24s}"
                  f" (weight {_w:.2f}): {float(np.mean(_acc[0])) / _np:.3f}")
        except Exception:
            pass

    try:
        _rj_acc  = np.asarray(rj_move.accepted, dtype=float)
        rj_n_acc = int(np.sum(_rj_acc[0] if _rj_acc.ndim > 1 else _rj_acc))
        rj_n_prop = int(rj_move.num_proposals) * nwalkers
    except Exception:
        rj_n_acc, rj_n_prop = None, None

    return SimpleNamespace(
        samples                = samples,
        log_probs              = log_probs,
        best_params            = best_params,
        chain                  = None,
        chain_burn             = None,
        logprob_burn           = None,
        acceptance_fraction    = af_inmodel,
        rj_acceptance_fraction = af_rj,
        tau                    = None,
        tau_err                = None,
        nsteps_prod            = nsteps_prod,
        logz                   = None,
        logzerr                = None,
        n_chain                = n_chain,
        most_probable_n        = most_probable_n,
        n_distribution         = n_distribution,
        rj_accepted_count      = rj_n_acc,
        rj_proposed_count      = rj_n_prop,
        rj_tau                 = rj_tau,
        rj_tau_walkers         = rj_tau_walkers,
        rj_tau_means           = rj_tau_means,
        rj_tau_all             = rj_tau_all,
    )


def print_diagnostics(result, active_names):
    """Print RJFlow RJMCMC diagnostics to stdout."""
    print("\n--- RJFlow RJMCMC diagnostics ---")
    total = result.n_distribution.sum()
    print("  Posterior over number of sources N:")
    for n, count in enumerate(result.n_distribution):
        frac   = count / total if total > 0 else 0.0
        marker = '  ← MAP' if n == result.most_probable_n else ''
        print(f"    N = {n}: {count:7d} samples  ({frac * 100:5.1f}%){marker}")
    print(f"  Most probable N         : {result.most_probable_n}")
    print(f"  Samples at most prob. N : {len(result.samples)}")
    if result.acceptance_fraction is not None:
        print(f"  In-model acceptance     : {result.acceptance_fraction:.3f}")
    else:
        print("  In-model acceptance     : N/A (RJ run skipped)")
    if result.rj_acceptance_fraction is not None:
        _cnt = getattr(result, 'rj_accepted_count', None)
        _tot = getattr(result, 'rj_proposed_count', None)
        _extra = (f"  ({_cnt} / {_tot} accepted)"
                  if _cnt is not None and _tot else "")
        print(f"  Birth/death acceptance  : "
              f"{result.rj_acceptance_fraction:.4f}{_extra}")
    _print_tau(result, active_names)
    if result.logz is not None:
        print(f"  Log evidence (peel sum) : {result.logz:.2f}")
    else:
        print("  Note: Eryn does not compute a log-evidence; use the N-posterior")
        print("        occupation fractions for model comparison.")
