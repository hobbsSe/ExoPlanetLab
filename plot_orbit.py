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
Plotting functions for exoplanet astrometric orbit parameter estimation.

Each function saves one PNG file and prints a confirmation line.
All functions accept only plain Python/numpy arguments so they remain
independent of the solver and the main problem module.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm

import orbit


# ── Corner-plot rendering style ───────────────────────────────────────────────
# corner.py's defaults draw every posterior sample as an individual black point
# and build jagged 20-bin contours at the wrong enclosed-probability levels.
# With a ~1M-sample emcee chain that turns into a solid black smear that hides
# the real structure -- e.g. the curved / ring-like Thiele-Innes (A,B,F,G)
# orientation degeneracy for a lower-SNR planet.  Suppressing the scatter and
# density layers, smoothing, and drawing proper 1/2/3-sigma filled contours is
# a pure *display* fix: it needs no rerun and does not change the 1-D marginals
# or the quantile titles (computed from the full sample set).
_SIGMA_LEVELS_2D = (0.393, 0.865, 0.989)   # enclosed prob. at 1, 2, 3 sigma

CORNER_STYLE = dict(
    bins=40,
    smooth=1.0,
    smooth1d=1.0,
    levels=_SIGMA_LEVELS_2D,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=True,
    contour_kwargs={'linewidths': 0.9},
    hist_kwargs={'linewidth': 1.1},
    title_fmt='.4g',   # ".2f" default renders a small sigma (e.g. e ~0.004) as +0.00
)


def _corner_ranges(samples_display, weights=None):
    """Axis ranges for corner: the posterior bulk (1-99th percentile + 12%
    padding) when unweighted, or a 99.5% credible fraction when weighted
    (percentiles are unreliable on a weighted sample)."""
    if weights is not None:
        return [0.995] * samples_display.shape[1]
    lo  = np.percentile(samples_display, 1.0,  axis=0)
    hi  = np.percentile(samples_display, 99.0, axis=0)
    pad = 0.12 * np.maximum(hi - lo, 1e-10)
    return [(l - p, h + p) for l, p, h in zip(lo, pad, hi)]


def plot_along_scan(t_obs, along_scan_obs, along_scan_true, sigma_total,
                    true_params_all, outdir='.'):
    """Observed and true along-scan signal: full time series + best-fit overlay.

    Parameters
    ----------
    t_obs            : (N,) observation times (days)
    along_scan_obs   : (N,) observed along-scan residuals (mas)
    along_scan_true  : (N,) noiseless true signal (mas), or None for a
                       blind fit (no reference orbit) -- the "True signal"
                       curve is omitted.
    sigma_total      : (N,) per-epoch uncertainties (mas)
    true_params_all  : (N_planets, 7) true orbital parameters, or None to
                       match along_scan_true=None.
    outdir           : str  output directory
    """
    have_truth = true_params_all is not None
    if have_truth:
        n_plt   = len(true_params_all)
        periods = [true_params_all[i, 0] for i in range(n_plt)]
        p_str   = ', '.join(f'{P:.0f} d' for P in periods)
        title   = f'Gaia along-scan residuals  —  {n_plt} planet(s): P = {p_str}'
    else:
        title   = 'Gaia along-scan residuals  —  blind fit (no reference orbit)'

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].errorbar(t_obs, along_scan_obs, yerr=sigma_total,
                     fmt='k.', ms=3, elinewidth=0.5, alpha=0.7,
                     label='Observed (signal + noise)')
    if have_truth:
        axes[0].plot(t_obs, along_scan_true, 'r-', lw=1.5, alpha=0.8,
                     label='True signal')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Along-scan residual (mas)')
    axes[0].set_title(title)
    axes[0].legend()

    # First planet's period worth of data if known, else the first 10% of
    # the observation span (blind fit -- no true period to zoom in on).
    P0    = true_params_all[0, 0] if have_truth else 0.1 * (t_obs.max() - t_obs.min())
    mask  = t_obs <= min(t_obs.min() + P0, t_obs.max())
    axes[1].errorbar(t_obs[mask], along_scan_obs[mask], yerr=sigma_total[mask],
                     fmt='k.', ms=4, elinewidth=0.7, label='Observed')
    if have_truth:
        axes[1].plot(t_obs[mask], along_scan_true[mask], 'r-', lw=1.5, label='True')
        axes[1].set_title(f'First orbit (planet 1, P = {P0:.0f} days)')
    else:
        axes[1].set_title('First 10% of the observation span')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Along-scan residual (mas)')
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(outdir, 'along_scan.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_orbit_sky(t_obs, best_params, true_params, scan_angles, source_id,
                   outdir='.'):
    """Best-fit orbit on the sky plane (ΔRA vs ΔDec).

    Parameters
    ----------
    t_obs       : (N,) observation times (days)
    best_params : (7,) MAP orbital parameters [P, e, T0, A, B, F, G]
    true_params : (7,) true orbital parameters, or None
    scan_angles : (N,) scan angles (rad)
    source_id   : int or str  Gaia source identifier for the title
    outdir      : str
    """
    P, e, T0, A, B, F, G = best_params
    # One full period of each orbit (not just the observed window): a period
    # longer than the baseline would otherwise show as a partial arc, and a
    # reference orbit with a very different period couldn't be compared.
    t_dense = t_obs.min() + P * np.linspace(0.0, 1.0, 2000)

    dra, ddec = orbit.astrometric_orbit(t_dense, P, e, T0, A, B, F, G)
    dra_obs, ddec_obs = orbit.astrometric_orbit(t_obs, P, e, T0, A, B, F, G)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dra, ddec, 'b-', lw=0.8, label='Best-fit orbit')
    ax.plot(dra_obs, ddec_obs, 'r.', ms=5, label='Observation epochs')
    ax.plot(0, 0, 'k+', ms=12, mew=2, label='Barycentre (0, 0)')

    # Periastron
    dra_p, ddec_p = orbit.astrometric_orbit(
        np.array([T0 % P]), P, e, T0, A, B, F, G)
    ax.plot(dra_p, ddec_p, 'g*', ms=12, label='Periastron')

    xs, ys = [dra, np.array([0.0])], [ddec, np.array([0.0])]
    if true_params is not None:
        Pt, et, T0t, At, Bt, Ft, Gt = true_params
        t_true = t_obs.min() + Pt * np.linspace(0.0, 1.0, 2000)
        dra_t, ddec_t = orbit.astrometric_orbit(t_true, Pt, et, T0t, At, Bt, Ft, Gt)
        ax.plot(dra_t, ddec_t, 'k--', lw=0.8, alpha=0.6, label='True orbit')
        # The part of the reference orbit the data actually span.
        t_win = np.linspace(t_obs.min(), t_obs.max(), 500)
        dra_w, ddec_w = orbit.astrometric_orbit(t_win, Pt, et, T0t, At, Bt, Ft, Gt)
        ax.plot(dra_w, ddec_w, 'k-', lw=1.8, alpha=0.7,
                label='True orbit (over the observed epochs)')
        xs.append(dra_t)
        ys.append(ddec_t)

    ax.set_xlabel(r'$\Delta$ RA (mas)')
    ax.set_ylabel(r'$\Delta$ Dec (mas)')
    ax.set_title(f'Astrometric orbit  —  source {source_id}', fontsize=10)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_aspect('equal', adjustable='datalim')

    # Limits cover every orbit drawn (fit and, when present, the reference).
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    x_range = x_all.max() - x_all.min()
    y_range = y_all.max() - y_all.min()
    pad_x   = x_range * 0.08
    pad_y   = max(y_range * 0.08, x_range * 0.05)
    ax.set_xlim(x_all.min() - pad_x, x_all.max() + pad_x)
    ax.set_ylim(y_all.min() - pad_y, y_all.max() + pad_y)

    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'orbit_sky.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_joint_orbit_sky(best_params_list, true_params_list=None, t_span=None,
                         source_id='', outdir='.'):
    """All planets' best-fit orbits together on one sky-plane (ΔRA vs ΔDec) plot.

    Unlike plot_orbit_sky, this only needs each planet's 7-parameter point
    estimate (e.g. the "Median" column already written to summary.txt), not
    the observation epochs or posterior samples -- so it can be regenerated
    from a completed run's saved output without re-running the sampler.

    Parameters
    ----------
    best_params_list : list of (7,) arrays  [P, e, T0, A, B, F, G] per planet
    true_params_list : list of (7,) arrays or None, same length/order, or
                        None per-entry where no truth is available
    t_span            : (t_min, t_max) in days, or None to default to
                        [0, 1.1 * max(P)] across all listed planets
    source_id         : label for the plot title
    outdir            : str
    """
    if t_span is None:
        p_max = max(float(bp[0]) for bp in best_params_list)
        t_span = (0.0, 1.1 * p_max)
    t_dense = np.linspace(t_span[0], t_span[1], 2000)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, best_params in enumerate(best_params_list):
        c = colors[i % len(colors)]
        P, e, T0, A, B, F, G = best_params
        dra, ddec = orbit.astrometric_orbit(t_dense, P, e, T0, A, B, F, G)
        ax.plot(dra, ddec, '-', lw=1.0, color=c, label=f'Planet {i+1} best-fit')

        dra_p, ddec_p = orbit.astrometric_orbit(np.array([T0 % P]), P, e, T0, A, B, F, G)
        ax.plot(dra_p, ddec_p, '*', ms=12, color=c)

        true_params = true_params_list[i] if true_params_list is not None else None
        if true_params is not None:
            Pt, et, T0t, At, Bt, Ft, Gt = true_params
            dra_t, ddec_t = orbit.astrometric_orbit(t_dense, Pt, et, T0t, At, Bt, Ft, Gt)
            ax.plot(dra_t, ddec_t, '--', lw=0.8, alpha=0.6, color=c,
                    label=f'Planet {i+1} true')

    ax.plot(0, 0, 'k+', ms=12, mew=2, label='Barycentre (0, 0)')
    ax.set_xlabel(r'$\Delta$ RA (mas)')
    ax.set_ylabel(r'$\Delta$ Dec (mas)')
    title = 'Joint astrometric orbits'
    if source_id:
        title += f'  —  source {source_id}'
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'joint_orbit_sky.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def _draw_truth_hline(ax, value):
    """Draw a reference-value line without letting it rescale the axis.

    The reference orbit can be far from the fit (e.g. a literature period 2.6x
    the fitted one); letting axhline join the autoscale squashes every walker
    into a thin line at one edge of the panel. Limits stay data-driven; if the
    reference value is off-scale it is noted at the panel edge instead.
    """
    lo, hi = ax.get_ylim()
    if lo <= value <= hi:
        ax.axhline(value, color='r', lw=1.0, ls='--', label='truth')
    else:
        up = value > hi
        ax.annotate(f'truth = {value:.4g} ({"above" if up else "below"} range)',
                    xy=(0.995, 0.93 if up else 0.07), xycoords='axes fraction',
                    ha='right', va='center', color='r', fontsize=7)
    ax.set_ylim(lo, hi)


def plot_burnin(chain_burn, logprob_burn, active_names, active_units,
                active_scales, active_truths, outdir='.'):
    """Burn-in traces: log-probability + one panel per fitted parameter."""
    ndim = chain_burn.shape[2]
    fig, axes = plt.subplots(ndim + 1, 1, figsize=(12, 2*(ndim+1)), sharex=True)

    axes[0].plot(logprob_burn, alpha=0.2, lw=0.5, color='k', rasterized=True)
    axes[0].set_ylabel('log prob', fontsize=8)
    axes[0].set_title('Burn-in traces', fontsize=9)

    truths_iter = active_truths if active_truths is not None else [None] * len(active_names)
    for j, (ax, name, unit, scale, truth) in enumerate(
            zip(axes[1:], active_names, active_units, active_scales, truths_iter)):
        ax.plot(chain_burn[:, :, j] * scale, alpha=0.2, lw=0.5, color='k', rasterized=True)
        if truth is not None:
            _draw_truth_hline(ax, truth * scale)
        ax.set_ylabel(f'{name} ({unit})' if unit else name, fontsize=8)

    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=7)
    axes[-1].set_xlabel('Burn-in step')
    fig.tight_layout()
    path = os.path.join(outdir, 'mcmc_burnin.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_traces(chain, active_names, active_units, active_scales, active_truths,
                nsteps_prod, tau=None, outdir='.'):
    """Production chain traces with optional 50τ convergence markers."""
    ndim = chain.shape[2]
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2*ndim), sharex=True)
    if ndim == 1:
        axes = [axes]

    truths_iter = active_truths if active_truths is not None else [None] * len(active_names)
    for j, (ax, name, unit, scale, truth) in enumerate(
            zip(axes, active_names, active_units, active_scales, truths_iter)):
        # real production steps on x (a thinned stored chain -- solver_ptemcee's
        # PT_THIN -- has fewer rows than nsteps_prod; tau is in real steps)
        _x = np.arange(chain.shape[0]) * max(1, (nsteps_prod or 0) // max(1, chain.shape[0]))
        ax.plot(_x, chain[:, :, j] * scale, alpha=0.2, lw=0.5, color='k', rasterized=True)
        if truth is not None:
            _draw_truth_hline(ax, truth * scale)
        ax.set_ylabel(f'{name} ({unit})' if unit else name, fontsize=8)
        if tau is not None:
            ax.axvline(50*tau[j], color='b', lw=0.8, ls=':',
                       label=f'50τ = {50*tau[j]:.0f}')

    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    axes[-1].set_xlabel('Production step')
    fig.suptitle('MCMC Traces — each line is one walker', y=1.001)
    fig.tight_layout()
    path = os.path.join(outdir, 'mcmc_traces.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_corner(samples, active_names, active_units, active_scales, active_truths,
                outdir='.', weights=None):
    """Corner plot of the posterior in display units.

    weights : optional per-sample weights (e.g. nessai nested-sampling
    posterior weights).  When given, ``samples`` may be the full weighted set.
    """
    from matplotlib.ticker import ScalarFormatter

    samples_display = samples * active_scales
    labels = [f'{n} ({u})' if u else str(n)
              for n, u in zip(active_names, active_units)]

    ranges = _corner_ranges(samples_display, weights)

    truths_display = active_truths * active_scales if active_truths is not None else None
    fig = corner.corner(
        samples_display,
        labels=labels,
        truths=truths_display,
        weights=weights,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        range=ranges,
        label_kwargs={'fontsize': 9, 'labelpad': 12},
        title_kwargs={'fontsize': 8},
        **CORNER_STYLE,
    )

    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(axis='both', labelsize=7, pad=3)

    fig.subplots_adjust(hspace=0.12, wspace=0.12)
    path = os.path.join(outdir, 'corner_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"Saved {path}")
    plt.close(fig)


def plot_joint_corner(samples, n_planets, active_names, active_units, active_scales,
                      truth_for, true_params_all, fit_idx, outdir='.', weights=None):
    """Joint corner plot of all N planets' parameters in a single figure.

    Parameters
    ----------
    samples         : (n_flat, n_planets * ndim_per) joint posterior samples
    n_planets       : int
    active_names    : list[str] — parameter names, length ndim_per
    active_units    : list[str] — parameter units, length ndim_per
    active_scales   : ndarray (ndim_per,)
    truth_for       : list[int | None] — maps inferred planet index → true planet index
    true_params_all : ndarray (N_true, 7)
    fit_idx         : list[int]
    outdir          : str
    """
    from matplotlib.ticker import ScalarFormatter

    ndim_per = len(active_names)
    labels = [
        f'p{pi+1}: {name} ({unit})' if unit else f'p{pi+1}: {name}'
        for pi in range(n_planets)
        for name, unit in zip(active_names, active_units)
    ]

    joint_scales    = np.tile(active_scales, n_planets)
    samples_display = samples * joint_scales

    joint_truths = []
    any_truth    = False
    for pi in range(n_planets):
        k_true = truth_for[pi] if pi < len(truth_for) else None
        if k_true is not None:
            tp = true_params_all[k_true][fit_idx] * active_scales
            joint_truths.extend(tp.tolist())
            any_truth = True
        else:
            joint_truths.extend([np.nan] * ndim_per)
    joint_truths_arr = np.array(joint_truths) if any_truth else None

    ranges = _corner_ranges(samples_display, weights)

    fig = corner.corner(
        samples_display,
        labels=labels,
        truths=joint_truths_arr,
        weights=weights,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        range=ranges,
        label_kwargs={'fontsize': 7, 'labelpad': 12},
        title_kwargs={'fontsize': 6},
        **CORNER_STYLE,
    )

    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(axis='both', labelsize=6, pad=3)

    fig.subplots_adjust(hspace=0.10, wspace=0.10)
    path = os.path.join(outdir, 'joint_corner_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"Saved {path}")
    plt.close(fig)


def plot_residuals(t_obs, residual, sigma_total, outdir='.', fraction=0.5):
    """Best-fit along-scan residual time series and histogram with Gaussian fit."""
    show_r        = int(-fraction * len(residual))
    mu_r, sigma_r = norm.fit(residual[show_r:])
    xgauss        = np.linspace(residual[show_r:].min(), residual[show_r:].max(), 200)
    sigma_noise   = float(np.median(sigma_total))

    print(f"\nBest-fit residual (last {int(fraction*100)}% of time series):")
    print(f"  mu    = {mu_r:.4f} mas")
    print(f"  sigma = {sigma_r:.4f} mas  (median sigma_total = {sigma_noise:.4f} mas)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].errorbar(t_obs[show_r:], residual[show_r:], yerr=sigma_total[show_r:],
                     fmt='k.', ms=3, elinewidth=0.5, alpha=0.7)
    axes[0].axhline(0, color='r', ls='--', lw=0.8)
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Residual (mas)')
    axes[0].set_title(f'Best-fit residual (last {int(fraction*100)}%)')

    axes[1].hist(residual[show_r:], bins=40, density=True, alpha=0.7)
    axes[1].plot(xgauss, norm.pdf(xgauss, mu_r, sigma_r), 'k-', lw=2)
    axes[1].set_xlabel('Residual (mas)')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Residual:  μ = {mu_r:.4f},  σ = {sigma_r:.4f} mas')

    fig.tight_layout()
    path = os.path.join(outdir, 'fit_residuals.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_n_posterior(n_chain, n_true, outdir='.'):
    """Bar chart of the RJMCMC posterior over the number of planets N.

    Parameters
    ----------
    n_chain : ndarray  shape (nsteps_prod, nwalkers) — N value per step/walker.
    n_true  : int      true (injected) number of planets, marked with a dashed line.
    outdir  : str      output directory.
    """
    n_flat   = n_chain.flatten().astype(int)
    n_max    = n_flat.max()
    counts   = np.bincount(n_flat, minlength=n_max + 1)
    n_values = np.arange(len(counts))
    fracs    = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(n_values, fracs * 100, color='steelblue', edgecolor='white', linewidth=0.5)

    map_n = int(np.argmax(counts))
    bars[map_n].set_color('darkorange')

    ax.axvline(n_true, color='red', linestyle='--', linewidth=1.5,
               label=f'True N = {n_true}')

    ax.set_xlabel('Number of planets N')
    ax.set_ylabel('Posterior probability (%)')
    ax.set_title('RJMCMC posterior over model order')
    ax.set_xticks(n_values)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(outdir, 'n_posterior.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def write_summary(outdir, solver_name, n_transits, t_total_days, obs_years,
                  snr, sigma_noise,
                  param_names, param_units, param_scales, true_params,
                  fit_flags, fit_idx, active_names, active_units,
                  active_scales, active_truths, result, residual,
                  resid_fraction=0.5, mass_info=None, t0_raw=None,
                  sigma_label='Instrument noise', rv_instruments=None,
                  rv_K=None, astro_jitter=None, astro_noise=None, single_star=None,
                  rel_astrometry=None, hipparcos=None,
                  truth_is_literature=False,
                  full_tau=None, full_active_names=None,
                  full_nsteps_prod=None, full_tau_err=None):
    """Write a plain-text summary to <outdir>/summary.txt.

    ``t0_raw``, if given, is ``(local_idx, unit, scale, q16, q50, q84, k, P0)``
    for the raw (unfolded) fitted T0: ``result`` already carries the folded T0,
    and this adds a 'T0(raw)' row plus a note that raw = folded + k*P.

    ``sigma_noise`` is the representative per-epoch noise reported in the
    summary; ``sigma_label`` names what it actually is (e.g. 'Instrument
    noise' for simulated/NSS data where it is a clean instrument floor, or
    'Median measured sigma' for real epoch data with no instrument/
    astrometric decomposition) -- callers must pick the label that matches
    what they pass in, since this function can't tell the difference.

    ``full_tau``/``full_active_names``/``full_nsteps_prod``/``full_tau_err``,
    if given, override ``result``'s own (per-planet-sliced) tau for the
    Autocorrelation-times table and CONVERGED/NOT CONVERGED verdict. For a
    joint fit with RV (K/gamma/jitter) or M_star/parallax nuisance
    parameters, ``result`` here is a single planet's column-sliced view
    (see ExoPlanetLab.py's extract_planet_result) -- its own ``.tau`` only
    covers that planet's 7 orbital elements, so reporting convergence from
    it silently ignores whether the *shared* chain's other parameters
    converged. That's not hypothetical: on a real run, every orbital
    element converged fine (tau ~250-280) while one RV jitter's tau came
    back nan (i.e. not converged at all) -- the per-planet-only view would
    have reported a clean "CONVERGED" while the actual chain was not.
    Pass the full joint result's tau/names/nsteps_prod/tau_err here
    whenever one exists; omitting them falls back to ``result``'s own
    (correct only when there's nothing else sharing the chain, e.g. no RV
    and a single planet)."""
    show_r      = int(-resid_fraction * len(residual))
    mu_r, sig_r = norm.fit(residual[show_r:])

    path = os.path.join(outdir, 'summary.txt')
    with open(path, 'w') as f:
        def w(line=''):
            f.write(line + '\n')

        w('=== Exoplanet Orbit Parameter Estimation Summary ===')
        w(f'Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        w(f'Solver    : {solver_name}')
        w()

        w('--- Observation Setup ---')
        w(f'Number of transits : {n_transits}')
        w(f'Observation span   : {t_total_days:.1f} days  ({obs_years:.2f} yr)')
        w(f'{sigma_label:19s} : {sigma_noise:.4f} mas')
        w(f'Signal-to-noise    : {snr:.2f}')
        w()

        w('--- True Orbital Parameters ---')
        for name, unit, scale, val in zip(param_names, param_units,
                                          param_scales, true_params):
            u = f' {unit}' if unit else ''
            w(f'  {name:6s} = {val * scale:.6g}{u}')
        if truth_is_literature:
            w("  NOTE: literature value (see gaia_dr4_epoch.KNOWN_TRUTH), shown")
            w("  for comparison only -- it never initialises or constrains the")
            w("  fit. The fitted data are residuals of a single-star fit, so both")
            w("  orbits are compared through the same (I-H) projection.")
        w()

        w('--- Fit Configuration ---')
        flag_str  = '[' + ', '.join(str(b) for b in fit_flags) + ']'
        fit_names = ', '.join(param_names[i] for i in fit_idx)
        w(f'FIT_FLAGS : {flag_str}')
        w(f'Fitted parameters (ndim={len(fit_idx)}): {fit_names}')
        w()

        w('--- Posterior Results ---')
        w(f'{"Parameter":10s} {"Unit":6s} {"Median":>12s} '
          f'{"16th":>12s} {"84th":>12s} {"True":>12s}')
        w('-' * 68)
        truths_iter = active_truths if active_truths is not None else [None] * len(active_names)
        for j, (name, unit, scale, truth) in enumerate(
                zip(active_names, active_units, active_scales, truths_iter)):
            vals          = result.samples[:, j] * scale
            q16, q50, q84 = np.percentile(vals, [16, 50, 84])
            truth_str     = f'{truth * scale:12.4f}' if truth is not None else f'{"N/A":>12s}'
            w(f'{name:10s} {unit:6s} {q50:12.4f} '
              f'{q16:12.4f} {q84:12.4f} {truth_str}')
        if t0_raw is not None:
            _tl, _tu, _ts, _r16, _r50, _r84, _k, _P0 = t0_raw
            w(f'{"T0(raw)":10s} {_tu:6s} {_r50 * _ts:12.4f} '
              f'{_r16 * _ts:12.4f} {_r84 * _ts:12.4f} {"":>12s}')
            w(f'  note: T0 is degenerate mod P; the T0 row above is folded to '
              f'the first-epoch cycle, raw = folded {_k:+d}*P  (P ~ {_P0:.4g} d)')
        w()

        w('--- Solver Diagnostics ---')
        if solver_name == 'emcee':
            nw    = result.chain.shape[1] if result.chain is not None else '?'
            nburn = result.chain_burn.shape[0] if result.chain_burn is not None else '?'
            if nburn != '?' and result.chain is not None and result.chain.shape[0]:
                # stored chains may be thinned (solver_ptemcee PT_THIN): burn-in
                # in real steps, as nsteps_prod
                nburn *= max(1, round(result.nsteps_prod / result.chain.shape[0]))
            w(f'Walkers          : {nw}')
            w(f'Burn-in steps    : {nburn}')
            w(f'Production steps : {result.nsteps_prod}')
            w(f'Acceptance frac  : {result.acceptance_fraction:.3f}')
            # See this function's docstring: use the full joint chain's tau
            # for the convergence verdict when given, not result's own
            # per-planet-sliced tau, which silently misses RV/M_star/
            # parallax nuisance parameters sharing the same chain.
            _tau          = full_tau if full_tau is not None else result.tau
            _names        = full_active_names if full_active_names is not None else active_names
            _nsteps_prod  = full_nsteps_prod if full_nsteps_prod is not None else result.nsteps_prod
            _tau_err      = full_tau_err if full_tau_err is not None else result.tau_err
            if _tau is None:
                w('Autocorrelation  : NOT CONVERGED -- estimation failed (chain too')
                w('                   short relative to tau). Treat the corner/burn-in/')
                w('                   trace plots with suspicion -- apparent bimodality')
                w('                   or non-Gaussian posteriors are the expected symptom.')
                w('                   If the corner plot shows distinct, well-separated')
                w('                   modes, more steps alone may not resolve it (real')
                w('                   degeneracy, e.g. RV epochs many orbital cycles from')
                w('                   the astrometry baseline) -- consider parallel')
                w('                   tempering.')
                w(f'                   Raw (unreliable) estimate: {_tau_err}')
            else:
                w('Autocorrelation times:')
                w(f'  {"Parameter":10s} {"tau (steps)":>12s} {"Need >":>10s}  Status')
                w('  ' + '-' * 46)
                for name, t in zip(_names, _tau):
                    need   = 50 * t
                    status = 'OK' if (np.isfinite(t) and _nsteps_prod > need) else 'NOT CONVERGED'
                    w(f'  {name:10s} {t:12.1f} {need:10.0f}  {status}')
                if not np.all(np.isfinite(_tau)):
                    w('  Max tau = nan (at least one parameter\'s tau estimate failed)')
                    w(f'  production = {_nsteps_prod} steps')
                    w('  Overall: NOT CONVERGED -- see the nan row(s) above; that '
                      'parameter\'s true tau is unknown and may be far longer than '
                      'the others\'.')
                else:
                    tau_max = np.max(_tau)
                    verdict = ('CONVERGED' if _nsteps_prod > 50 * tau_max
                               else f'NOT CONVERGED — need ~{int(50 * tau_max)} steps')
                    w(f'  Max tau = {tau_max:.1f},  production = {_nsteps_prod} steps')
                    w(f'  Overall: {verdict}')
        else:
            w(f'Posterior samples: {len(result.samples)}')
            if result.logz is not None:
                w(f'Log evidence     : {result.logz:.2f} ± {result.logzerr:.2f}')
        w()

        if mass_info is not None and not mass_info.get('is_posterior', False):
            # Point-estimate fallback (RJMCMC/rjflow, or no usable posterior
            # samples): single best-fit values, uncertainty not propagated.
            _lbl = lambda s: f'{s:<35s}'
            w('--- Physical Orbital Parameters (best-fit point estimate) ---')
            w(f'{_lbl("Host star mass (input, external)")}: {mass_info["m_star_msun"]:.4f} Msun')
            w(f'{_lbl("Parallax (input)")}: {mass_info["parallax_mas"]:.4f} mas')
            w(f'{_lbl("Orbital period")}: {mass_info["P_yr"]:.4f} yr')
            w(f'{_lbl("Photocenter semi-major axis")}: {mass_info["a_mas"]:.4f} mas '
              f'({mass_info["a_au"]:.4f} AU)')
            w(f'{_lbl("Orbital inclination (astrometric)")}: {mass_info["i_deg"]:.2f} deg')
            w(f'{_lbl("Longitude of ascending node, Omega")}: {mass_info["Omega_deg"]:.2f} deg'
              + ('' if 'K_ms' in mass_info else '  (or Omega + 180 with omega + 180)'))
            w(f'{_lbl("Argument of periastron, omega")}: {mass_info["omega_deg"]:.2f} deg')
            if 'K_ms' in mass_info:
                w(f'{_lbl("RV semi-amplitude, K")}: {mass_info["K_ms"]:.4f} m/s')
                if mass_info.get('rv_branch_flipped'):
                    w('  (best-fit K came out negative -- astrometry alone cannot tell this')
                    w('   orbit from its (Omega+180, omega+180) twin, only RV data can; K and')
                    w('   omega above are the RV-selected branch, reported with the')
                    w('   conventional K > 0.)')
                if 'gamma_ms' in mass_info:
                    w(f'{_lbl("Systemic velocity, gamma")}: {mass_info["gamma_ms"]:.4f} m/s')
            w()
            w('--- Planet Mass Estimate (best-fit point estimate) ---')
            w(f'{_lbl("Planet mass (astrometric)")}: {mass_info["m_planet_astrometric_mjup"]:.4f} Mjup '
              f'({mass_info["m_planet_astrometric_msun"]:.6f} Msun)')
            if 'm_planet_rv_mjup' in mass_info:
                w(f'{_lbl("Planet mass (RV + astrometric i)")}: {mass_info["m_planet_rv_mjup"]:.4f} Mjup '
                  f'({mass_info["m_planet_rv_msun"]:.6f} Msun)')
                _ratio = mass_info['m_planet_rv_mjup'] / mass_info['m_planet_astrometric_mjup']
                w(f'{_lbl("Astrometric/RV mass ratio")}: {_ratio:.3f}  (1.0 = perfect agreement)')
            w('NOTE: relies on an externally supplied host-star mass (see above);')
            w('      this point estimate does not propagate that mass\'s own')
            w('      uncertainty, nor the fitted posterior\'s spread in P, e, A,')
            w('      B, F, G (median/best-fit values only).')
            w()

        elif mass_info is not None:
            # Posterior propagation: every derived quantity below was
            # computed per posterior sample (mass_estimate.
            # estimate_planet_mass_posterior), not from one fixed point --
            # so this correctly carries the orbital posterior's own spread,
            # and (when PROPAGATE_MSTAR_PARALLAX) the external M_star/parallax
            # uncertainty too. Our equivalent of the credible intervals
            # Stefansson et al. 2025 report on every parameter in Eq. 6.
            _lbl  = lambda s: f'{s:<32s}'
            _draw = mass_info.get('n_draw', '?')

            def _row(label, key, fmt='.4f', unit=''):
                lo, mid, hi = mass_info[key]
                w(f'{_lbl(label)}: {mid:{fmt}} [{lo:{fmt}}, {hi:{fmt}}] {unit}')

            w(f'--- Physical Orbital Parameters (posterior, {_draw} draws, '
              f'16/50/84%) ---')
            _row('Host star mass', 'm_star_msun', unit='Msun')
            _row('Parallax', 'parallax_mas', unit='mas')
            _row('Orbital period', 'P_yr', unit='yr')
            _row('Photocenter semi-major axis', 'a_mas', unit='mas')
            _row('  (same, in AU)', 'a_au', unit='AU')
            _row('Orbital inclination (astrometric)', 'i_deg', unit='deg')
            _row('Longitude of ascending node, Omega', 'Omega_deg',
                 unit='deg' if 'K_ms' in mass_info else 'deg (or +180 with omega + 180)')
            _row('Argument of periastron, omega', 'omega_deg', unit='deg')
            if 'K_ms' in mass_info:
                _row('RV semi-amplitude, K', 'K_ms', unit='m/s')
                _flip_frac = mass_info.get('rv_branch_flipped', 0.0)
                if _flip_frac > 0.0:
                    w(f'  ({_flip_frac*100:.1f}% of draws were on the negative-K/'
                      f'(omega+180) branch -- see the K_MIN comment in '
                      f'ExoPlanetLab.py. K and omega above are canonicalized to')
                    w('   the conventional K > 0 for every draw before taking percentiles.)')
            w()
            w('--- Planet Mass Estimate (posterior) ---')
            _row('Planet mass (astrometric)', 'm_planet_astrometric_mjup', unit='Mjup')
            if 'm_planet_rv_mjup' in mass_info:
                _row('Planet mass (RV + astrometric i)', 'm_planet_rv_mjup', unit='Mjup')
                _ratio = (mass_info['m_planet_rv_mjup'][1]
                          / mass_info['m_planet_astrometric_mjup'][1])
                w(f'{_lbl("Astrometric/RV mass ratio (medians)")}: {_ratio:.3f}  '
                  f'(1.0 = perfect agreement)')
            _psrc = mass_info.get('parallax_source')
            if _psrc:
                w(f'Parallax used for the mass       : {_psrc}')
            w('NOTE: the stellar mass is drawn from its Gaussian prior for each')
            w('      draw when its uncertainty is given (RV runs). The parallax is')
            w('      the orbit-corrected one recovered from the epoch data for')
            w('      each draw (EXOPLANETLAB_MASS_PARALLAX=orbit, DR4 data), or')
            w('      else the external prior; it converts a0 to AU and so enters')
            w('      the astrometric mass (not the RV-based one). Without the')
            w('      uncertainties the inputs are fixed and the spread comes only')
            w('      from the orbital (and K) posterior.')
            w()

        if astro_jitter is not None:
            _j16, _j50, _j84, _jsig = astro_jitter
            w('--- Astrometric Jitter (posterior median [16%, 84%]) ---')
            w(f'jitter = {_j50:.4f} [{_j16:.4f}, {_j84:.4f}] mas   '
              f'(median formal error {_jsig:.4f} mas)')
            w('NOTE: one extra noise term per source, added in quadrature to every')
            w('      along-scan error (EXOPLANETLAB_ASTRO_JITTER). A value well above')
            w('      zero means the formal errors alone do not describe the scatter.')
            w()

        if astro_noise is not None:
            _n16, _n50, _n84 = astro_noise['nu']
            _f16, _f50, _f84 = astro_noise['f']
            w('--- Astrometric Noise: Student-t (posterior median [16%, 84%]) ---')
            w(f'nu = {_n50:.2f} [{_n16:.2f}, {_n84:.2f}]   f = {_f50:.3f} [{_f16:.3f}, {_f84:.3f}]'
              f'   (median formal error {astro_noise["sigma_med"]:.4f} mas)')
            if 'nu_hip' in astro_noise:
                _n16, _n50, _n84 = astro_noise['nu_hip']
                _f16, _f50, _f84 = astro_noise['f_hip']
                w(f'Hipparcos: nu = {_n50:.2f} [{_n16:.2f}, {_n84:.2f}]   f = {_f50:.3f} '
                  f'[{_f16:.3f}, {_f84:.3f}]   (median error {astro_noise["sigma_med_hip"]:.3f} mas)')
            w('NOTE: along-scan errors t_nu(0, f^2 sigma_i^2) per CCD, with the single-star')
            w('      terms fitted robustly (EXOPLANETLAB_ASTRO_NOISE=student). Small nu means')
            w('      heavy tails (outliers); nu -> 100 is Gaussian; f scales the quoted errors.')
            w()

        if hipparcos is not None:
            hp = hipparcos
            w('--- Hipparcos Intermediate Data ---')
            for _ln in hp['warning'].splitlines():
                w(_ln)
            w(f'HIP {hp["hip_id"]}: solution type {hp["isol_n"]}, {hp["n"]} records'
              + ('   SYNTHETIC (injection)' if hp['synthetic'] else ''))
            for _n in hp['notes']:
                w(f'  {_n}')
            if not hp['synthetic']:
                w(f'Perspective acceleration {hp["persp"]:.2f} mas (rv {hp["rv"]:+.2f} km/s). '
                  f'Consistency of the two missions: the fitted frame terms')
                w('(hip_frame_*, hip_spin_*, below; priors 0.6 mas, 0.25 mas/yr) and the '
                  'Hipparcos nu, f (above).')
            w()

        if rel_astrometry is not None:
            ra = rel_astrometry
            f3 = lambda q, nd: f'{q[1]:.{nd}g} [{q[0]:.{nd}g}, {q[2]:.{nd}g}]'
            w('--- Relative (Imaging) Astrometry (posterior median [16%, 84%]) ---')
            w(f'Epochs                : {ra["n_obs"]}')
            w(f's = a_rel / a0        : {f3(ra["s"], 5)}'
              + (f'   (true {ra["true_s"]:g})' if ra.get('true_s') is not None else ''))
            w(f'a_rel (relative orbit): {f3(ra["a_rel"], 5)} mas')
            if 'm_tot' in ra:
                w(f'a_rel                 : {f3(ra["a_rel_au"], 5)} AU   (parallax: {ra["plx_source"]})')
                w(f'Total mass (dynamical): {f3(ra["m_tot"], 4)} Msun')
                w(f'M2 if dark (f_flux=0) : {f3(ra["m2_dark"], 4)} Msun'
                  f' = {f3(tuple(1047.57 * x for x in ra["m2_dark"]), 4)} Mjup')
                if 'm2' in ra:
                    fr = ra['flux_ratio']
                    w(f'Flux ratio L2 / L1 (G): {fr[0]:g}' + (f' +/- {fr[1]:g}' if fr[1] else '')
                      + '   (EXOPLANETLAB_RELAST_FLUX_RATIO)')
                    w(f'Mass fraction M2/M_tot: {f3(ra["mass_fraction"], 4)}   (= 1 / s + f_flux)')
                    w(f'M2 (luminous)         : {f3(ra["m2"], 4)} Msun')
                    w(f'M1 = M_tot - M2       : {f3(ra["m1"], 4)} Msun')
                if ra.get('f_unphysical'):
                    w(f'WARNING: with this flux ratio {100 * ra["f_unphysical"]:.1f}% of the draws '
                      f'give a mass fraction >= 1 (left out above).')
            if 'jitter' in ra:
                w(f'Imaging jitter        : {f3(ra["jitter"], 4)} mas   '
                  f'(median quoted error {ra["sigma_med"]:.4g} mas)')
            w('NOTE: the companion\'s position relative to its host is its photocentre')
            w('      orbit times -s, s = 1 / (f_mass - f_flux), with f_mass = M2 / M_tot the')
            w('      mass fraction and f_flux the flux fraction; M2 = M_tot / s holds for a dark')
            w('      companion only (a luminous one gives a larger M2).')
            w()

        if single_star:
            w('--- Single-star Parameters Corrected for the Orbit '
              '(posterior median [16%, 84%]) ---')
            for _nm, _s16, _s50, _s84, _un in single_star:
                w(f'{_nm:<13s}: {_s50:.4f} [{_s16:.4f}, {_s84:.4f}] {_un}')
            w('NOTE: beta = N^-1 D^T W (d - s(theta)) per posterior draw, plus its')
            w('      conditional scatter N^-1; position offsets are relative to the')
            w('      reference position of the epoch data.')
            w()

        if rv_K is not None:
            _k16, _k50, _k84, _ktrue = rv_K
            w('--- RV Semi-amplitude (posterior median [16%, 84%], |K|) ---')
            _kline = f'K = {_k50:.4f} [{_k16:.4f}, {_k84:.4f}] m/s'
            if _ktrue is not None:
                _kline += f'   (true {_ktrue:.4f} m/s)'
            w(_kline)
            w()

        if rv_instruments:
            w('--- RV Instrument Offsets & Jitter (posterior median [16%, 84%]) ---')
            w(f'{"Instrument":<14s} {"N":>5s} {"gamma (m/s)":>30s} {"jitter (m/s)":>30s}')
            w('-' * 82)

            def _fmt(_inst, _key):
                _m = _inst[f'{_key}_ms']
                if f'{_key}_lo' in _inst:
                    return f'{_m:.1f} [{_inst[f"{_key}_lo"]:.1f}, {_inst[f"{_key}_hi"]:.1f}]'
                return f'{_m:.1f}'

            for _inst in rv_instruments:
                _jstr = (f'held at {_inst["jitter_ms"]:.2f}' if _inst.get('jitter_fixed')
                         else _fmt(_inst, "jitter"))
                w(f'{_inst["name"]:<14s} {_inst["n_obs"]:5d} '
                  f'{_fmt(_inst, "gamma"):>30s} {_jstr:>30s}')
            w('NOTE: for instruments with only 1-2 measurements gamma is largely')
            w('      prior-determined (wide intervals); do not interpret it. Their')
            w('      jitter is not fitted but held at ~0 ("held at"; the data cannot')
            w('      constrain it -- EXOPLANETLAB_RV_JITTER_MIN_N).')
            w('NOTE: each instrument gets its own zero-point offset (gamma) and')
            w('      extra Gaussian noise term (jitter, added in quadrature to the')
            w('      quoted rv_err) -- heterogeneous spectrographs are not assumed')
            w('      to share a common velocity zero-point or noise floor.')
            w()

        w(f'--- Best-fit Residuals (last {int(resid_fraction * 100)}%) ---')
        w(f'Residual mu    : {mu_r:.4f} mas')
        w(f'Residual sigma : {sig_r:.4f} mas')
        w(f'{sigma_label:15s}: {sigma_noise:.4f} mas')
        w()

        w('--- Output Files ---')
        for fname in ['along_scan.png', 'mcmc_burnin.png', 'mcmc_traces.png',
                      'corner_plot.png', 'fit_residuals.png',
                      'orbit_sky.png', 'summary.txt']:
            fpath = os.path.join(outdir, fname)
            if os.path.exists(fpath):
                w(f'  {fpath}')

    print(f"Saved {path}")


def plot_relative_orbit(t_rel, x_obs, y_obs, sx, sy, rows, scales, t_obs, outdir='.'):
    """Relative (imaging) astrometry of one companion: the measured positions
    relative to the host with their errors, and the relative orbit of up to
    100 posterior draws (rows (k, 7) [P, e, T0, A, B, F, G], scales (k,)),
    each over one full period. East is to the left. Saved as
    relative_orbit.png."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for j in range(len(rows)):
        tt = rows[j, 2] + np.linspace(0.0, rows[j, 0], 400)
        xr, yr = orbit.relative_orbit_vec(tt, rows[j][None, :], [scales[j]])
        ax.plot(xr[0], yr[0], color='tab:blue', alpha=0.08, lw=0.8)
    xg, yg = orbit.relative_orbit_vec(np.array([t_obs.min(), t_obs.max()]), rows[:1], scales[:1])
    ax.errorbar(x_obs, y_obs, xerr=sx, yerr=sy, fmt='o', ms=4, color='k',
                ecolor='0.3', zorder=5, label='relative astrometry')
    ax.plot(0, 0, marker='*', ms=14, color='gold', mec='k', zorder=6, label='host')
    ax.plot(xg[0], yg[0], 's', ms=5, mfc='none', mec='tab:red', zorder=6,
            label='Gaia baseline start / end (first draw)')
    ax.set_xlabel(r'$\Delta\alpha^*$ [mas]')
    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc='best')
    ax.set_title('Relative orbit (posterior draws) and imaging astrometry', fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, 'relative_orbit.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_hipparcos(t_year, abscissa, sigma, model, outdir='.', synthetic=False):
    """Hipparcos IAD of the joint fit: the along-scan abscissae with
    the fitted single-star and frame terms removed, the best-fit orbit's
    abscissa at the same records, and the normalised residuals. t_year:
    epochs [yr]; abscissa, model [mas]; sigma: the fitted per-record scale
    (f_hip x de-normalised error) [mas]. Saved as hipparcos.png."""
    t_year, abscissa, sigma, model = (np.asarray(v, dtype=float)
                                      for v in (t_year, abscissa, sigma, model))
    resid = abscissa - model
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    a1.errorbar(t_year, abscissa, yerr=sigma, fmt='o', ms=3, color='k', ecolor='0.6',
                label='Hipparcos abscissa - single-star - frame terms')
    a1.plot(t_year, model, 'x', ms=6, color='tab:red', label='best-fit orbit (along scan)')
    a1.axhline(0.0, color='0.7', lw=0.8)
    a1.set_ylabel('along-scan [mas]')
    a1.legend(fontsize=8, loc='best')
    a1.set_title(('SYNTHETIC ' if synthetic else '') + 'Hipparcos intermediate data, joint fit',
                 fontsize=10)
    a2.plot(t_year, resid / sigma, 'o', ms=3, color='k')
    a2.axhline(0.0, color='0.7', lw=0.8)
    for s in (-3, 3):
        a2.axhline(s, color='0.85', lw=0.8, ls='--')
    a2.set_ylabel(r'residual / $\sigma$')
    a2.set_xlabel('epoch [yr]')
    fig.tight_layout()
    out = os.path.join(outdir, 'hipparcos.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_rv(t_rv, rv_obs, rv_sigma, rv_inst_idx, instruments,
            best_params, joint_samples,
            n_planets, ndim_per,
            true_params_all=None, K_true=None, gamma_true=None,
            outdir='.'):
    """Plot RV time series: observed data, posterior model envelope, residuals.

    Each instrument gets its own fitted (gamma, jitter): the smooth model
    curve is the pure Keplerian signal (gamma=0), and each instrument's
    *data* is plotted offset-subtracted (rv_obs - gamma_instrument) so all
    instruments land on the same zero-point as that curve, colour-coded per
    instrument. Error bars show the jitter-inflated effective uncertainty
    actually used in the fit, not just the quoted rv_sigma.

    Parameters
    ----------
    rv_inst_idx  : (len(t_rv),) int array -- each point's index into `instruments`
    instruments  : list of instrument names, length M
    best_params  : (n_planets*ndim_per + n_planets + 2*M,) best-fit joint params
    joint_samples: (S, n_planets*ndim_per + n_planets + 2*M) posterior samples
    n_planets    : int
    ndim_per     : int  active TI parameter count per planet (7 when all fitted)
    true_params_all : (n_planets, 7) true TI params, or None
    K_true       : sequence of true K values (m/s), or None
    gamma_true   : true systemic velocity (m/s), or None -- only meaningful
                   with a single instrument (SOURCE_MODE='simulated', where
                   the call site already restricts this branch to M=1)
    """
    n_ti        = n_planets * ndim_per
    n_inst      = len(instruments)
    gamma_start = n_ti + n_planets
    jit_start   = gamma_start + n_inst

    def _keplerian(params, t):
        """Pure planetary signal, no instrument offset (gamma=0)."""
        rv = np.zeros(len(t))
        for i in range(n_planets):
            P, e, T0, A, B, F, G = params[i*ndim_per:(i+1)*ndim_per]
            omega = orbit.omega_from_ti(A, B, F, G)
            K_i   = float(params[n_ti + i])
            rv   += orbit.rv_model_single(t, P, e, T0, K_i, omega)
        return rv

    def _model_at(params, t, inst_idx):
        """Keplerian signal plus each point's own instrument offset."""
        gammas = np.asarray(params[gamma_start:gamma_start + n_inst], dtype=float)
        return _keplerian(params, t) + gammas[inst_idx]

    t_fine   = np.linspace(t_rv.min(), t_rv.max(), 1000)
    rv_best  = _keplerian(best_params, t_fine)

    n_draw   = min(200, len(joint_samples))
    idx      = np.random.choice(len(joint_samples), n_draw, replace=False)
    rv_env   = np.array([_keplerian(joint_samples[s], t_fine) for s in idx])

    gammas_best  = np.asarray(best_params[gamma_start:gamma_start + n_inst], dtype=float)
    jitters_best = np.asarray(best_params[jit_start:jit_start + n_inst], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    axes[0].fill_between(t_fine,
                         np.percentile(rv_env, 16, axis=0),
                         np.percentile(rv_env, 84, axis=0),
                         alpha=0.25, color='steelblue', label='68% posterior')
    axes[0].plot(t_fine, rv_best, 'steelblue', lw=1.5,
                 label='Best fit (Keplerian, zero offset)')

    _colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for _i, _name in enumerate(instruments):
        _sel = (rv_inst_idx == _i)
        if not np.any(_sel):
            continue
        _color   = _colors[_i % len(_colors)]
        _sig_eff = np.sqrt(rv_sigma[_sel]**2 + jitters_best[_i]**2)
        axes[0].errorbar(t_rv[_sel], rv_obs[_sel] - gammas_best[_i], yerr=_sig_eff,
                         fmt='.', ms=5, elinewidth=0.8, alpha=0.7, color=_color,
                         label=f'{_name}  (offset {gammas_best[_i]:+.1f}, '
                               f'jitter {jitters_best[_i]:.1f} m/s)')

    if true_params_all is not None and K_true is not None and gamma_true is not None:
        rv_true_fine = np.zeros(len(t_fine))
        for i, ti in enumerate(true_params_all):
            P, e, T0, A, B, F, G = ti
            omega         = orbit.omega_from_ti(A, B, F, G)
            rv_true_fine += orbit.rv_model_single(t_fine, P, e, T0, float(K_true[i]), omega)
        rv_true_fine += float(gamma_true)   # single synthetic instrument
        axes[0].plot(t_fine, rv_true_fine, 'r--', lw=1.5, label='True model')

    axes[0].set_ylabel('Radial velocity (m/s), offset-subtracted')
    axes[0].set_title(f'Radial velocity — {n_planets} planet(s), {n_inst} instrument(s)')
    axes[0].legend(fontsize=8)

    rv_best_obs = _model_at(best_params, t_rv, rv_inst_idx)
    resid       = rv_obs - rv_best_obs
    for _i, _name in enumerate(instruments):
        _sel = (rv_inst_idx == _i)
        if not np.any(_sel):
            continue
        _color   = _colors[_i % len(_colors)]
        _sig_eff = np.sqrt(rv_sigma[_sel]**2 + jitters_best[_i]**2)
        axes[1].errorbar(t_rv[_sel], resid[_sel], yerr=_sig_eff,
                         fmt='.', ms=5, elinewidth=0.8, alpha=0.7, color=_color)
    axes[1].axhline(0, color='steelblue', lw=1)
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Residuals (m/s)')

    fig.tight_layout()
    fname = os.path.join(outdir, 'rv_fit.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")
