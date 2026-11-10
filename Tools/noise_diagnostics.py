#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Diagnostics of the astrometric residuals of the Gaia DR4 pre-release sources
: is the noise beyond the quoted errors correlated within a transit,
or does it depend on anything the noise model ignores?

The noise model of the fits is one Student-t distribution per CCD measurement,
t_nu(0, (f sigma_i)^2), independent between measurements. For every source the
residuals of the robust single-star fit (for the three companions: after
subtracting the orbit, the mean model of posterior draws of the latest
EV-<name>-N1 run) are normalised, z = r / (f sigma), and tested:

  transit   the nine CCD measurements of a field-of-view transit share the
            attitude; a common error would make the transit medians of z
            scatter more than independent noise allows. Permutation test (z
            shuffled between measurements), the excess expressed as a
            per-transit term in mas, with a 95% upper limit; and the rank
            correlation of the CCDs of a transit against their separation.
  time      rank correlation of the medians of consecutive transits.
  factors   location (Kruskal-Wallis) and scale (Fligner-Killeen) of z against
            field of view, CCD strip, CCD row, gate, time, scan angle,
            sub-pixel phase and across-scan position.
  gates     for the sources observed with more than one gate: the scale of z
            per gate, and the fit repeated with one error factor f per gate
            (gates with fewer than 20 measurements share one), to see how
            much of the heavy tail is a gate with underestimated errors.

p-values are per test; with about a dozen tests on each of twelve sources, one
or two below 0.01 are expected by chance.

    python3 Tools/noise_diagnostics.py                # all twelve sources
    python3 Tools/noise_diagnostics.py --n-perm 5000

Writes Results/noise_diagnostics.txt and .npz.
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gaia_dr4_epoch   # noqa: E402
import limits           # noqa: E402
import orbit            # noqa: E402
import robust           # noqa: E402

COMPANIONS = {1457486023639239296: 'Gaia-4', 4318465066420528000: 'Gaia-BH3',
              3937211745905473024: 'HD-114762'}
NS_PER_DAY = 86400.0e9


def _arr(v):
    return np.asarray(np.ma.filled(np.ma.asarray(v, dtype=float), np.nan), dtype=float)


def metadata(df, source_id, t_fit, psi_fit):
    """Per-measurement metadata of the rows used in the fit, matched to the
    loader's rows by scan angle and time order."""
    sub = df[df['source_id'] == source_id]
    rows = []
    for _, r in sub.iterrows():
        used = np.asarray(np.ma.filled(np.ma.asarray(r['used_by_agis_al']), False), dtype=bool)
        t = _arr(r['obs_time_tcb'])
        ang, gate, spx, mu = (_arr(r[c]) for c in ('scan_pos_angle', 'gates', 'sub_pixel_coord', 'mu'))
        tid = int(r['transit_id'])
        for k in range(1, 10):                       # AF1 ... AF9 (index 0 is the sky mapper)
            if used[k] and np.isfinite(t[k]) and np.isfinite(ang[k]):
                rows.append((t[k] + float(r['obs_time_bary_corr']), ang[k], tid, k,
                             1 + ((tid >> 15) & 0x03), (tid >> 12) & 0x07, gate[k], spx[k], mu[k]))
    m = np.array(rows, dtype=float)
    if len(m) != len(t_fit):
        raise RuntimeError(f"{source_id}: {len(m)} flagged measurements, {len(t_fit)} in the fit")
    # same order: sort both by time (ties broken by scan angle)
    o_raw = np.lexsort((m[:, 1], m[:, 0]))
    o_fit = np.lexsort((psi_fit, t_fit))
    m = m[o_raw]
    dpsi = np.degrees(np.angle(np.exp(1j * (np.radians(m[:, 1]) - psi_fit[o_fit]))))
    dt = np.diff(m[:, 0]) / NS_PER_DAY - np.diff(t_fit[o_fit])
    if np.max(np.abs(dpsi)) > 1e-3 or np.max(np.abs(dt)) > 1e-4:
        raise RuntimeError(f"{source_id}: rows do not match (scan angle {np.max(np.abs(dpsi)):.2g} deg, "
                           f"time {np.max(np.abs(dt)):.2g} d)")
    out = np.empty_like(m)
    out[o_fit] = m
    names = ('time_ns', 'psi_deg', 'transit', 'strip', 'fov', 'row', 'gate', 'subpix', 'mu')
    return {n: out[:, j] for j, n in enumerate(names)}


def orbit_model(source_id, t, psi, res, tag, n_draw=300):
    """Mean along-scan model of posterior draws of the latest run with the tag."""
    name = COMPANIONS[source_id]
    runs = sorted(glob.glob(os.path.join(res, f'1plt_*-{tag.format(name=name)}')), key=os.path.getmtime)
    runs = [r for r in runs if os.path.exists(os.path.join(r, 'posterior_samples.npz'))]
    if not runs:
        return None, None
    S = np.load(os.path.join(runs[-1], 'posterior_samples.npz'), allow_pickle=True)['samples']
    idx = np.random.default_rng(0).choice(len(S), min(n_draw, len(S)), replace=False)
    return orbit.orbital_model_AL_vec(t, S[idx, :7], psi).mean(axis=0), os.path.basename(runs[-1])


def _fit(y, X, sig, lam):
    """nu, f, ln L and the normalised residuals of the robust single-star fit."""
    nu, f = limits.noise_fit(y, X, sig, lam=lam)
    rob = robust.StudentTSingleStar(y, X, sig, lam=lam)
    lnL = float(rob.loglike(np.zeros((1, len(y))), np.array([nu]), np.array([f]))[0])
    _, r, _ = rob.fit_beta(y[None, :], (f * sig)[None, :] ** 2, np.full((1, 1), nu))
    return nu, f, lnL, r[0] / (f * sig)


def residuals(d, model=None):
    """Normalised residuals z of the robust single-star fit, and nu, f."""
    y = np.asarray(d['proj_data'], float) - (0.0 if model is None else model)
    X, sig = np.asarray(d['proj_design'], float), np.asarray(d['sigma_mas'], float)
    nu, f, _, z = _fit(y, X, sig, d.get('proj_prior_prec'))
    return z, nu, f


def _mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def gate_fit(d, model, z, gate, transit, n_min=20, n_iter=8):
    """Scale of z per gate, and the fit with one error factor per gate.

    Gates with fewer than n_min measurements share one factor (group -1).
    Returns None for a source observed with a single gate."""
    gate = gate.astype(int)
    vals = np.unique(gate)
    if len(vals) < 2:
        return None
    table = [(int(v), int(np.sum(gate == v)), len(np.unique(transit[gate == v])), _mad(z[gate == v]),
              float(np.mean(np.abs(z[gate == v]) > 3.0))) for v in vals]
    y = np.asarray(d['proj_data'], float) - (0.0 if model is None else model)
    X, sig = np.asarray(d['proj_design'], float), np.asarray(d['sigma_mas'], float)
    lam = d.get('proj_prior_prec')
    grp = np.where(np.isin(gate, [v for v, n, *_ in table if n >= n_min]), gate, -1)
    nu0, _, lnL0, _ = _fit(y, X, sig, lam)
    c = {int(v): 1.0 for v in np.unique(grp)}
    for it in range(n_iter + 1):                     # alternate (nu, f) and the factors
        nu, f, lnL, zg = _fit(y, X, sig * np.array([c[v] for v in grp]), lam)
        if it < n_iter:
            for v in c:
                c[v] *= stats.t.fit(zg[grp == v], fdf=nu, floc=0.0)[2]
    k = len(c) - 1
    return dict(table=table, nu_common=nu0, nu=nu, f={v: c[v] * f for v in sorted(c)},
                dlnL=lnL - lnL0, extra=k, p=float(stats.chi2.sf(2.0 * max(lnL - lnL0, 0.0), k)),
                frac3=float(np.mean(np.abs(zg) > 3.0)))


def transit_test(z, transit, scale_mas, n_perm, rng):
    """Excess scatter of the transit medians of z over independent noise."""
    ids, inv = np.unique(transit, return_inverse=True)
    cnt = np.bincount(inv)
    keep = cnt[inv] >= 3
    order = np.argsort(inv[keep], kind='stable')
    zz, gi = z[keep][order], inv[keep][order]
    bounds = np.flatnonzero(np.diff(gi)) + 1

    def stat(v):
        med = np.array([np.median(g) for g in np.split(v, bounds)])
        return float(np.mean(np.clip(med, -4.0, 4.0) ** 2))    # clipped: robust to one bad transit

    obs = stat(zz)
    null = np.array([stat(rng.permutation(zz)) for _ in range(n_perm)])
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    exc = obs - null.mean()
    upper = obs - np.quantile(null, 0.05)
    # the transit median of n values has variance ~ (pi / 2) / n of one value;
    # an added common term s (in units of f sigma) adds s^2
    return dict(n_transits=int(len(bounds) + 1), obs=obs, null=float(null.mean()), p=p,
                s_mas=float(np.sqrt(max(exc, 0.0)) * scale_mas),
                s_upper_mas=float(np.sqrt(max(upper, 0.0)) * scale_mas))


def strip_pairs(z, transit, strip, sep=1):
    """Pairs of z on CCD strips sep apart in the same transit, (n, 2)."""
    key = {(a, b): v for a, b, v in zip(transit, strip, z)}
    return np.array([(v, key[(a, b + sep)]) for (a, b), v in key.items() if (a, b + sep) in key])


def adjacent_corr(z, transit, strip, sep=1):
    """Spearman correlation of z on CCD strips sep apart in the same transit."""
    pairs = strip_pairs(z, transit, strip, sep)
    if len(pairs) < 10:
        return np.nan, np.nan, 0
    rho, p = stats.spearmanr(pairs[:, 0], pairs[:, 1])
    return float(rho), float(p), len(pairs)


def lag_corr(z, transit, time):
    """Spearman correlation of the medians of consecutive transits."""
    ids = np.unique(transit)
    med = np.array([np.median(z[transit == i]) for i in ids])
    tt = np.array([np.mean(time[transit == i]) for i in ids])
    med = med[np.argsort(tt)]
    rho, p = stats.spearmanr(med[:-1], med[1:])
    return float(rho), float(p)


def factor_tests(z, meta, t_day, psi):
    """{factor: (p location, p scale, number of groups)}."""
    def bins(v, n):
        return np.searchsorted(np.quantile(v, np.linspace(0, 1, n + 1)[1:-1]), v)

    factors = dict(fov=meta['fov'], strip=meta['strip'], row=meta['row'], gate=meta['gate'],
                   time=bins(t_day, 8), scan_angle=bins(np.mod(psi, np.pi), 8),
                   subpixel=bins(np.mod(meta['subpix'], 1.0), 4), ac_position=bins(meta['mu'], 4))
    out = {}
    for name, g in factors.items():
        groups = [z[g == v] for v in np.unique(g[np.isfinite(g)]) if np.sum(g == v) >= 8]
        if len(groups) < 2:
            out[name] = (np.nan, np.nan, len(groups))
            continue
        out[name] = (float(stats.kruskal(*groups).pvalue),
                     float(stats.fligner(*groups).pvalue), len(groups))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-perm', type=int, default=2000)
    ap.add_argument('--tag', default='EV-{name}-N1', help='run tag of the companions\' orbits')
    ap.add_argument('--results', default=os.path.join(ROOT, 'Results'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    df = gaia_dr4_epoch.load_prerelease_table()
    fac_names = ('fov', 'strip', 'row', 'gate', 'time', 'scan_angle', 'subpixel', 'ac_position')
    L = ["Noise diagnostics of the DR4 pre-release sources (Tools/noise_diagnostics.py).",
         "z = residual / (f sigma) of the robust single-star fit (companions: orbit subtracted).", ""]
    T = [f"{'source':>20s} {'G':>5s} {'N':>4s} {'nu':>6s} {'f':>5s} {'transits':>8s} "
         f"{'var(med z)':>10s} {'null':>6s} {'p':>6s} {'s_tr [mas]':>10s} {'95% upper':>9s} "
         f"{'med f*sig':>9s} {'adj. rho':>8s} {'p':>6s} {'lag rho':>8s} {'p':>6s}"]
    F = [f"{'source':>20s} " + ' '.join(f"{n:>17s}" for n in fac_names),
         f"{'':>20s} " + ' '.join(f"{'p loc / p scale':>17s}" for _ in fac_names)]
    seps = (1, 2, 3, 4)
    C = [f"{'source':>20s} " + ' '.join(f"{f'{k} apart: rho / p':>17s}" for k in seps)]
    G = []
    store, pooled = {}, {}
    for sid in sorted(set(int(v) for v in df['source_id'])):
        d = gaia_dr4_epoch.load_source_epoch_data(sid)
        t, psi = np.asarray(d['t_obs_day'], float), np.asarray(d['scan_angle_rad'], float)
        model, run = (orbit_model(sid, t, psi, args.results, args.tag) if sid in COMPANIONS
                      else (None, None))
        if sid in COMPANIONS and model is None:
            L.append(f"{sid}: no run with tag {args.tag}; skipped")
            continue
        z, nu, f = residuals(d, model)
        meta = metadata(df, sid, t, psi)
        scale = float(np.median(f * np.asarray(d['sigma_mas'], float)))
        tr = transit_test(z, meta['transit'], scale, args.n_perm, rng)
        rho_a, p_a, _ = adjacent_corr(z, meta['transit'], meta['strip'])
        rho_l, p_l = lag_corr(z, meta['transit'], meta['time_ns'])
        fac = factor_tests(z, meta, t, psi)
        sep = [adjacent_corr(z, meta['transit'], meta['strip'], k) for k in seps]
        gf = gate_fit(d, model, z, meta['gate'], meta['transit'])
        label = f"{sid}" + ('*' if sid in COMPANIONS else '')
        T.append(f"{label:>20s} {d.get('g_mag', np.nan):5.1f} {len(z):4d} {nu:6.2f} {f:5.2f} "
                 f"{tr['n_transits']:8d} {tr['obs']:10.3f} {tr['null']:6.3f} {tr['p']:6.3f} "
                 f"{tr['s_mas']:10.4f} {tr['s_upper_mas']:9.4f} {scale:9.4f} "
                 f"{rho_a:+8.3f} {p_a:6.3f} {rho_l:+8.3f} {p_l:6.3f}")
        F.append(f"{label:>20s} " + ' '.join(f"{fac[n][0]:8.3f} /{fac[n][1]:7.3f}" for n in fac_names))
        C.append(f"{label:>20s} " + ' '.join(f"{r:+8.3f} /{p:7.3f}" for r, p, _ in sep))
        if gf is not None:
            G.append(f"{label}: nu {gf['nu_common']:.2f} -> {gf['nu']:.2f} with one f per gate; "
                     f"d ln L {gf['dlnL']:+.2f} for {gf['extra']} more parameters (p {gf['p']:.2g}); "
                     f"|z| > 3: {100 * gf['frac3']:.1f}%")
            G += [f"    gate {v:2d}: {n:4d} measurements in {nt:3d} transits, robust scale of z {s:5.2f}, "
                  f"|z| > 3: {100 * f3:4.1f}%, f {gf['f'][v if v in gf['f'] else -1]:5.2f}"
                  + ('' if v in gf['f'] else ' (shared)') for v, n, nt, s, f3 in gf['table']]
        store[str(sid)] = dict(z=z, nu=nu, f=f, transit_test=tr, adjacent=(rho_a, p_a), lag=(rho_l, p_l),
                               separations=sep, gates=gf,
                               factors=fac, run=run, **{k: meta[k] for k in ('transit', 'strip', 'fov', 'row', 'gate')})
        if sid not in COMPANIONS:
            for k in seps:
                pooled.setdefault(('sep', k), []).append(strip_pairs(z, meta['transit'], meta['strip'], k))
            for k in ('strip', 'fov', 'row', 'gate'):
                pooled.setdefault(k, []).append(meta[k])
            pooled.setdefault('z', []).append(z)
    L += ["Per-transit term and correlations (* = companion, orbit subtracted):"] + T + [""]
    for k in seps:
        if ('sep', k) in pooled:
            pr = np.concatenate(pooled[('sep', k)])
            pooled[('sep', k)] = stats.spearmanr(pr[:, 0], pr[:, 1])
    L += ["Correlation of z between the CCDs of a transit against their separation in strips (Spearman):"] + C
    L += [f"{'pooled, no companion':>20s} " + ' '.join(f"{pooled[('sep', k)][0]:+8.3f} /{pooled[('sep', k)][1]:7.3f}"
                                                        for k in seps if ('sep', k) in pooled), ""]
    L += ["Gates (sources observed with more than one):"] + G + [""]
    L += ["Factors: p-value of a difference in location (Kruskal-Wallis) / in scale (Fligner-Killeen):"] + F + [""]
    if pooled:
        zp = np.concatenate(pooled['z'])
        L.append(f"Pooled over the {len(pooled['z'])} sources without a companion ({len(zp)} measurements):")
        for k in ('fov', 'strip', 'row', 'gate'):
            g = np.concatenate(pooled[k])
            vals = [v for v in np.unique(g[np.isfinite(g)]) if np.sum(g == v) >= 20]
            groups = [zp[g == v] for v in vals]
            if len(groups) < 2:
                continue
            L.append(f"  {k:6s} p loc {stats.kruskal(*groups).pvalue:.3f}, p scale "
                     f"{stats.fligner(*groups).pvalue:.3f};  robust scale of z per group: "
                     + ', '.join(f"{int(v)}: {1.4826 * np.median(np.abs(x - np.median(x))):.2f}"
                                 for v, x in zip(vals, groups)))
    text = '\n'.join(L)
    print(text)
    out = os.path.join(args.results, 'noise_diagnostics')
    with open(out + '.txt', 'w') as fh:
        fh.write(text + '\n')
    np.savez_compressed(out + '.npz', results=np.array(store, dtype=object))
    print(f"\nWritten: {os.path.relpath(out)}.txt, .npz")


if __name__ == '__main__':
    main()
