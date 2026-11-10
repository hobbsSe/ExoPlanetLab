#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.

"""
Export DR4 pre-release epoch astrometry in Octofitter's CSV format.

Octofitter.jl (v9, GaiaDR4AstromObs; tutorial "Gaia DR4 (Pre-release)") reads
one row per CCD observation with the columns

    source_id, transit_id, epoch, scan_pos_angle, parallax_factor_al,
    centroid_pos_al, centroid_pos_error_al, ipd_error_al, used_by_agis_al,
    outlier_flag

built from the same VOTable ExoPlanetLab reads, with the conversions stated in
the header of its gaia4_epoch_astrometry.csv:

    epoch          obs_time_tcb [ns since JD 2455197.5, TCB] -> MJD
                   (no obs_time_bary_corr: the time at Gaia, not barycentric;
                   ExoPlanetLab adds the correction once -- <= ~0.006 d)
    scan_pos_angle degrees, as published
    parallax_factor_al  per-transit scalar, repeated for each CCD
    outlier_flag   0 where used_by_agis_al, else 1

Rows with any required quantity masked are dropped. The data are ESA's; the
output goes to Results/crosscheck/ (not tracked by git).

Usage:
    python3 crosscheck/export_octofitter_csv.py 4318465066420528000 1457486023639239296
    python3 crosscheck/export_octofitter_csv.py --compare REF.csv 1457486023639239296
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import gaia_dr4_epoch  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'Results', 'crosscheck')
NAMES = {4318465066420528000: 'gaia_bh3', 1457486023639239296: 'gaia4',
         3937211745905473024: 'hd114762'}
MJD_OF_TCB_ZERO = 55197.0            # JD 2455197.5
COLUMNS = ['source_id', 'transit_id', 'epoch', 'scan_pos_angle', 'parallax_factor_al',
           'centroid_pos_al', 'centroid_pos_error_al', 'ipd_error_al',
           'used_by_agis_al', 'outlier_flag']


def _val(a, k):
    """Element k of a (possibly masked) array; None if masked or not finite."""
    if np.ma.is_masked(a) and np.ma.getmaskarray(a)[k]:
        return None
    v = np.asarray(a)[k]
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    return v


def export(df, sid):
    src = df[df['source_id'] == sid].sort_values('transit_id')
    rows, n_all = [], 0
    for _, r in src.iterrows():
        plx_f = r['parallax_factor_al']
        if np.ma.is_masked(plx_f) or not np.isfinite(float(plx_f)):
            n_all += len(r['centroid_pos_al'])
            continue
        for k in range(len(r['centroid_pos_al'])):
            n_all += 1
            vals = [_val(r[c], k) for c in ('obs_time_tcb', 'scan_pos_angle', 'centroid_pos_al',
                                            'centroid_pos_error_al', 'ipd_error_al',
                                            'used_by_agis_al')]
            if any(v is None for v in vals):
                continue
            t_ns, spa, pos, err, ipd, used = vals
            rows.append((int(sid), int(r['transit_id']),
                         MJD_OF_TCB_ZERO + int(t_ns) / 86400e9,
                         float(spa), float(plx_f), float(pos), float(err), float(ipd),
                         bool(used), 0 if bool(used) else 1))
    ra0, dec0 = float(src['ra0'].iloc[0]), float(src['dec0'].iloc[0])
    return rows, n_all, ra0, dec0


def write(rows, n_all, ra0, dec0, sid, path):
    n_tr = len({r[1] for r in rows})
    n_used = sum(r[8] for r in rows)
    with open(path, 'w') as fh:
        fh.write(f"# Gaia DR4 pre-release epoch astrometry, source_id {sid}, in Octofitter's\n"
                 f"# CSV format (written by crosscheck/export_octofitter_csv.py).\n"
                 f"# Copyright (c) European Space Agency / Gaia DPAC.\n"
                 f"# epoch = MJD from obs_time_tcb (TCB, no barycentric correction);\n"
                 f"# scan_pos_angle in degrees; outlier_flag = 0 where used_by_agis_al.\n"
                 f"# Rows with a masked quantity dropped ({n_all - len(rows)} of {n_all}).\n"
                 f"# {len(rows)} CCD observations over {n_tr} transits; {n_used} used_by_agis_al.\n"
                 f"# Reference point for centroid_pos_al: ra0 = {ra0:.9f} deg, dec0 = {dec0:.9f} deg.\n")
        fh.write(','.join(COLUMNS) + '\n')
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.9f},{r[3]:.9f},{r[4]:.8g},{r[5]:.9f},{r[6]:.8g},"
                     f"{r[7]:.8g},{'true' if r[8] else 'false'},{r[9]}\n")
    return n_tr, n_used


def compare(path_ours, path_ref):
    import pandas as pd
    a = pd.read_csv(path_ours, comment='#').sort_values(['transit_id', 'epoch']).reset_index(drop=True)
    b = pd.read_csv(path_ref, comment='#').sort_values(['transit_id', 'epoch']).reset_index(drop=True)
    print(f"rows: ours {len(a)}, reference {len(b)}")
    if len(a) != len(b):
        return
    for c in COLUMNS:
        if a[c].dtype.kind in 'fi':
            d = np.max(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))
            print(f"  {c:22s} max |diff| {d:.3g}" + ('  (days)' if c == 'epoch' else ''))
        else:
            print(f"  {c:22s} identical: {bool((a[c] == b[c]).all())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source_ids', nargs='+', type=int)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--compare', default=None,
                    help="reference CSV (e.g. Octofitter's gaia4_epoch_astrometry.csv) to "
                         "compare the first source's export with")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = gaia_dr4_epoch.load_prerelease_table()
    for sid in args.source_ids:
        rows, n_all, ra0, dec0 = export(df, sid)
        path = os.path.join(args.out_dir, f"{NAMES.get(sid, str(sid))}_epoch_astrometry.csv")
        n_tr, n_used = write(rows, n_all, ra0, dec0, sid, path)
        print(f"{sid}: {len(rows)} CCD rows ({n_all - len(rows)} masked dropped), "
              f"{n_tr} transits, {n_used} used_by_agis_al -> {path}")
        if args.compare and sid == args.source_ids[0]:
            compare(path, args.compare)


if __name__ == '__main__':
    main()
