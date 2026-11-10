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
Run many ExoPlanetLab.py jobs side by side, one core each.

Why
---
After vectorising the likelihood, one run no longer needs more than about one
core: the remaining per-step cost is the sampler's own serial bookkeeping.
Running several sources (or configurations) at the same time, each limited to
one core, therefore gives close to N_cores times the throughput of running
them one after another -- the way to process DR4-scale samples.

Each job is a fresh `python3 ExoPlanetLab.py` process whose settings come from
EXOPLANETLAB_* environment variables (see the user guide, "Environment-Variable
Overrides"). Every job gets
    EXOPLANETLAB_N_POOL=1, EXOPLANETLAB_VECTORIZE_WORKERS=1
    OPENBLAS/OMP/MKL/VECLIB thread counts = 1
    EXOPLANETLAB_RUN_TAG=<job name>   (appended to its Results/ directory)
so that the jobs share the machine without oversubscribing it.

Usage
-----
From a jobs file (one job per line: a name, then KEY=VALUE settings; KEY is an
EXOPLANETLAB_* variable with or without the prefix; '#' starts a comment):

    python3 Tools/run_batch.py jobs.txt
    python3 Tools/run_batch.py jobs.txt --max-parallel 6 --dry-run
    python3 Tools/run_batch.py jobs.txt --only '^Gaia'      # the matching jobs only

    # jobs.txt
    Gaia-BH3-NoRV    SOURCE_ID=4318465066420528000 RV_MODE=none
    Gaia-BH3-RV      SOURCE_ID=4318465066420528000 RV_MODE=external
    Gaia-4-RV        SOURCE_ID=1457486023639239296 RV_MODE=external SOLVER=dynesty

Or build the job list on the command line (every source x every RV mode):

    python3 Tools/run_batch.py --sources 4318465066420528000 1457486023639239296 \\
                         --rv none external --solver emcee

The console shows one line per started and finished job; each job's full
output goes to Results/batch_<stamp>/<name>.log (as well as to the run.log of
its own Results/ directory), and a summary table is written to
Results/batch_<stamp>/summary.txt.
"""

import argparse
import datetime
import glob
import itertools
import os
import shlex
import signal
import subprocess
import sys
import time

# the main code, data and Results/ live one level up (this script is in Tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXOPLANETLAB_PY = os.path.join(ROOT, 'ExoPlanetLab.py')

ONE_CORE_ENV = {
    'EXOPLANETLAB_N_POOL': '1',
    'EXOPLANETLAB_VECTORIZE_WORKERS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}

KNOWN_NAMES = {   # used only to label jobs built from --sources
    '4318465066420528000': 'Gaia-BH3',
    '1457486023639239296': 'Gaia-4',
    '3937211745905473024': 'HD-114762',
}


def parse_jobs_file(path):
    jobs = []
    for n, line in enumerate(open(path), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = shlex.split(line)
        name, settings = parts[0], {}
        for kv in parts[1:]:
            if '=' not in kv:
                raise SystemExit(f"{path}:{n}: expected KEY=VALUE, got {kv!r}")
            k, v = kv.split('=', 1)
            k = k.strip().upper()
            if not k.startswith('EXOPLANETLAB_'):
                k = 'EXOPLANETLAB_' + k
            settings[k] = v.strip()
        jobs.append((name, settings))
    return jobs


def jobs_from_args(args):
    jobs = []
    for sid, rv in itertools.product(args.sources, args.rv):
        name = f"{KNOWN_NAMES.get(sid, sid)}-{'NoRV' if rv == 'none' else 'RV' if rv == 'external' else rv}"
        settings = {'EXOPLANETLAB_SOURCE_ID': sid, 'EXOPLANETLAB_RV_MODE': rv,
                    'EXOPLANETLAB_SOURCE_MODE': args.source_mode}
        if args.solver:
            settings['EXOPLANETLAB_SOLVER'] = args.solver
        jobs.append((name, settings))
    return jobs


def results_dir_for(name, settings):
    """The job's Results/ directory, '<N>plt_<solver>_<stamp>-<tag>': the tag is
    the job name without a trailing '-<solver>' (ExoPlanetLab.py drops it, the
    solver being in the name already), so match on the solver as well -- the job
    'Gaia-BH3-RV' must not pick up the dynesty run of job 'Gaia-BH3-RV-dynesty'."""
    solver = settings.get('EXOPLANETLAB_SOLVER', 'emcee').strip().lower()
    tag = name[:-(len(solver) + 1)] if name.lower().endswith('-' + solver) else name
    hits = sorted(glob.glob(os.path.join(ROOT, 'Results', f'*plt_{solver}_*-{tag}')),
                  key=os.path.getmtime)
    return os.path.relpath(hits[-1], ROOT) if hits else '?'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jobs_file', nargs='?', help='jobs file (see above)')
    ap.add_argument('--sources', nargs='+', help='source_ids to run (instead of a jobs file)')
    ap.add_argument('--rv', nargs='+', default=['none'], choices=['none', 'external', 'simulated'],
                    help='RV modes to run for each source (default: none)')
    ap.add_argument('--solver', default=None, help='SOLVER for every job built from --sources')
    ap.add_argument('--source-mode', default='gaia_dr4_prerelease',
                    help='SOURCE_MODE for jobs built from --sources')
    ap.add_argument('--max-parallel', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help='jobs running at once (default: cores - 1)')
    ap.add_argument('--stagger', type=float, default=2.0,
                    help='seconds between job starts (default 2)')
    ap.add_argument('--only', default=None, metavar='REGEX',
                    help='run only the jobs whose name matches this regular expression '
                         '(e.g. --only "^(Gaia|HD)" for the DR4 jobs of jobs_reference.txt)')
    ap.add_argument('--dry-run', action='store_true', help='print the jobs and exit')
    args = ap.parse_args()

    if args.jobs_file:
        jobs = parse_jobs_file(args.jobs_file)
    elif args.sources:
        jobs = jobs_from_args(args)
    else:
        ap.error('give a jobs file or --sources')
    if args.only:
        import re
        jobs = [(n, st) for n, st in jobs if re.search(args.only, n)]
        if not jobs:
            raise SystemExit(f'no job name matches --only {args.only!r}')
    names = [n for n, _ in jobs]
    job_settings = dict(jobs)
    if len(set(names)) != len(names):
        raise SystemExit('job names must be unique (they label the Results/ directories)')

    print(f"{len(jobs)} job(s), up to {args.max_parallel} at a time, one core each:")
    for name, st in jobs:
        print(f"  {name:28s} " + ' '.join(f"{k.replace('EXOPLANETLAB_', '')}={v}" for k, v in st.items()))
    if args.dry_run:
        return

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = os.path.join(ROOT, 'Results', f'batch_{stamp}')
    os.makedirs(logdir)

    pending = list(jobs)
    running = {}      # name -> (Popen, start time, log file handle)
    finished = []     # (name, status, returncode, elapsed_s)
    t_batch = time.time()

    def _stop_all(*_):
        print("\nInterrupted -- terminating running jobs ...", flush=True)
        for p, _, fh in running.values():
            p.terminate()
        for p, _, fh in running.values():
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _stop_all)
    signal.signal(signal.SIGTERM, _stop_all)

    while pending or running:
        while pending and len(running) < args.max_parallel:
            name, st = pending.pop(0)
            env = os.environ.copy()
            env.update(ONE_CORE_ENV)
            env.update(st)
            env['EXOPLANETLAB_RUN_TAG'] = name
            fh = open(os.path.join(logdir, f'{name}.log'), 'w')
            p = subprocess.Popen([sys.executable, EXOPLANETLAB_PY], cwd=ROOT,
                                 env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[name] = (p, time.time(), fh)
            print(f"[{datetime.datetime.now():%H:%M:%S}] started  {name}  "
                  f"({len(running)} running, {len(pending)} waiting)", flush=True)
            if pending:
                time.sleep(args.stagger)
        for name in list(running):
            p, t0, fh = running[name]
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            dt = time.time() - t0
            status = 'OK' if rc == 0 else f'FAIL({rc})'
            finished.append((name, status, rc, dt))
            del running[name]
            print(f"[{datetime.datetime.now():%H:%M:%S}] finished {name}  {status}  "
                  f"{dt/60:.1f} min  -> {results_dir_for(name, job_settings[name])}", flush=True)
        time.sleep(2.0)

    total = time.time() - t_batch
    lines = [f"Batch {stamp}: {len(finished)} job(s), up to {args.max_parallel} in parallel, "
             f"total {total/60:.1f} min (sum of job times {sum(f[3] for f in finished)/60:.1f} min)",
             f"{'job':28s} {'status':10s} {'minutes':>8s}  results"]
    for name, status, rc, dt in finished:
        lines.append(f"{name:28s} {status:10s} {dt/60:8.1f}  {results_dir_for(name, job_settings[name])}")
    open(os.path.join(logdir, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f"\nLogs and summary: {os.path.relpath(logdir, ROOT)}/")
    if any(f[2] != 0 for f in finished):
        sys.exit(1)


if __name__ == '__main__':
    main()
