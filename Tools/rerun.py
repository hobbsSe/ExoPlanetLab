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
Repeat an ExoPlanetLab.py run from the run_info.json in its Results folder.

Every setting of ExoPlanetLab.py is an EXOPLANETLAB_* environment variable,
and run_info.json records all of those that were set, together with the seed
of the fit. This script starts ExoPlanetLab.py again with exactly those
settings and that seed. Settings of the calling shell are not passed on, so
nothing else leaks into the run. With the same code and package versions the
new run gives the same samples. The new run writes a new Results folder; the
old one is not touched.

Usage
-----
    python3 Tools/rerun.py Results/<run folder>
    python3 Tools/rerun.py Results/<run folder>/run_info.json
    python3 Tools/rerun.py Results/<run folder> --tag Ref-RV3b
    python3 Tools/rerun.py Results/<run folder> --set SOLVER=emcee --set SEED=
    python3 Tools/rerun.py Results/<run folder> --dry-run

--set KEY=VALUE changes one setting (KEY with or without the EXOPLANETLAB_
prefix); --set KEY= removes it, so that the default is used (SEED= draws a
new seed). --tag sets the run tag of the new folder. --dry-run prints the
command without running it.

A run is not continued from where it stopped: it starts again from the
beginning.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = 'EXOPLANETLAB_'
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _key(k):
    k = k.strip().upper()
    return k if k.startswith(PREFIX) else PREFIX + k


def _git_head():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=ROOT, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return out + (' (modified)' if dirty else '')
    except Exception:
        return 'unknown'


def _version_changes(recorded):
    from importlib import metadata
    changes = []
    for pkg, old in recorded.items():
        if pkg == 'python':
            new = sys.version.split()[0]
        else:
            try:
                new = metadata.version(pkg)
            except Exception:
                new = 'not installed'
        if new != old:
            changes.append(f'{pkg} {old} -> {new}')
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='Results folder of the run, or its run_info.json')
    ap.add_argument('--tag', help='run tag of the new Results folder (default: the old one)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='change a setting; KEY= removes it (repeatable)')
    ap.add_argument('--dry-run', action='store_true', help='print the command only')
    args = ap.parse_args()

    path = args.run
    if os.path.isdir(path):
        path = os.path.join(path, 'run_info.json')
    if not os.path.isfile(path):
        sys.exit(f'No run_info.json at {path}')
    with open(path) as fh:
        info = json.load(fh)

    settings = dict(info.get('settings', {}))
    settings[PREFIX + 'SEED'] = str(info['seed'])
    if args.tag is not None:
        settings[PREFIX + 'RUN_TAG'] = args.tag
    for item in args.set:
        if '=' not in item:
            sys.exit(f'--set needs KEY=VALUE, not {item!r}')
        k, v = item.split('=', 1)
        if v == '':
            settings.pop(_key(k), None)
        else:
            settings[_key(k)] = v

    # The environment of this shell without its EXOPLANETLAB_* settings, then
    # the recorded ones. A run made by run_batch.py (one core) gets one BLAS
    # thread again, as it had then.
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    env.update(settings)
    if settings.get(PREFIX + 'N_POOL') == '1':
        env.update({v: '1' for v in THREAD_VARS})

    print(f'Repeating the run of {info.get("started", "?")} '
          f'(solver {info.get("solver", "?")}, seed {info["seed"]}), with:')
    cmd = [sys.executable, os.path.join(ROOT, 'ExoPlanetLab.py')]
    print(' \\\n'.join(f'  {k}={settings[k]}' for k in sorted(settings))
          + ' \\\n    python3 ExoPlanetLab.py')
    if 'finished' in info and not info['finished']:
        print('Note: the original run did not finish.')
    old_git, new_git = info.get('git', 'unknown'), _git_head()
    if old_git != new_git or old_git.endswith('(modified)'):
        print(f'Note: the code was {old_git} then and is {new_git} now; '
              f'the samples may differ from those of the original run.')
    changes = _version_changes(info.get('versions', {}))
    if changes:
        print('Note: package versions differ: ' + ', '.join(changes))
    if args.dry_run:
        return
    sys.exit(subprocess.run(cmd, cwd=ROOT, env=env).returncode)


if __name__ == '__main__':
    main()
