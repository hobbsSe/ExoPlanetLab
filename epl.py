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
Run ExoPlanetLab from a configuration file and/or the command line, without
editing ExoPlanetLab.py.

    python3 epl.py examples/gaia4.toml
    python3 epl.py examples/gaia4.toml --set solver=dynesty --tag Gaia-4-dynesty
    python3 epl.py --source-id 4318465066420528000 --rv-mode none
    python3 epl.py examples/injection.toml --dry-run
    python3 epl.py --list-settings

Every setting of a run is one of the EXOPLANETLAB_* variables that
ExoPlanetLab.py reads (user guide, "Environment-Variable Overrides"); this
launcher only gives them a file format and checks their names. A setting is
written without the prefix, in either case:

    # gaia4.toml
    source_mode = "gaia_dr4_prerelease"
    source_id   = 1457486023639239296
    rv_mode     = "external"
    astro_noise = "student"
    evidence    = true                      # booleans become 1 / 0
    inject      = [500, 0.3, 0.4, 60, 30, 90]   # lists become comma-separated

Precedence, highest first: --set and the named options; the configuration
file; EXOPLANETLAB_* variables already in the environment; the defaults in
ExoPlanetLab.py. An unknown name is an error (with the closest known names),
so that a misspelt setting cannot be silently ignored -- which is what happens
to a misspelt environment variable.

The run itself is a fresh `python3 ExoPlanetLab.py` process, as in
Tools/run_batch.py, whose jobs files use the same names.
"""

import argparse
import difflib
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PREFIX = 'EXOPLANETLAB_'
MAIN = os.path.join(ROOT, 'ExoPlanetLab.py')

# the named options: (flag, setting)
SHORTCUTS = (('--source-mode', 'SOURCE_MODE'), ('--source-id', 'SOURCE_ID'),
             ('--solver', 'SOLVER'), ('--rv-mode', 'RV_MODE'), ('--n-fit', 'N_FIT'),
             ('--seed', 'SEED'), ('--tag', 'RUN_TAG'))


def known_settings(root=ROOT):
    """Names (without the prefix) of every EXOPLANETLAB_* variable read by the
    modules next to this file."""
    names = set()
    for path in glob.glob(os.path.join(root, '*.py')):
        with open(path, encoding='utf-8') as fh:
            names.update(re.findall(PREFIX + r'([A-Z][A-Z0-9_]*[A-Z0-9])', fh.read()))
    return names


def to_env_value(value):
    """A TOML / command-line value as the string ExoPlanetLab.py expects."""
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, (list, tuple, dict)) for v in value):
            raise ValueError("nested lists are not a setting")
        return ','.join(to_env_value(v) for v in value)
    if isinstance(value, dict):
        raise ValueError("tables are not supported: write every setting at the top level")
    if isinstance(value, float):
        return repr(value)
    return str(value)


def normalise(name):
    name = name.strip().upper().replace('-', '_')
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def read_config(path):
    """{SETTING: string} from a TOML file."""
    try:
        import tomllib
    except ModuleNotFoundError:             # Python < 3.11
        sys.exit("epl.py needs Python 3.11 or later to read a configuration file "
                 "(tomllib); use --set NAME=VALUE or the EXOPLANETLAB_* variables.")
    with open(path, 'rb') as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as ex:
            sys.exit(f"{path}: not valid TOML ({ex})")
    out = {}
    for key, value in raw.items():
        try:
            out[normalise(key)] = to_env_value(value)
        except ValueError as ex:
            sys.exit(f"{path}: setting '{key}': {ex}")
    return out


def check_names(settings, known, where):
    bad = sorted(k for k in settings if k not in known)
    if bad:
        lines = [f"Unknown setting(s) in {where}:"]
        for k in bad:
            near = difflib.get_close_matches(k, known, n=3, cutoff=0.6)
            lines.append(f"  {k.lower()}" + (f"   (did you mean: {', '.join(n.lower() for n in near)}?)"
                                             if near else ''))
        lines.append("List the known names with:  python3 epl.py --list-settings")
        sys.exit('\n'.join(lines))


def build_settings(args, known):
    """The settings of this run, {SETTING: string}, from file and command line."""
    settings = {}
    if args.config:
        if not os.path.exists(args.config):
            sys.exit(f"Configuration file not found: {args.config}")
        cfg = read_config(args.config)
        check_names(cfg, known, args.config)
        settings.update(cfg)
    cli = {}
    for _flag, name in SHORTCUTS:
        value = getattr(args, name.lower())
        if value is not None:
            cli[name] = str(value)
    for item in args.set or []:
        if '=' not in item:
            sys.exit(f"--set needs NAME=VALUE, got '{item}'")
        key, value = item.split('=', 1)
        cli[normalise(key)] = value.strip()
    check_names(cli, known, 'the command line')
    settings.update(cli)
    return settings


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='epl.py', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', nargs='?', help='configuration file (TOML)')
    ap.add_argument('--set', action='append', metavar='NAME=VALUE',
                    help='one setting; may be repeated')
    for flag, name in SHORTCUTS:
        ap.add_argument(flag, dest=name.lower(), default=None, metavar='VALUE',
                        help=f'same as --set {name.lower()}=VALUE')
    ap.add_argument('--list-settings', action='store_true',
                    help='print the names of all settings and exit')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the settings of the run and exit')
    args = ap.parse_args(argv)

    known = known_settings()
    if args.list_settings:
        print(f"{len(known)} settings (described in the user guide, "
              f"'Environment-Variable Overrides'):")
        for name in sorted(known):
            print(f"  {name.lower()}")
        return 0

    settings = build_settings(args, known)
    inherited = {k[len(PREFIX):]: v for k, v in os.environ.items()
                 if k.startswith(PREFIX) and k[len(PREFIX):] not in settings}
    print(f"ExoPlanetLab run: {len(settings)} setting(s) from "
          f"{'the configuration file and ' if args.config else ''}the command line"
          + (f", {len(inherited)} inherited from the environment" if inherited else '') + ':')
    for k in sorted(settings):
        print(f"  {k.lower():28s} = {settings[k]}")
    for k in sorted(inherited):
        print(f"  {k.lower():28s} = {inherited[k]}   (environment)")
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env.update({PREFIX + k: v for k, v in settings.items()})
    return subprocess.call([sys.executable, MAIN], cwd=ROOT, env=env)


if __name__ == '__main__':
    sys.exit(main())
