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
Shared configuration for all ExoPlanetLab solver backends.

Edit here to change parameters that apply across solvers.
"""

import os

N_MIN = 1   # minimum number of planets in trans-dimensional searches
N_MAX = 3   # maximum number of planets in trans-dimensional searches

# Number of CPU cores available.  Set to an integer to override auto-detection;
# leave as None to use os.cpu_count() automatically.
N_CORES = None

_cpu  = N_CORES if N_CORES is not None else (os.cpu_count() or 2)
N_POOL = max(1, _cpu - 1)   # worker processes (one core reserved for the main process)
# EXOPLANETLAB_N_POOL overrides it -- Tools/run_batch.py sets 1 so that several runs
# can share the machine, one core each, without oversubscribing it.
N_POOL = max(1, int(os.environ.get('EXOPLANETLAB_N_POOL', N_POOL)))


def limit_blas_threads(n=1):
    """Limit OpenBLAS to n threads in the calling process; returns True on success.

    Used as the initializer of every multiprocessing worker pool: the pools
    already use all cores (N_POOL workers), and a multi-threaded BLAS inside
    each worker oversubscribes them (load averages of ~300 were seen on a
    10-core machine once detect.py switched to batched matrix products).
    Environment variables such as OPENBLAS_NUM_THREADS only act before numpy
    is loaded, which is too late in Spyder and in forked workers, so this sets
    the thread count at run time through OpenBLAS's own API. A no-op if the
    library or the call is not found (e.g. a numpy built against Accelerate).
    """
    import ctypes
    import glob
    import sys
    paths = (glob.glob(os.path.join(sys.prefix, 'lib', 'libopenblas*'))
             + glob.glob(os.path.join(os.path.dirname(__import__('numpy').__file__),
                                      '.dylibs', '*openblas*')))
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for fn in ('openblas_set_num_threads', 'scipy_openblas_set_num_threads64_',
                   'openblas_set_num_threads64_'):
            if hasattr(lib, fn):
                getattr(lib, fn)(int(n))
                return True
    return False
