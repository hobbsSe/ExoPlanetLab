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
Shared constants for ExoPlanetLab.
"""

import numpy as np

# ── Unit conversions ──────────────────────────────────────────────────────────
rad2deg  = 180.0 / np.pi
JulianYr = 365.25          # days per Julian year
JulianDay = 24.0 * 3600.0 # seconds per day

# ── Eccentricity floor ────────────────────────────────────────────────────────
# Prevents e = 0 (circular) or e ≥ 1 (hyperbolic) which would break kepler_eq.
epsilon = 1e-4
