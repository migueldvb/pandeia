# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, absolute_import

"""
These are the wavelength and flux units that are used internally for all calculations.
"""
pandeia_waveunits = "micron"
pandeia_fluxunits = "mjy"

"""
Absolute and relative tolerance values to use to check if values are close enough to be considered equal.
See https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html for details on how they're applied
in comparisons.
"""
pandeia_atol = 1.0e-8
pandeia_rtol = 1.0e-6
