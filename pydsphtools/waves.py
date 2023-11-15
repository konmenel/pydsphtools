"""Module containing functions useful when working with oceaning waves."""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
from ._waves import (
    find_celerity,
    find_wavenumber,
    ricker_spectrum,
    ricker_spectrum_simple,
    ricker_wavelet_simple,
    wavemaker_transfer_func,
    generate_ricker_signal,
)

__all__ = [
    "find_celerity",
    "find_wavenumber",
    "ricker_spectrum",
    "ricker_spectrum_simple",
    "ricker_wavelet_simple",
    "wavemaker_transfer_func",
    "generate_ricker_signal",
]
