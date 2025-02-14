"""Module that enables coupling of different DualSPHysics simulations using
the Relaxation Zone technic of DualSPHysics.
"""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
from ._relaxzones import (
    relaxzone_from_dsph,
    write_rzexternal_xml,
)

__all__ = [
    "relaxzone_from_dsph",
    "write_rzexternal_xml",
]
