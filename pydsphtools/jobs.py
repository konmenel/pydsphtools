"""Module that handles DualSPHysics jobs. Jobs are sequential runs of the DualSPHysics solver, pre-proccessing
tools or post-proccessing tools.
"""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.

from ._jobs import Binary, Dualsphysics, Gencase, Job

__all__ = ["Binary", "Dualsphysics", "Gencase", "Job"]
