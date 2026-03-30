"""Copyright (C) 2023 Constantinos Menelaou <https://github.com/konmenel>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Constantinos Menelaou<br>
Github: https://github.com/konmenel <br>
Year: 2023<br>
"""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
from ._version import __version__
from . import waves, exceptions, stats, mlpistons, relaxzones, io, jobs, postprocess
from ._main import __all__ as _all_in_main

# flake8: noqa: F403
from ._main import *

__all__ = [
    *_all_in_main,
    "__version__",
    "waves",
    "exceptions",
    "stats",
    "mlpistons",
    "relaxzones",
    "io",
    "jobs",
    "postprocess",
]
