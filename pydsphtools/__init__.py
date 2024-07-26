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
from . import waves, exceptions, stats, mlpistons, relaxzones
from ._main import __all__ as _all_in_main
# flake8: noqa: F403
from ._main import *

__version__ = "1.0"
__all__ = [
    *_all_in_main,
    "waves",
    "exceptions",
    "stats",
    "mlpistons",
    "relaxzones",
    "io",
]
