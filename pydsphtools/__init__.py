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

Author: Constantinos Menelaou  
Github: https://github.com/konmenel  
Year: 2023  
"""
from . import waves, exceptions, stats, mlpistons, relaxzones
from ._main import (
    DEG2RAD,
    RAD2DEG,
    RE_PATTERNS,
    read_and_fix_csv,
    get_dp,
    get_var,
    get_usr_def_var,
    get_chrono_mass,
    get_chrono_inertia,
    get_chrono_property,
    run_measuretool,
    xml_get_or_create_subelement,
    get_dualsphysics_root,
    get_times_of_partfiles,
    get_number_of_partfiles,
)

__version__ = "1.0"
__all__ = [
    "DEG2RAD",
    "RAD2DEG",
    "RE_PATTERNS",
    "read_and_fix_csv",
    "get_dp",
    "get_var",
    "get_usr_def_var",
    "get_chrono_mass",
    "get_chrono_inertia",
    "get_chrono_property",
    "run_measuretool",
    "xml_get_or_create_subelement",
    "get_dualsphysics_root",
    "get_times_of_partfiles",
    "get_number_of_partfiles",
    "waves",
    "exceptions",
    "stats",
    "mlpistons",
    "relaxzones",
]
