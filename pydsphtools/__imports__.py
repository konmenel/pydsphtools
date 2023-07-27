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
import os
import io
import re
import sys
import glob
import errno
import pathlib
import platform
import subprocess
from typing import Callable, TypeVar, Union, Tuple, List, Dict, Iterable, Optional

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from scipy import optimize, interpolate
import pandas as pd
import lxml.etree as ET
