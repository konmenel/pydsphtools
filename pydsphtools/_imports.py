"""
This file is part of Foo Project. It is subject to the license terms in the
LICENSE file found in the top-level directory of this distribution and at
https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of Foo Project,
including this file, may be copied, modified, propagated, or distributed except
according to the terms contained in the LICENSE file.

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
