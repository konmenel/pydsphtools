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

@author: Constantinos Menelaou
@github: https://github.com/konmenel
@year: 2023
"""
import io
import re
import pathlib
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

# BUG: There is a bug with the way DualSPHysics creates the `Run.csv`. It replaces all `,` with `;`
#   which causes the Shifting(_txt_,_num_,_num_,_txt_) in the configure section to become
#   Shifting(_txt_;_num_;_num_;_txt_). This causes a parsing bug with `pandas.read_csv` (any csv parser really)

R = TypeVar("R")


class RE_PATTERNS:
    """Just constant class to store useful regex. Will move into a file if it grows too large."""
    # pattern to capture any number (eg 1.23, -1523, -12.3e-45)
    NUMBER = r"[\-\+]?\d+\.?\d*[Ee]?[\+\-]?\d*"
    # Pattern to captures the chrono floating section of the output. Returns the
    # "ID" and "name" of the chorno object floating
    FLOATING = r"Body_(?P<ID>\d+) \"(?P<name>\w*)\" -  type: Floating"


class NotFoundInOutput(Exception):
    """Raised when a variable is not found in the output file.

    Attributes
    ----------
    missing : str
        What was not found.
    filename : str, optional
        The name of the output file (either `Run.out` or `Run.csv`). By default, `Run.out`.
    """

    missing: str
    filename: str

    def __init__(self, missing: str, filename: str = "Run.out") -> None:
        self.missing = missing
        self.filename = filename
        self.message = f"{missing} not found in `{self.filename}`"
        super().__init__(self.message)


def read_and_fix_csv(dirout: str | pathlib.Path) -> io.StringIO:
    """Fixed the bug with the csv where if shifting is present in the `Run.csv` it has key that are
    `;` separated, e.g. Shifting(_txt_;_num_;_num_;_txt_). This causes a parsing bug with `pandas.read_csv`
    (any csv parser really). This function reads the data in fixes the bug in memory.

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.

    Returns
    -------
    io.StringIO
        The corrected IO object to be used instead of the file.

    Examples
    --------
    >>> stream = read_and_fix_csv(dirout=".")
    >>> df = pd.read_csv(stream, sep=";")
    """
    with open(f"{dirout}/Run.csv") as file:
        ORIG_RE = r"Shifting\((\w*);({0});({0});(\w*)\)".format(RE_PATTERNS.NUMBER)
        REPL_RE = r"Shifting(\1:\2:\3:\4)"
        txt = file.read()
        txt = re.sub(ORIG_RE, REPL_RE, txt)
        return io.StringIO(txt)


def get_dp(dirout: str | pathlib.Path) -> float:
    """Gets the inital particle distance of the simulation, aka `Dp`.

    Parameters
    ----------
    dirout : str, path object or file-like object
        The output directory of the simulation.

    Returns
    -------
    float
        The value of `Dp`.

    Raises
    ------
    NotFoundInOutput
        If `Dp` is not pressent in `Run.out`.
    """
    try:
        stream = read_and_fix_csv(dirout)
        df = pd.read_csv(stream, sep=";")
        return df["Dp"][0]

    except FileNotFoundError:
        with open(f"{dirout}/Run.out") as file:
            for line in file:
                pattern = r"Dp=({0})".format(RE_PATTERNS.NUMBER)
                number = re.match(pattern, line)
                if number:
                    return float(number.groups()[0])

        raise NotFoundInOutput("Variable `Dp`")


def get_usr_def_var(
    dirout: str | pathlib.Path, var: str, dtype: Callable[[str], R] = float
) -> R:
    """Finds and parses the value of any user defined variable from the simulation output.

    Parameters
    ----------
    dirout : str, path object or file-like object
        The output directory of the simulation.
    var : str
        The name of the user defined variable.
    dtype : Callable[[str], R], optional
        The return type of the function. The return type will be the same as the
        return type of the callable passed. The callable should accept a string as
        the input. E.g. if `int` is used the return type will be in `int`. By default `float`.

    Returns
    -------
    R
        The value of the variable, By default `float`.

    Raises
    ------
    NotFoundInOutput
        If the user defined variable is not pressent in `Run.out`.
    """

    with open(f"{dirout}/Run.out") as file:
        for line in file:
            if not line.startswith("XML-Vars (uservars + ctes)"):
                continue

            pattern = r"{0}=\[({1})\]".format(var, RE_PATTERNS.NUMBER)
            mass = re.search(pattern, line)
            if mass:
                return dtype(mass.groups()[0])

    raise NotFoundInOutput(f"User defined variable `{var}`")


def get_chrono_mass(dirout: str | pathlib.Path, bname: str) -> float:
    """Finds the mass of a floating chrono body.

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.
    bname : str
        The name of the body.

    Returns
    -------
    float
        The mass of the object

    Raises
    ------
    NotFoundInOutput
        If the either the chorno section or a chrono body with the
        specified name doesn't exist.
    """
    FLOATING_RE = re.compile(RE_PATTERNS.FLOATING)
    PATTERN = r"Mass.........: ({0})".format(RE_PATTERNS.NUMBER)

    with open(f"{dirout}/Run.out") as file:
        found_floating = False
        for line in file:
            if not found_floating:
                res = FLOATING_RE.search(line)
                if res and res.group("name") == bname:
                    found_floating = True
                continue

            mass = re.search(PATTERN, line)
            if mass:
                return float(mass.groups()[0])

    raise NotFoundInOutput(f'Chrono floating mass for "{bname}"')


def get_chrono_inertia(dirout: str | pathlib.Path, bname: str) -> np.ndarray:
    """Finds the inertia tensor of a floating chrono body (only diagonal elements).

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.
    bname : str
        The name of the body.

    Returns
    -------
    np.ndarray
        1D array with the digonal elements of inertia tensor of the body.

    Raises
    ------
    NotFoundInOutput
        If the either the chorno section or a chrono body with the
        specified name doesn't exist.
    """
    FLOATING_RE = re.compile(RE_PATTERNS.FLOATING)
    PATTERN = r"Inertia......: \(({0}),({0}),({0})\)".format(RE_PATTERNS.NUMBER)

    with open(f"{dirout}/Run.out") as file:
        found_floating = False
        for line in file:
            if not found_floating:
                res = FLOATING_RE.search(line)
                if res and res.group("name") == bname:
                    found_floating = True
                continue

            inertia = re.search(PATTERN, line)
            if inertia:
                return np.array([float(i) for i in inertia.groups()])

    raise NotFoundInOutput(f'Chrono floating mass for "{bname}"')


def get_chrono_property(
    dirout: str | pathlib.Path, bname: str, pname: str
) -> float | np.ndarray | str:
    """Finds and returns any property for a specified chrono floating body.

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.
    bname : str
        The name of the body.
    pname : str
        The name of the property. Should be a property that can be found at
        `Body_XXXX "<bname>" -  type: Floating` section of the `Run.out` file.

    Returns
    -------
    float | np.ndarray | str
        If the property can be interpreted as a number it will return a `float`.
        Else if it can be interpreted as an array it will return an `numpy.array`.
        In any other case it will return the `str` of the property

    Raises
    ------
    NotFoundInOutput
        If the either the chorno section or a chrono body with the specified name
        or property with the specified name doesn't exist.
    """
    FLOATING_RE = re.compile(RE_PATTERNS.FLOATING)
    PATTERN = r"{0}\.*: (.+)".format(pname)
    ELEM_PAT = r"\(({0}),({0}),({0})\)".format(RE_PATTERNS.NUMBER)

    with open(f"{dirout}/Run.out") as file:
        found_floating = False
        for line in file:
            if not found_floating:
                res = FLOATING_RE.search(line)
                if res and res.group("name") == bname:
                    found_floating = True
                continue

            res = re.search(PATTERN, line)
            if res:
                value = res.groups()[0]
                try:
                    return float(value)

                except ValueError as e:
                    if value[0] == '(':
                        elems = re.search(ELEM_PAT, value)
                        elems = elems.groups()
                        return np.array([float(i) for i in elems])

                    return value

    raise NotFoundInOutput(f'Property "{pname}" for chrono body "{bname}"')


if __name__ == "__main__":
    pass
