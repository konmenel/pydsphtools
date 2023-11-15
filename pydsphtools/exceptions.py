"""The module with the definition of the exceptions used in the packages."""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.


class NotFoundInOutput(Exception):
    """Raised when a variable is not found in the output file.

    Attributes
    ----------
    missing : str
        What was not found.
    filename : str, optional
        The name of the output file (either `Run.out` or `Run.csv`). By default,
         `Run.out`.
    """

    missing: str
    filename: str

    def __init__(self, missing: str, filename: str = "Run.out") -> None:
        self.missing = missing
        self.filename = filename
        self.message = f"{missing} not found in `{self.filename}`"
        super().__init__(self.message)


class InvalidTimeInterval(Exception):
    """Raised when a variable is not found in the output file.

    Attributes
    ----------
    tmin : float
        The lower bound of the time interval.
    tmax : float
        The higher bound of the time interval.
    """

    tmin: float
    tmax: float

    def __init__(self, tmin: float, tmax: float) -> None:
        self.tmin = tmin
        self.tmax = tmax
        self.message = f"Invalid time interval: ({tmin}-{tmax})."
        super().__init__(self.message)


class MissingEnvironmentVariable(Exception):
    var_name: str

    def __init__(self, var_name: str, *args) -> None:
        self.var_name = var_name
        super().__init__(*args)
