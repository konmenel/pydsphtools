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
    """Raised if an enviroment variable is undefined.

    Attributes
    ----------
        var_name : str
            The name of the environment variable.
    """

    var_name: str

    def __init__(self, var_name: str, *args) -> None:
        self.var_name = var_name
        super().__init__(*args)


class UnsupportedPlatform(Exception):
    """Raised if the platform is not supperted by DualSPHysics/

    Attributes
    ----------
    platform : str
        Name of the platform
    """

    platform: str

    def __init__(self, platform: str, *args: object) -> None:
        self.platform = platform
        super().__init__(f'Unsupported platform "{self.platform}".')


class DSPHBinaryNotFound(Exception):
    """Raised if a binary is not found in the path.

    Attributes
    ----------
    binary_name : str
        Name of the binary
    path : str
        The path that was searched.
    """

    binary_name: str
    path: str

    def __init__(self, binary_name: str, path: str, *args: object) -> None:
        self.binary_name = binary_name
        self.path = path
        super().__init__(
            f'DualSPHysics binary "{self.binary_name}" not found in "{path}".'
        )
