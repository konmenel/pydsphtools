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


class InvalidTimeInterval(Exception):
    """Raised when a variable is not found in the output file.

    Attributes
    ----------
    missing : str
        What was not found.
    filename : str, optional
        The name of the output file (either `Run.out` or `Run.csv`). By default, `Run.out`.
    """

    tmin: float
    tmax: float

    def __init__(self, tmin: float, tmax: float) -> None:
        self.tmin = tmin
        self.tmax = tmax
        self.message = f"Invalid time interval: ({tmin}-{tmax})."
        super().__init__(self.message)
