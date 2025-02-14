"""Implementation of DualSPHysics simulation jobs."""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Self

from ._main import get_binary_path, get_dualsphysics_root

# TODO: Implement all binaries


class Binary:
    name: str
    path: Path
    args: list[str]

    def __init__(self, name: str, *args: *tuple[str]):
        self.name = name
        self.args = list(args)
        self.path = get_binary_path(name)

    def __str__(self) -> str:
        return f"{self.path} {' '.join(self.args)}"

    def add_args(self, *args: *tuple[str]) -> Self:
        self.args.extend(args)
        return self

    def get_command_list(self) -> list[str]:
        return [self.path] + self.args

    def run(self, *args, **kwargs) -> Self:
        subprocess.run(self.get_command_list(), *args, **kwargs)


class Dualsphysics(Binary):
    def __init__(self, *args: *tuple[str], gpu: bool = True, version: str = "5.2"):
        name = f"dualsphysics{version}" + "" if gpu else "cpu"
        super().__init__(name, *args)


class Gencase(Binary):
    def __init__(self, *args: *tuple[str]):
        super().__init__("gencase", *args)


class Job:
    """Class that that create a dualsphysics job.

    Attributes
    ----------
    binaries: list[Binary]
        The binary objects that will be run. Each binary will be run sequentially
        in the order that they have been passed.
    verbose : bool
        Prints information the job when `run()` is called. Default, `True`.

    Examples
    --------
    ```python
    from pydsphtools import Job

    job = Job()
    job.add_binaries(Dualsphysics("-h"), verbose=True)
    job.run()
    ```
    """

    binaries: list[Binary]
    verbose: bool

    # Private attributes
    _env: dict[str, str]  # Environment variables

    def __init__(self, *binaries: *tuple[Binary], verbose: bool = True):
        self.verbose = verbose
        self.binaries = list(binaries)
        self._env = dict(**os.environ)
        dirbin = Path(get_dualsphysics_root()) / "bin" / "linux"
        self._env.update(
            {
                "LD_LIBRARY_PATH": f"{dirbin}"
                + (
                    f":{self._env['LD_LIBRARY_PATH']}"
                    if self._env["LD_LIBRARY_PATH"]
                    else ""
                )
            }
        )

    @property
    def env(self) -> dict[str, str]:
        return self._env

    def __str__(self) -> str:
        return "\n".join(str(bin) for bin in self.binaries)

    def print_env(self):
        for k, v in self.env:
            print(f"{k}: {v}")

    def add_binaries(self, *binaries: *tuple[Binary]) -> Self:
        self.binaries.extend(binaries)
        return self

    def run(self):
        if self.verbose:
            print("Job commands:")
            print(self)
            print()

        for bin in self.binaries:
            if self.verbose:
                print(f"[INFO] Running command {bin}")
            bin.run(check=True, env=self._env)
