"""Implementation of DualSPHysics simulation jobs."""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from collections.abc import Iterable

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

    def add_args(self, *args: *tuple[str]):
        self.args.extend(args)

    def get_command_list(self) -> list[str]:
        return [self.path] + self.args


class Dualsphysics(Binary):
    def __init__(self, *args: *tuple[str], gpu: bool = True):
        name = "dualsphysics" + "" if gpu else "cpu"
        super.__init__(name, *args)


class Gencase(Binary):
    def __init__(self, *args: *tuple[str]):
        super.__init__("gencase", *args)



class Job:
    """Class that that create a dualsphysics job.

    Attributes
    ----------
    commands : list[list[str] | str]
        Commands that will be passed to `subproccess.run`. It is recommended
        that all commands are the same type, i.e. `list[str]` or `str`. If
        `str` is passed the command must be the name of the program without
        arguments eitherwise `shell=True` must be passed. See `subproccess`
        package documentation.
    shell : bool
        Same as `shell` argument in `subproccess.run` and `subproccess.Popen`.
    """
    commands: list[list[str] | str]
    shell: bool

    # Private attributes
    _env: dict[str, str]  # Environment variables

    def __init__(self, *commands: *tuple[str | list[str]], shell: bool = False):
        self.shell = shell
        self.commands = []
        self.add_commands(*commands)
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

    def __str__(self) -> str:
        return "\n".join(" ".join(c) for c in self.commands)

    def add_commands(self, *commands: *tuple[str | list[str]]):
        for command in commands:
            if isinstance(command, str):
                self.commands.append(command)
            elif isinstance(command, Iterable):
                self.commands.append(list(command))
            else:
                TypeError(f"'{command=}' is not of type `list[str]` or `str`")

    def add_binary_run(self, binary: Binary):
        if isinstance(binary, Binary):
            self.commands.append(binary.get_command_list())
        else:
            raise TypeError(f"{binary=} is not of type `Binary`")

    def run(self):
        for command in self.commands:
            print(f"[INFO] Running command '{command}'")
            subprocess.run(command, check=True, shell=self.shell, env=self._env)
