"""The main file where all the global scope variables, functions and classes
are defined.
"""

import os
import io
import re
import pathlib
from pathlib import Path
import platform
import subprocess
from typing import Callable, TypeVar, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
import lxml.etree as ET

from .exceptions import NotFoundInOutput

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
]

# BUG: There is a bug with the way DualSPHysics creates the `Run.csv`. It replaces all
#      `,` with `;` which causes the Shifting(_txt_,_num_,_num_,_txt_) in the configure
#      section to become Shifting(_txt_;_num_;_num_;_txt_). This causes a parsing bug
#      with `pandas.read_csv` (any csv parser really)

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

_R = TypeVar("_R")


class RE_PATTERNS:
    """Just constant class to store useful regex. Will move into a file if it grows too
    large.
    """

    # pattern to capture any number (eg 1.23, -1523, -12.3e-45)
    NUMBER = r"[\-\+]?\d+\.?\d*[Ee]?[\+\-]?\d*"
    # Pattern to captures the chrono floating section of the output. Returns the
    # "ID" and "name" of the chorno object floating
    FLOATING = r"Body_(?P<ID>\d+) \"(?P<name>\w*)\" -  type: Floating"


def read_and_fix_csv(dirout: Union[str, pathlib.Path]) -> io.StringIO:
    """Fixed the bug with the csv where if shifting is present in the `Run.csv` it has
    key that are `;` separated, e.g. Shifting(_txt_;_num_;_num_;_txt_). This causes a
    parsing bug with `pandas.read_csv` (any csv parser really). This function reads the
    data in fixes the bug in memory.

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


def get_dualsphysics_root() -> str:
    """Returns the path of the DualSPHysics root from the
    environment variables. `DUALSPH_HOME` or `DUALSPH_HOME`
    should be defined. If not returns empty `str`.

    Returns
    -------
    str
        The path of DualSPHysics. Empty if environment variables
        are undefined.
    """
    ret = ""
    if "DUALSPH_HOME" in os.environ:
        ret = os.environ["DUALSPH_HOME"]
    if "DUALSPH_HOME2" in os.environ:
        ret = os.environ["DUALSPH_HOME2"]
    return ret


def get_times_of_partfiles(dirout: Union[str, pathlib.Path]) -> list[tuple[int, float]]:
    """Reads the times of each part file in output directory from the `Run.out` file.

    Parameters
    ----------
    dirout : Union[str, pathlib.Path]
        The output directory of the simulations

    Returns
    -------
    list[tuple[int, float]]
        A list of the part number and the corresponding time.
    """
    ret = [(0, 0.0)]
    with open(f"{dirout}/Run.out", "r") as file:
        print(file)
        # Skip useless header and `Part_0000 section`.
        for line in file:
            print(line)
            if line.startswith("[Initialising simulation"):
                break

        for line in file:
            pattern = r"Part_(\d*)[ ]+({0})".format(RE_PATTERNS.NUMBER)
            part_and_time = re.match(pattern, line)
            if part_and_time:
                print(part_and_time)
                ret.append((int(part_and_time.group(1)), float(part_and_time.group(2))))
    ret[0] = (ret[1][0] - 1, 0.0)
    return ret


def get_dp(dirout: Union[str, pathlib.Path]) -> float:
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


def get_var(
    dirout: Union[str, pathlib.Path], var: str, dtype: Callable[[str], _R] = str
) -> _R:
    """Gets any variable that is defined in `Run.csv` or `Run.out` files.

    Parameters
    ----------
    dirout : str, path object or file-like object
        The output directory of the simulation.
    var: str
        The name of the varible in `Run.csv` or `Run.out`.
    dtype : Callable[[str], R], optional
        The return type of the function. The return type will be the same as the
        return type of the callable passed. The callable should accept a string as
        the input. E.g. if `int` is used the return type will be in `int`. By default
        `str`.

    Returns
    -------
    dtype
        The value of the variable, By default `str`.

    Raises
    ------
    NotFoundInOutput
        If `Dp` is not pressent in `Run.out`.
    """
    try:
        stream = read_and_fix_csv(dirout)
        df = pd.read_csv(stream, sep=";")
        return dtype(df[var][0])

    except (FileNotFoundError, KeyError):
        with open(f"{dirout}/Run.out") as file:
            for line in file:
                pattern = r"{0}=\[({1})\]".format(var, RE_PATTERNS.NUMBER)
                number = re.match(pattern, line)
                if number:
                    return dtype(number.groups()[0])


def get_usr_def_var(
    dirout: Union[str, pathlib.Path], var: str, dtype: Callable[[str], _R] = float
) -> _R:
    """Finds and parses the value of any user defined variable from the simulation
    output.

    Parameters
    ----------
    dirout : str, path object or file-like object
        The output directory of the simulation.
    var : str
        The name of the user defined variable.
    dtype : Callable[[str], RType], optional
        The return type of the function. The return type will be the same as the
        return type of the callable passed. The callable should accept a string as
        the input. E.g. if `int` is used the return type will be in `int`. By default
        `float`.

    Returns
    -------
    RType
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


def get_chrono_mass(dirout: Union[str, pathlib.Path], bname: str) -> float:
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


def get_chrono_inertia(dirout: Union[str, pathlib.Path], bname: str) -> np.ndarray:
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
    dirout: Union[str, pathlib.Path], bname: str, pname: str
) -> Union[float, np.ndarray, str]:
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

                except ValueError:
                    if value[0] == "(":
                        elems = re.search(ELEM_PAT, value)
                        elems = elems.groups()
                        return np.array([float(i) for i in elems])

                    return value

    raise NotFoundInOutput(f'Property "{pname}" for chrono body "{bname}"')


def run_measuretool(
    dirin: str,
    *,
    first_file: int = None,
    last_file: int = None,
    file_nums: List[int] = None,
    dirout: str = None,
    savecsv: str = "Measure",
    saveascii: str = None,
    savevtk: str = None,
    csvsep: bool = None,
    onlypos: Dict[str, Tuple[float, float, float]] = None,
    onlymk: int = None,
    onlyid: int = None,
    include_types: List[str] = None,
    exclude_types: List[str] = None,
    points_file=None,
    pt_list: np.ndarray = None,
    ptls_list: np.ndarray = None,
    ptels_list: np.ndarray = None,
    kclimit: float = None,
    kcdummy: float = None,
    kcusedummy: bool = None,
    kcmass: bool = None,
    distinter_2h: float = None,
    distinter: float = None,
    elevations: Union[bool, float] = None,
    enable_hvars: List[str] = None,
    disable_hvars: List[str] = None,
    enable_vars: List[str] = None,
    disable_vars: List[str] = None,
    binpath: str = None,
    options: str = None,
    print_options: bool = False,
) -> None:
    """A python wrapper of "measuretool" of DualSPHysics. If `None` is used in any
    of the option the default option of the tool will be used (check `-h` option).

    Parameters
    ----------
    dirin : path-like or str
        Indicates the directory with particle data.
    first_file : int, optional
        Indicates the first file to be computed. By default None
    last_file : int, optional
        Indicates the last file to be computed. By default None
    file_nums : List[int], optional
        Indicates the number of files to be processed. By default None
    dirout : path-like or str, optional
        The directory of the output of measuretool. By default None
    savecsv : str, optional
        Generates one CSV file with the time history of the obtained values.
        By default "Measure"
    saveascii : str, optional
        Generates one ASCII file without headers with the time history of the
        obtained values. By default None
    savevtk : str, optional
        Generates VTK(polydata) file with the given interpolation points.
        By default None
    csvsep : bool, optional
        Separator character in CSV files (0=semicolon, 1=coma)
        (value by default is read from DsphConfig.xml or 0). By default None
    onlypos : Dict[str, Tuple[float, float, float]], optional
        Indicates limits of particles. By default None
    onlymk : int, optional
        Indicates the mk of selected particles. By default None
    onlyid : int, optional
        Indicates the id of selected particles. By default None
    include_types : List[str], optional
        Indicates the type of selected particles to be included. Accepted values:
        "all", "bound", "fixed", "moving", "floating", "fluid". By default "all"
    exclude_types : List[str], optional
        Indicates the type of selected particles to be excluded. Accepted values:
        "all", "bound", "fixed", "moving", "floating", "fluid". By default None
    points_file : _type_, optional
        Defines the points where interpolated data will be computed (each value
        separated by space or a new line). By default None
    pt_list : np.ndarray, optional
        A list of points to where interpolated data will be computed. The shape
        of the array should be (n, 3) or should be a valid array for the numpy
        function `numpy.reshape((-1, 3))` to be used.  By default None
    ptls_list : np.ndarray, optional
        A list of "POINTSLIST" in the format:
        [[<x0>,<dx>:<nx>],
         [<y0>:<dy>:<ny>],
         [<z0>:<dz>:<nz>]].
        The shape of the array should be (n, 3, 3) or should be a valid array for
        the numpy function `numpy.reshape((-1, 3, 3))` to be used.
        By default None
    ptels_list : np.ndarray, optional
        A list of "POINTSENDLIST" in the format:
        [[<x0>,<dx>:<xf>],
         [<y0>:<dy>:<yf>]  ,
         [<z0>:<dz>:<zf>]].
        The shape of the array should be (n, 3, 3) or should be a valid array for
        the numpy function `numpy.reshape((-1, 3, 3))` to be used.
        By default None
    kclimit : float
        Defines the minimum value of sum_wab_vol to apply the Kernel Correction.
        Use value >= 2 to disable this correction. By default None
    kcdummy : float
        Defines the dummy value for the interpolated quantity if Kernel Correction
        is not applied. By default None
    kcusedummy : bool
        Defines whether or not to use the dummy value. By default None.
    kcmass: bool
        Enables/disables Kernel Correction for Mass variable. By default None.
    distinter_2h : float
        Coefficient of 2h that defines the maximum distance for the interaction
        among particles depending on 2h. By default None.
    distinter : float
        Defines the maximum distance for the interaction among particles in an
        absolute way. By default None.
    elevations : Union[bool, float], optional
        Fluid elevation is calculated starting from mass values for each point
        x,y. The reference mass to obtain the elevation is calculated according
        to the mass values of the selected particles. By default 0.5 in 3D (half
        the mass) and 0.4 in 2D.
    enable_hvars : List[str], optional
        Enable height values to be computed. By default None
    disable_hvars : List[str], optional
        Disable height values to be computed. By default "All"
    enable_vars : List[str], optional
        Enable the variables or magnitudes that are going to be computed as an
        interpolation of the selected particles around a given position. By
        default vel,rhop or empty when elevation/tke calculation is enabled.
    disable_vars : List[str], optional
        Enable the variables or magnitudes that are going to be computed as an
        interpolation of the selected particles around a given position.
    binpath : str, optional
        The path of the binary folder of DualSPHysics. If not defined the an
        environment variable "DUALSPH_HOME" must be defined. By default None.
    options : str, optional
        A string of the command line option to be pass. If this argument is pass
        all other arguments are ignored. By default None.
    print_options : bool, optional
        if `True` prints the options pass before the execution. By default, `False`.

    Raises
    ------
    subprocess.CalledProcessError
        If the exitcode is not 0
    Exception
        If a binary path is not passed and an environment variable "DUALSPH_HOME"
        doesn't exist.
    """
    if binpath is None and "DUALSPH_HOME" not in os.environ:
        err_msg = (
            '"DUALSPH_HOME" environment variable not specified '
            + "and `binpath` not specified. Please specify one of them."
        )
        raise Exception(err_msg)

    if binpath is None:
        binpath = f"{os.environ['DUALSPH_HOME']}/bin"

    plat = platform.system()
    binpath = Path(binpath)
    dirin = Path(dirin)
    if plat == "Linux":
        dirbin = binpath / "linux"
        binary = dirbin / "MeasureTool_linux64"
    elif plat == "Windows":
        dirbin = binpath / "windows"
        binary = dirbin / "MeasureTool_win64"

    # If `options` is specified run use those.
    if options is not None:
        if print_options:
            print(f'Running MeasureTool with options: "{options}"')
        # subprocess.run([binary, *options.split(" ")])
        return _run_and_capture_measuretool([binary, *options.split(" ")])
    else:
        # just because pylance is active wierd without the else
        pass

    if dirout is None:
        dirout = dirin / "measuretool"
    else:
        dirout = Path(dirout)

    opts = ["-dirin", str(dirin / "data")]
    types = []
    vars = []
    hvars = []
    pointsdef = None

    # Input options
    if first_file is not None:
        opts.append(f"-first:{first_file}")

    if last_file is not None:
        opts.append(f"-last:{last_file}")

    if file_nums is not None:
        files_to_str = ",".join(map(str, file_nums))
        opts.append(f"-files:{files_to_str}")

    # Save options
    if savecsv is not None:
        opts.extend(("-savecsv", str(dirout / savecsv)))

    if savevtk is not None:
        opts.extend(("-savevtk", str(dirout / savevtk)))

    if saveascii is not None:
        opts.extend(("-saveascii", str(dirout / saveascii)))

    if csvsep is not None:
        opts.append(f"-csvsep:{int(savecsv)}")

    # Point definitions options
    if pt_list is not None:
        pointsdef = "-pointsdef:"
        points_array = np.array(pt_list)
        if points_array.ndim != 2:
            points_array = points_array.reshape((-1, 3))
        tmp_iter = (":".join(map(str, point)) for point in points_array)
        pointsdef += "pt=" + ",pt=".join(tmp_iter)

    if ptls_list is not None:
        if pointsdef is None:
            pointsdef = "-pointsdef:"
        else:
            pointsdef += ","
        grid_array = np.array(ptls_list)
        if grid_array.ndim != 3:
            grid_array = grid_array.reshape((-1, 3, 3))

        array_to_str_list = []
        for grid in grid_array:
            array_to_str = "ptls["
            for a, prefix in zip(grid, ("x=", ",y=", ",z=")):
                array_to_str += prefix + ":".join(map(str, a))
            array_to_str += "]"
            array_to_str_list.append(array_to_str)
        pointsdef += ",".join(array_to_str_list)

    if ptels_list is not None:
        if pointsdef is None:
            pointsdef = "-pointsdef:"
        else:
            pointsdef += ","
        grid_array = np.array(ptels_list)
        if grid_array.ndim != 3:
            grid_array = grid_array.reshape((-1, 3, 3))

        array_to_str_list = []
        for grid in grid_array:
            array_to_str = "ptels["
            for a, prefix in zip(grid, ("x=", ",y=", ",z=")):
                array_to_str += prefix + ":".join(map(str, a))
            array_to_str += "]"
            array_to_str_list.append(array_to_str)
        pointsdef += ",".join(array_to_str_list)

    if pointsdef is not None:
        opts.append(pointsdef)

    if points_file is not None:
        opts.extend(("-points", points_file))

    # Interpolation options
    if kclimit is not None:
        opts.append(f"-kclimit:{kclimit}")

    if kcdummy is not None:
        opts.append(f"-kcdummy:{kcdummy}")

    if kcusedummy is not None:
        opts.append(f"-kcusedummy:{int(kcusedummy)}")

    if kcmass is not None:
        opts.append(f"-kcmass:{int(kcmass)}")

    if distinter_2h is not None:
        opts.append(f"-distinter_2h:{distinter_2h}")

    if distinter is not None:
        opts.append(f"-distinter:{distinter}")

    # Filter options
    if onlymk is not None:
        opts.append(f"-onlymk:{onlymk}")

    if onlyid is not None:
        opts.append(f"-onlyid:{onlyid}")

    if onlypos is not None:
        pts_to_str = (
            "-onlypos:"
            + ":".join(map(str, onlypos["min"]))
            + ":"
            + ":".join(map(str, onlypos["max"]))
        )
        opts.append(pts_to_str)

    if exclude_types is not None:
        types.extend((f"-{t}" for t in exclude_types))

    if include_types is not None:
        types.extend((f"+{t}" for t in include_types))

    if types:
        types_to_str = "-onlytype:" + ",".join(types)
        opts.append(types_to_str)

    # Calculations variables options
    if elevations:
        tmp = "-elevation"
        if isinstance(elevations, (float, int)):
            tmp += f":{elevations}"
        opts.append(tmp)
        opts.append("-elevationoutput:all")

    if disable_vars is not None:
        vars.extend((f"-{t}" for t in disable_vars))

    if enable_vars is not None:
        vars.extend((f"+{t}" for t in enable_vars))

    if vars:
        vars_to_str = "-vars:" + ",".join(vars)
        opts.append(vars_to_str)

    if disable_hvars is not None:
        hvars.extend((f"-{t}" for t in disable_hvars))

    if enable_hvars is not None:
        hvars.extend((f"+{t}" for t in enable_hvars))

    if hvars:
        hvars_to_str = "-hvars:" + ",".join(hvars)
        opts.append(hvars_to_str)

    if print_options:
        print(f"Running MeasureTool with options: \"{' '.join(opts)}\"")
    _run_and_capture_measuretool([binary, *opts])


def _run_and_capture_measuretool(cmd) -> None:
    """Runs MeasureTool and captures stdout.

    Parameters
    ----------
    cmd : List[str]
        The command list to be passed to `subprocess.Popen`

    Raises
    ------
    subprocess.CalledProcessError
        If the exitcode is not 0
    """
    line_re = re.compile("LoadData>|Save.*>")
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        universal_newlines=True,
        bufsize=1,
    ) as process:
        while process.poll() is None:
            line = process.stdout.readline()

            # print(line)
            if line_re.match(line):
                print(line, end="", flush=True)

        for _ in process.stdout.readlines():
            if line_re.match(line):
                print(line, end="", flush=True)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)


def xml_get_or_create_subelement(parent_elem: ET.Element, child: str):
    """Get or created a subelement of an "lxml" element.

    Parameters
    ----------
    parent_elem : lxml.ET.Element
        The parent element
    child : str
        The name of the child element

    Returns
    -------
    lxml.ET.SubElement
        The child element if it exist or a new child element.
    """
    child_elem = parent_elem.find(child)
    if child_elem is None:
        child_elem = ET.SubElement(parent_elem, child)

    return child_elem
