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
import os
import io
import re
import errno
import pathlib
import platform
import subprocess
from typing import Callable, TypeVar, Union, Tuple, List, Dict, Iterable

import numpy as np
import pandas as pd
import lxml.etree as ET

from pydsphtools.exceptions import NotFoundInOutput


# BUG: There is a bug with the way DualSPHysics creates the `Run.csv`. It replaces all `,` with `;`
#   which causes the Shifting(_txt_,_num_,_num_,_txt_) in the configure section to become
#   Shifting(_txt_;_num_;_num_;_txt_). This causes a parsing bug with `pandas.read_csv` (any csv parser really)

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

_R = TypeVar("_R")


class RE_PATTERNS:
    """Just constant class to store useful regex. Will move into a file if it grows too large."""

    # pattern to capture any number (eg 1.23, -1523, -12.3e-45)
    NUMBER = r"[\-\+]?\d+\.?\d*[Ee]?[\+\-]?\d*"
    # Pattern to captures the chrono floating section of the output. Returns the
    # "ID" and "name" of the chorno object floating
    FLOATING = r"Body_(?P<ID>\d+) \"(?P<name>\w*)\" -  type: Floating"


def read_and_fix_csv(dirout: Union[str, pathlib.Path]) -> io.StringIO:
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


def get_usr_def_var(
    dirout: Union[str, pathlib.Path], var: str, dtype: Callable[[str], _R] = float
) -> _R:
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

                except ValueError as e:
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
    elevations: Union[bool, float] = None,
    enable_hvars: List[str] = None,
    disable_hvars: List[str] = None,
    enable_vars: List[str] = None,
    disable_vars: List[str] = None,
    binpath: str = None,
    options: str = None,
) -> None:
    """A python wrapper of "measuretool" of DualSPHysics.

    Parameters
    ----------
    dirin : str
        Indicates the directory with particle data.
    first_file : int, optional
        Indicates the first file to be computed. By default None
    last_file : int, optional
        Indicates the last file to be computed. By default None
    file_nums : List[int], optional
        Indicates the number of files to be processed. By default None
    dirout : str, optional
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
        By default None By default None
    ptels_list : np.ndarray, optional
        A list of "POINTSENDLIST" in the format:
        [[<x0>,<dx>:<nx>],
         [<y0>:<dy>:<ny>],
         [<z0>:<dz>:<nz>]].
        The shape of the array should be (n, 3, 3) or should be a valid array for
        the numpy function `numpy.reshape((-1, 3, 3))` to be used.
        By default None By default None
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

    Raises
    ------
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
    if plat == "Linux":
        dirbin = f"{binpath}/linux"
        binary = f"{dirbin}/MeasureTool_linux64"
    elif plat == "Windows":
        dirbin = f"{binpath}/windows"
        binary = f"{dirbin}/MeasureTool_win64"

    # If `options are specified run use those.
    if options is not None:
        subprocess.run([binary, *options.split(" ")])
        return

    if dirout is None:
        dirout = f"{dirin}/measuretool"

    opts = ["-dirin", f"{dirin}/data"]
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
        opts.extend(("-savecsv", f"{dirout}/{savecsv}"))

    if savevtk is not None:
        opts.extend(("-savevtk", f"{dirout}/{savevtk}"))

    if saveascii is not None:
        opts.extend(("-saveascii", f"{dirout}/{saveascii}"))

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

    subprocess.run([binary, *opts])


def mlpistons2D_from_dsph(
    xmlfile: str,
    dirin: str,
    xloc: float,
    yrange: Tuple[float, float],
    zrange: Tuple[float, float],
    ylayers: int,
    zlayers: int,
    mkbound: int,
    *,
    smoothz: int = 0,
    smoothy: int = 0,
    file_prefix: str = "MLPiston2D_SPH_velx",
    dirout: str = "MLPiston2D",
    binpath: str = None,
) -> None:
    """Create the nessesary csv file to run a DualSPHysics Multi-Layer 2D Piston
    simulation using data from a previous DualSPHysics simulation. The function
    uses "measuretool" to find the surface elevation at a specific x-location
    and creates a grid at every timestep with a given number of vertical layers.

    Parameters
    ----------
    xmlfile : str
        The xml file that defines the simulation. The file will be modified to
        create the "mlayerpistons" element. If no extention is provided the
        code assumes a ".xml" at the end.
    dirin : str
        The output directory of the old simulation.
    xloc : float
        The x-location where the velocities will be interpolated.
    yrange : Tuple[float, float]
        The domain limits of the fluid in the y-direction.
    zrange : Tuple[float, float]
        The domain limits of the fluid in the z-direction.
    ylayers : int
        The number of layers in the lateral direction.
    zlayers : int
        The number of layers in the vertical direction.
    mkbound : int
        The mk value of the piston particles.
    smoothz : int, optional
        Smooth motion level in Y (xml attribute), by default 0.
    smoothy : int, optional
        Smooth motion level in Y (xml attribute), by default 0.
    dirout : str
    The name of the folder where the csv files will be placed.
    By default "MLPiston2D".
    file_prefix : str, optional
        The prefix of the csv files, by default "MLPiston2D_SPH_velx".
    binpath : str, optional
        The path of the binary folder of DualSPHysics. If not defined the an
        environment variable "DUALSPH_HOME" must be defined. By default None.

    Raises
    ------
    FileNotFoundError
        If the xml file could not be found. The rest of the code will still
        run but modification will not be made no valid xml is provided.
    """
    xmlfile = os.path.abspath(xmlfile)
    xmldir, _ = os.path.split(xmlfile)

    dp = get_dp(dirin)
    ylen = yrange[1] - yrange[0]
    dy = ylen / ylayers
    y0 = yrange[0] + 0.5 * dy

    # Save configuration
    config = pd.DataFrame(
        {"NoLayers_y": [ylayers], "NoLayers_z": [zlayers], "xLocation": [xloc]}
    )

    # Find the free surface for each column (y-direction)
    ptels_list = [[xloc, 0, xloc], [y0, dy, yrange[1]], [0, 0.0001, zrange[1]]]
    exclude_list = ["all"]
    include_list = ["fluid"]
    disable_hvars = ["all"]
    enable_hvars = ["eta"]

    freesurface_fname = "MLPiston_freesurf"
    freesurface_fpath = f"{dirin}/measuretool/{freesurface_fname}_Elevation.csv"

    config_fpath = f"{dirin}/measuretool/MLPiston_config.csv"
    old_config = None
    if os.path.exists(config_fpath):
        old_config = np.loadtxt(config_fpath, delimiter=";", skiprows=1)
    if (
        not os.path.exists(freesurface_fpath)
        or old_config is None
        or (old_config != config.iloc[0].values).any()
    ):
        config.to_csv(config_fpath, index=False, sep=";")
        run_measuretool(
            dirin,
            savecsv=freesurface_fname,
            ptels_list=ptels_list,
            include_types=include_list,
            exclude_types=exclude_list,
            elevations=True,
            enable_hvars=enable_hvars,
            disable_hvars=disable_hvars,
            binpath=binpath,
        )

    # Read elevations and create the point list for with the layers
    df_fs = pd.read_csv(freesurface_fpath, header=3, sep=";")
    # remove units, eg `Vel [m/s^2]` -> `Vel`
    df_fs.columns = df_fs.columns.map(
        lambda x: re.sub(r"\ \[[A-Za-z\^0-9/]*\]$", "", x)
    )
    # time_series = df_fs.Time
    points_per_time = []
    ys = np.arange(y0, ylen, dy)
    grid_y = np.tile(ys, (zlayers, 1))
    for _, row in df_fs.iterrows():
        row = row.drop(["Time", "Part"])
        grid_z = np.zeros((zlayers, ylayers))
        for j, elevation in enumerate(row):
            highest_point = elevation - 3*dp
            # dz = highest_point / zlayers
            grid_z[:, j] = np.linspace(highest_point, dp, zlayers)
        points_per_time.append((grid_y, grid_z))

    # Find velocity data from the points and create the dataframe
    outfiles_dir = f"{xmldir}/{dirout}"
    if not os.path.exists(outfiles_dir):
        os.mkdir(outfiles_dir)

    rawdata_fname = "MLPiston_data_raw"
    rawdata_fpath = f"{dirin}/measuretool/{rawdata_fname}_Vel.csv"
    disable_vars = ["all"]
    enable_vars = ["vel"]

    columns = pd.Series(
        [
            "time",
            *(f"pz_{i}" for i in range(zlayers)),
            *(f"vel_{i}" for i in range(zlayers)),
        ]
    )
    _get_key_fmt = lambda x, y: f"px:;{x};py:;{y}"
    dfs = pd.DataFrame(columns=columns, index=range(len(df_fs.Part)))
    outfiles = {_get_key_fmt(xloc, y): dfs.copy() for y in ys}

    for i, (grid_y, grid_z) in enumerate(points_per_time):
        if os.path.exists(f"{outfiles_dir}/{file_prefix}_y00.csv"):
            break

        run_measuretool(
            dirin,
            savecsv=rawdata_fname,
            file_nums=(i,),
            pt_list=[
                (xloc, grid_y[i, j], grid_z[i, j])
                for i in range(zlayers)
                for j in range(ylayers)
            ],
            include_types=include_list,
            exclude_types=exclude_list,
            enable_vars=enable_vars,
            disable_vars=disable_vars,
            binpath=binpath,
        )
        df_vel = pd.read_csv(rawdata_fpath, header=1, sep=";")
        time = df_vel.loc[0, "Time [s]"]
        df_vel = df_vel.loc[:, df_vel.columns.str.contains("x")]

        for j, y in enumerate(ys):
            key = _get_key_fmt(xloc, y)
            zs = grid_z[:, j]
            free_surface = zs.max() + 3*dp
            outfiles[key].iloc[i, 0] = time
            outfiles[key].iloc[i, 1 : zlayers + 1] = zs - free_surface
            outfiles[key].iloc[i, zlayers + 1 :] = df_vel.iloc[0, j::ylayers]

    # Save the input files
    for i in range(len(ys)):
        fname = f"{outfiles_dir}/{file_prefix}_y{i:02d}.csv"
        key = _get_key_fmt(xloc, ys[i])
        if os.path.exists(fname):
            break

        with open(fname, "a") as f:
            f.write(f"{key}\n")
            outfiles[key].to_csv(f, sep=";", index=False)
    print(f"Csv input files created in directory \"{outfiles_dir}\".")

    velfiles = (f"{outfiles_dir}/{file_prefix}_y{i:02d}.csv" for i in range(ylayers))
    write_mlpiston_xml(xmlfile, mkbound, velfiles, ys, smoothz=smoothz, smoothy=smoothy)

    # Clean-up
    if os.path.exists(rawdata_fpath):
        os.remove(rawdata_fpath)
    if os.path.exists(f"{dirin}/measuretool/{rawdata_fname}_PointsDef.vtk"):
        os.remove(f"{dirin}/measuretool/{rawdata_fname}_PointsDef.vtk")


def write_mlpiston_xml(
    xmlfile: str,
    mkbound: int,
    velfiles: Iterable[str],
    yvals: Iterable[float],
    *,
    smoothz: int = 0,
    smoothy: int = 0,
) -> None:
    # Check if the xml exist
    if not os.path.exists(xmlfile):
        if xmlfile.endswith(".xml"):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlfile)

        xmlfile = f"{xmlfile}.xml"
        if not os.path.exists(xmlfile):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlfile)

    xmldir, _ = os.path.split(xmlfile)
    tree = ET.parse(xmlfile)

    # Add to `motion` section
    elem_casedef = xml_get_or_create_subelement(tree, "casedef")
    elem_motion = xml_get_or_create_subelement(elem_casedef, "motion")
    elem_objreal = ET.SubElement(
        elem_motion,
        "objreal",
        attrib={"ref": str(mkbound)},
    )
    ET.SubElement(
        elem_objreal,
        "begin",
        attrib={"mov": "100", "start": "0"},
    )
    ET.SubElement(
        elem_objreal,
        "mvnull",
        attrib={"id": "100"},
    )
    print("[xml file] `motion` section updated.")

    # Add to `special` section
    elem_exec = xml_get_or_create_subelement(tree, "execution")
    elem_special = xml_get_or_create_subelement(elem_exec, "special")

    mlayer_elem = xml_get_or_create_subelement(elem_special, "mlayerpistons")
    if mlayer_elem.find("piston2d") is not None:
        print(
            "*WARNING* [xml file]`piston2d` already exist in xml. Exitting without modifying the xml."
        )
        return
    piston2d = ET.SubElement(mlayer_elem, "piston2d")
    ET.SubElement(
        piston2d,
        "mkbound",
        attrib={"value": str(mkbound), "comment": "Mk-Bound of selected particles"},
    )
    ET.SubElement(
        piston2d,
        "smoothz",
        attrib={"value": str(smoothz), "comment": "Smooth motion level in Z (def=0)"},
    )
    ET.SubElement(
        piston2d,
        "smoothy",
        attrib={"value": str(smoothy), "comment": "Smooth motion level in Y (def=0)"},
    )
    for fname, y in zip(velfiles, yvals):
        fname = os.path.relpath(fname, xmldir)
        veldata = ET.SubElement(piston2d, "veldata")
        ET.SubElement(
            veldata,
            "filevelx",
            attrib={"value": fname, "comment": "File name with X velocity"},
        )
        ET.SubElement(
            veldata,
            "posy",
            attrib={"value": str(y), "comment": "Position Y of data"},
        )
    print("[xml file] `special` section updated. `piston2d` added.")

    ET.indent(tree, " " * 4)
    tree.write(xmlfile)


def xml_get_or_create_subelement(parent_elem, child: str):
    child_elem = parent_elem.find(child)
    if child_elem is None:
        child_elem = ET.SubElement(parent_elem, child)

    return child_elem
