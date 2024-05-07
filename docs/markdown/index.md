Module pydsphtools
==================
Copyright (C) 2023 Constantinos Menelaou <https://github.com/konmenel>

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

Author: Constantinos Menelaou<br>
Github: https://github.com/konmenel <br>
Year: 2023<br>

Sub-modules
-----------
* pydsphtools.exceptions
* pydsphtools.mlpistons
* pydsphtools.relaxzones
* pydsphtools.stats
* pydsphtools.waves

Functions
---------

    
`get_binary_path(name: str, binpath: str = None) -> str`
:   Gets the full path of a binary of the DualSPHysics.
    
    Parameters
    ----------
    name : str
        The name of the binary. Case insensitive.
    binpath : str, optional
        The path of the binary folder of DualSPHysics. If not defined the
        environment variable "DUALSPH_HOME" must be defined. For example,
        "/home/myuser/DualSPHysics_5.2/bin". By default None.
    
    Returns
    -------
    str
        The absolute path of the binary.
    
    Raises
    ------
    MissingEnvironmentVariable
        If `binpath` is None, and environment variables `DUALSPH_HOME`
        and `DUALSPH_HOME2` are undefined.
    
    UnsupportedPlatform
        If the platform is neither windows or linux.
    
    DSPHBinaryNotFound
        If the binary does not exists in the binary directory.

    
`get_chrono_inertia(dirout: Union[str, pathlib.Path], bname: str) -> numpy.ndarray`
:   Finds the inertia tensor of a floating chrono body (only diagonal elements).
    
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

    
`get_chrono_mass(dirout: Union[str, pathlib.Path], bname: str) -> float`
:   Finds the mass of a floating chrono body.
    
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

    
`get_chrono_property(dirout: Union[str, pathlib.Path], bname: str, pname: str) -> Union[float, numpy.ndarray, str]`
:   Finds and returns any property for a specified chrono floating body.
    
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

    
`get_dp(dirout: Union[str, pathlib.Path]) -> float`
:   Gets the inital particle distance of the simulation, aka `Dp`.
    
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

    
`get_dualsphysics_root() -> str`
:   Returns the path of the DualSPHysics root from the
    environment variables. `DUALSPH_HOME` or `DUALSPH_HOME2`
    should be defined. If not returns empty `str`.
    
    Returns
    -------
    str
        The path of DualSPHysics. Empty if environment variables
        are undefined.

    
`get_number_of_partfiles(diroutdata: Union[str, pathlib.Path]) -> int`
:   Returns the total number of `Part_xxxx.bi4` files in the `diroutdata` directory.
    
    Parameters
    ----------
    diroutdata : Union[str, pathlib.Path]
        The output directory of the simulations containing the Part files
    
    Returns
    -------
    int
        The total number of `Part_xxxx.bi4` files in the `data` directory.

    
`get_partfiles(diroutdata: Union[str, pathlib.Path]) -> list[str]`
:   Returns a list of all `Part_xxxx.bi4` files in the `data` directory.
    
    Parameters
    ----------
    diroutdata : Union[str, pathlib.Path]
        The output directory of the simulations
    
    Returns
    -------
    int
        The total number of `Part_xxxx.bi4` files in the `data` directory.

    
`get_times_of_partfiles(dirout: Union[str, pathlib.Path]) -> list[tuple[int, float]]`
:   Reads the times of each part file in output directory from the `Run.out` file.
    
    Parameters
    ----------
    dirout : Union[str, pathlib.Path]
        The output directory of the simulations
    
    Returns
    -------
    list[tuple[int, float]]
        A list of the part number and the corresponding time.

    
`get_usr_def_var(dirout: Union[str, pathlib.Path], var: str, dtype: Callable[[str], ~_R] = builtins.float) -> ~_R`
:   Finds and parses the value of any user defined variable from the simulation
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

    
`get_var(dirout: Union[str, pathlib.Path], var: str, dtype: Callable[[str], ~_R] = builtins.str) -> ~_R`
:   Gets any variable that is defined in `Run.csv` or `Run.out` files.
    
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

    
`read_and_fix_csv(dirout: Union[str, pathlib.Path]) -> _io.StringIO`
:   Fixed the bug with the csv where if shifting is present in the `Run.csv` it has
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

    
`run_measuretool(dirin: str, *, first_file: int = None, last_file: int = None, file_nums: List[int] = None, dirout: str = None, savecsv: str = 'Measure', saveascii: str = None, savevtk: str = None, csvsep: bool = None, onlypos: Dict[str, Tuple[float, float, float]] = None, onlymk: int = None, onlyid: int = None, include_types: List[str] = None, exclude_types: List[str] = None, points_file=None, pt_list: numpy.ndarray = None, ptls_list: numpy.ndarray = None, ptels_list: numpy.ndarray = None, kclimit: float = None, kcdummy: float = None, kcusedummy: bool = None, kcmass: bool = None, distinter_2h: float = None, distinter: float = None, elevations: Union[bool, float] = None, enable_hvars: List[str] = None, disable_hvars: List[str] = None, enable_vars: List[str] = None, disable_vars: List[str] = None, binpath: str = None, options: str = None, print_options: bool = False) -> NoneType`
:   A python wrapper of "measuretool" of DualSPHysics. If `None` is used in any
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
        The path of the binary folder of DualSPHysics. If not defined the
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

    
`xml_get_or_create_subelement(parent_elem: <cyfunction Element at 0x7914d0d3a0c0>, child: str)`
:   Get or created a subelement of an "lxml" element.
    
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

Classes
-------

`RE_PATTERNS()`
:   Just constant class to store useful regex. Will move into a file if it grows too
    large.

    ### Class variables

    `FLOATING`
    :

    `NUMBER`
    :