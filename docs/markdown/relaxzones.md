Module pydsphtools.relaxzones
=============================
Module that enables coupling of different DualSPHysics simulations using
the Relaxation Zone technic of DualSPHysics.

Functions
---------

`relaxzone_from_dsph(xmlfile: str, dirin: str, xloc: float, yloc: float, width: float, zrange: Tuple[float, float], xlayers: int, zlayers: int, *, first_layer: float = 4, last_layer: float = 2, depth: float = 0, swl: float = 0, usevelz: bool = False, smooth: int = 0, psi: float = 0.9, beta: float = 1, drift_corr: float = 0, file_prefix: str = 'RZ_SPH', dirout: str = 'RelaxationZones', binpath: str = None, overwrite_fs: bool = False, overwrite_vel: bool = False, overwrite_xml: bool = False, cleanup: bool = False, dt: float = 0.01, tmin: float = 0.0, tmax: float = 0.0) ‑> None`
:   Create the nessesary csv file to run a DualSPHysics Multi-Layer 1D Piston
    simulation using data from a previous DualSPHysics simulation. The function
    uses "measuretool" to find the surface elevation at a specific x-location
    and creates an 1D grid at every timestep with a given number of vertical layers.
    
    Parameters
    ----------
    xmlfile : path-like or str
        The xml file that defines the simulation. The file will be modified to
        create the "mlayerpistons" element. If no extention is provided the
        code assumes a ".xml" at the end.
    dirin : path-like or str
        The output directory of the old simulation.
    xloc : float
        The x-location the center of the relaxation zone will be.
    yloc : float
        The y-location where the velocities will be interpolated.
    width : float
        The width of the relaxation zone.
    zrange : Tuple[float, float]
        The domain limits of the fluid in the z-direction.
    xlayers : int
        The number of layers within the relaxation zone.
    ylayers : int
        The number of layers in the vertical direction.
    first_layer : float, optional
        The distance from the free surface where the first layer in the
        vertical direction where the velocities will be calculated expressed
        in "dp". E.g. `first_layer = 4` => z0 = 4*dp. By default, 4.
    last_layer : float, optional
        The distance from the free surface where the first layer in the
        vertical direction where the velocities will be calculated expressed
        in "dp". E.g. `last_layer = 4` => z0 = 4*dp. By default, 2.
    depth : float, optional
        Water depth. It is necessary for drift correction, by default 0
    swl : float, optional
        Still water level (free-surface). It is necessary for drift correction,
        by default 0
    smooth : int, optional
        Smooth motion level between layers (xml attribute), by default 0.
    usevelz : bool, optional
        Use velocity in Z or not, by default False
    psi : float, optional
        Coefficient \( \psi \) of Relaxation Zone weight function, by default 0.9
    beta : float, optional
        Coefficient \( \beta \) of Relaxation Zone weight function, by default 1
    drift_corr : float, optional
        Coefficient of drift correction applied in velocity X. 0:Disabled, 1:Full
        correction. By default 0
    dirout : str
        The name of the folder where the csv files will be placed. By default
        "RelaxationZones".
    file_prefix : str, optional
        The prefix of the csv files, by default "RZ_SPH".
    binpath : str, optional
        The path of the binary folder of DualSPHysics. If not defined the an
        environment variable "DUALSPH_HOME" must be defined. By default None.
    overwrite_fs : bool
        If `True` free surface data will be overwritten. By default `False`.
    overwrite_vel : bool
        If `True` the raw velocity data files will be overwritten. By default
        `False`.
    overwrite_xml : bool
        If `True` the "rzwaves_external_1d" element in the xml will be overwitten.
        By default `False`.
    cleanup : bool
        If `True` the raw velocity data files will be removed at the end. Be
        default `False`.
    dt : float
        The dt used for the interpolation of the velocities. The calculated
        velocities are used in a cubic interpolation to created a smoother signal.
    tmin : float
        Defines the start time of the output file.
    tmax : float
        Defines the final time of the output file. If 0 the end time will be
        the final time of the simulation.
    
    Raises
    ------
    FileNotFoundError
        If the xml file could not be found. The rest of the code will still
        run but modification will not be made no valid xml is provided.
    InvalidTimeInterval
        If `tmax` is less than or equal to `tmin`.

`write_rzexternal_xml(xmlfile: Union[str, pathlib.Path], velfile_prefix: str, files_init: int, files_count: int, center: Tuple[float, float, float], width: float, *, depth: float = 0, swl: float = 0, smooth: int = 0, movedata: Tuple[float, float, float] = (0, 0, 0), usevelz: bool = False, psi: float = 0.9, beta: float = 1, drift_corr: float = 0, overwrite: bool = False) ‑> None`
:   Modifies the xml file to add the nessesary fields (in "special") for
    a relaxation zone simulation.
    
    Parameters
    ----------
    xmlfile : path-like or str
        The path to the xml file.
    velfile_prefix : str
        The prefix of the velocity files. To be used for the value in `filesvel`
        element.
    files_init : int
        The number of the first velocity files.
    files_count : int
        The total number of the velocity files to be used.
    width : float
        The width of the zone.
    center : Tuple[float, float, float]
        The center of the zone
    depth : float, optional
        Water depth. It is necessary for drift correction, by default 0
    swl : float, optional
        Still water level (free-surface). It is necessary for drift correction, by
        default 0
    smooth : int, optional
        Smooth motion level, by default 0
    movedata : Tuple[float, float, float], optional
        Movement of data in CSV files, by default (0, 0, 0)
    usevelz : bool, optional
        Use velocity in Z or not, by default False
    psi : float, optional
        Coefficient \( \psi \) of Relaxation Zone weight function, by default 0.9
    beta : float, optional
        Coefficient \( \beta \) of Relaxation Zone weight function, by default 1
    drift_corr : float, optional
        Coefficient of drift correction applied in velocity X. 0:Disabled, 1:Full
        correction. By default 0
    overwrite : bool
        If `True` the "rzwaves_external_1d" element in the xml will be overwitten
        if already exists. By default `False`.
    
    Raises
    ------
    FileNotFoundError
        If the xml file is not found