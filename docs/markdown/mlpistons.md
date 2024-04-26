Module pydsphtools.mlpistons
============================
The contains functions that allows for couple between two DualSPHysics
simulations using the Multi-Layer Pistons approach of DualSPHysics.

Functions
---------

    
`mlpistons1d_from_dsph(xmlfile: str, dirin: str, xloc: float, yloc: float, zrange: Tuple[float, float], layers: int, mkbound: int, *, smooth: int = 0, file_prefix: str = 'MLPiston1D_SPH_velx', dirout: str = 'MLPiston1D', binpath: str = None, overwrite: bool = False, cleanup: bool = False, dt: float = 0.01, tmin: float = 0.0, tmax: float = 0.0) -> NoneType`
:   Create the nessesary csv file to run a DualSPHysics Multi-Layer 1D Piston
    simulation using data from a previous DualSPHysics simulation. The function
    uses "measuretool" to find the surface elevation at a specific x-location
    and creates an 1D grid at every timestep with a given number of vertical layers.
    
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
    yloc : float
        The y-location where the velocities will be interpolated.
    zrange : Tuple[float, float]
        The domain limits of the fluid in the z-direction.
    layers : int
        The number of layers in the vertical direction.
    mkbound : int
        The mk value of the piston particles.
    smooth : int, optional
        Smooth motion level between layers (xml attribute), by default 0.
    dirout : str
        The name of the folder where the csv files will be placed. By default
        "MLPiston2D".
    file_prefix : str, optional
        The prefix of the csv files, by default "MLPiston2D_SPH_velx".
    binpath : str, optional
        The path of the binary folder of DualSPHysics. If not defined the an
        environment variable "DUALSPH_HOME" must be defined. By default None.
    overwrite : bool
        If `True` the raw velocity data files will be overwritten. By default
        `False`.
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

    
`mlpistons2d_from_dsph(xmlfile: str, dirin: str, xloc: float, yrange: Tuple[float, float], zrange: Tuple[float, float], ylayers: int, zlayers: int, mkbound: int, *, smoothz: int = 0, smoothy: int = 0, file_prefix: str = 'MLPiston2D_SPH_velx', dirout: str = 'MLPiston2D', binpath: str = None) -> NoneType`
:   Create the nessesary csv file to run a DualSPHysics Multi-Layer 2D Piston
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
        Smooth motion level in Z (xml attribute), by default 0.
    smoothy : int, optional
        Smooth motion level in Y (xml attribute), by default 0.
    dirout : str
        The name of the folder where the csv files will be placed. By default
        "MLPiston2D".
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

    
`write_mlpiston1d_xml(xmlfile: str, mkbound: int, velfile: str, *, smooth: int = 0) -> NoneType`
:   Modifies the xml file to add the nessesary fields (in "motion" and "special") for
    a 1D multilayer piston simulation.
    
    Parameters
    ----------
    xmlfile : str
        The path to the xml file.
    mkbound : int
        The mkbound of the piston boundary.
    velfile : str
        The file with the velocity data for the layers.
    smooth : int, optional
        Smooth motion level between layers (xml attribute), by default 0.
    
    Raises
    ------
    FileNotFoundError
        If the xml file is not found

    
`write_mlpiston2d_xml(xmlfile: str, mkbound: int, velfiles: Iterable[str], yvals: Iterable[float], *, smoothz: int = 0, smoothy: int = 0) -> NoneType`
:   Modifies the xml file to add the nessesary fields (in "motion" and "special") for
    a 2D multilayer piston simulation.
    
    Parameters
    ----------
    xmlfile : str
        The path to the xml file.
    mkbound : int
        The mkbound of the piston boundary.
    velfiles : Iterable[str]
        The list of files with the velocity data for the layers.
    yvals : Iterable[float]
        The list of y values of each file in the same order as `velfiles`.
    smoothz : int, optional
        Smooth motion level in Z (xml attribute), by default 0.
    smoothy : int, optional
        Smooth motion level in Y (xml attribute), by default 0.
    
    Raises
    ------
    FileNotFoundError
        If the xml file is not found