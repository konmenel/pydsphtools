"""The implementation of functions that enable coupling of different DualSPHysics
simulations using the Relaxation Zone technic of DualSPHysics. 
"""
# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
from ._imports import *

from pydsphtools._main import *
from .exceptions import InvalidTimeInterval

__all__ = [
    "relaxzone_from_dsph",
    "write_rzexternal_xml",
]


def relaxzone_from_dsph(
    xmlfile: str,
    dirin: str,
    xloc: float,
    yloc: float,
    width: float,
    zrange: Tuple[float, float],
    xlayers: int,
    zlayers: int,
    *,
    first_layer: float = 4,
    last_layer: float = 2,
    depth: float = 0,
    swl: float = 0,
    usevelz: bool = False,
    smooth: int = 0,
    psi: float = 0.9,
    beta: float = 1,
    drift_corr: float = 0,
    file_prefix: str = "RZ_SPH",
    dirout: str = "RelaxationZones",
    binpath: str = None,
    overwrite_fs: bool = False,
    overwrite_vel: bool = False,
    overwrite_xml: bool = False,
    cleanup: bool = False,
    dt: float = 0.01,
    tmin: float = 0.0,
    tmax: float = 0.0,
) -> None:
    """Create the nessesary csv file to run a DualSPHysics Multi-Layer 1D Piston
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
        Coefficient \\( \\psi \\) of Relaxation Zone weight function, by default 0.9
    beta : float, optional
        Coefficient \\( \\beta \\) of Relaxation Zone weight function, by default 1
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
    overwrite_vel : bool
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
    """
    # File and directory names
    xmlfile = Path(xmlfile).absolute()
    xmldir = xmlfile.parent
    dirin = Path(dirin)
    freesurface_fname = "RZ_freesurf"
    freesurface_fpath = dirin / f"measuretool/{freesurface_fname}_Elevation.csv"

    dp = get_dp(dirin)
    first_dist = first_layer * dp  # Distance of first point from free surface
    last_dist = last_layer * dp  # Distance of last point from free surface

    tmax = tmax or get_var(dirin, "PhysicalTime", float)
    if tmax <= tmin:
        raise InvalidTimeInterval(tmin, tmax)

    # Find the free surface for each column (y-direction)
    x0 = xloc - width / 2
    xf = xloc + width / 2
    dx = (xf - x0) / xlayers
    z0 = zrange[0]
    zf = zrange[1]
    dz = (zf - z0) / 1000
    ptels_list = [[x0, dx, xf], [yloc, 0, yloc], [z0, dz, zf]]
    # Current configuration
    config = pd.DataFrame(
        {
            "xLayers": [xlayers],
            "zLayers": [zlayers],
            "xLocation": [xloc],
            "yLocation": [yloc],
            "Width": [width],
        }
    )
    # Get elevations
    _run_elevation(dirin, config, ptels_list, overwrite_fs, binpath)
    df_fs = pd.read_csv(freesurface_fpath, header=3, sep=";")
    # remove units, eg `Vel [m/s^2]` -> `Vel`
    _sub_map = lambda x: re.sub(r"\ \[[A-Za-z\^0-9/]*\]$", "", x)
    df_fs.columns = df_fs.columns.map(_sub_map)

    # Create the points list where the velocities will be calculated
    xs = np.linspace(x0, xf, xlayers)
    zs_per_time: List[np.ndarray] = []
    for _, row in df_fs.iterrows():
        zs = np.zeros((xlayers, zlayers))
        row = row.drop(["Time", "Part"])
        for i in range(xlayers):
            elevation = row.iloc[i]
            highest_point = elevation - first_dist
            zs[i, :] = np.linspace(highest_point, last_dist, zlayers)
        zs_per_time.append((zs))

    # Run velocity raw data
    interp_pnts, interp_velx, interp_velz = _run_vel_raw(
        dirin,
        (xlayers, zlayers),
        df_fs,
        xs,
        yloc,
        zs_per_time,
        (tmin, tmax),
        overwrite_vel,
        binpath,
    )

    # Cubic spline interpolation of velocity at dt
    # Interpolation arrays format:
    # [np.array([t0,z0],[t0,z1],...,[t0,zn],[t1,z0],...,[tn,zn]), <- x0
    #  ...,
    #  np.array([t0,z0],[t0,z1],...,[t0,zn],[t1,z0],...,[tn,zn])] <- xn
    fsurf_splines, time_series, interp_xi, velx, velz = _run_interpolation(
        interp_pnts,
        interp_velx,
        interp_velz,
        (xlayers, zlayers),
        (tmin, tmax),
        dt,
        df_fs,
        (first_dist, last_dist),
    )

    # Prepare output dataframe
    _get_key_fmt = lambda x, y: f"px:;{x};py:;{y}"
    columns = pd.Series(
        [
            "time",
            *(f"pz_{i}" for i in range(zlayers)),
            *(f"vel_{i}" for i in range(zlayers)),
        ]
    )
    df = pd.DataFrame(columns=columns, index=range(len(time_series)))
    outfiles_x = {_get_key_fmt(xs[i], yloc): df.copy() for i in range(xlayers)}
    outfiles_z = {_get_key_fmt(xs[i], yloc): df.copy() for i in range(xlayers)}
    for j in range(xlayers):
        for i, time in enumerate(time_series):
            idx0 = i * zlayers
            idxf = idx0 + zlayers
            key = _get_key_fmt(xs[j], yloc)
            pzs = interp_xi[j][idx0:idxf, 1]
            outfiles_x[key].iloc[i, 0] = time - tmin
            outfiles_x[key].iloc[i, 1 : zlayers + 1] = pzs
            outfiles_x[key].iloc[i, zlayers + 1 :] = velx[j][idx0:idxf]

            outfiles_z[key].iloc[i, 0] = time - tmin
            outfiles_z[key].iloc[i, 1 : zlayers + 1] = pzs
            outfiles_z[key].iloc[i, zlayers + 1 :] = velz[j][idx0:idxf]

    # Save the input files
    outfiles_dir = xmldir / dirout
    if not os.path.exists(outfiles_dir):
        os.mkdir(outfiles_dir)

    for i in range(xlayers):
        for vel, outfiles in zip(("velx", "velz"), (outfiles_x, outfiles_z)):
            fname = outfiles_dir / f"{file_prefix}_{vel}_x{i:02d}_y00.csv"
            key = _get_key_fmt(xs[i], yloc)
            if os.path.exists(fname):
                os.remove(fname)
            with open(fname, "a") as f:
                f.write(f"{key}\n")
                outfiles[key].to_csv(f, sep=";", index=False)
        # Save cubic spline of free surface for debugging purposes.
        np.savetxt(
            outfiles_dir / f"freesurface_spline_x{i:02d}.csv",
            fsurf_splines[j](time_series),
        )
    print(f'Csv input files created in directory "{outfiles_dir}".')

    write_rzexternal_xml(
        xmlfile,
        file_prefix,
        0,
        xlayers,
        (xloc, yloc, z0),
        width,
        depth=depth,
        swl=swl,
        smooth=smooth,
        usevelz=usevelz,
        psi=psi,
        beta=beta,
        drift_corr=drift_corr,
        overwrite=overwrite_xml,
    )

    # Clean-up
    if cleanup:
        _clean_raw_vel_data(dirin)


def write_rzexternal_xml(
    xmlfile: Union[Path, str],
    velfile_prefix: str,
    files_init: int,
    files_count: int,
    center: Tuple[float, float, float],
    width: float,
    *,
    depth: float = 0,
    swl: float = 0,
    smooth: int = 0,
    movedata: Tuple[float, float, float] = (0, 0, 0),
    usevelz: bool = False,
    psi: float = 0.9,
    beta: float = 1,
    drift_corr: float = 0,
    overwrite: bool = False,
) -> None:
    """Modifies the xml file to add the nessesary fields (in "special") for
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
        Coefficient \\( \\psi \\) of Relaxation Zone weight function, by default 0.9
    beta : float, optional
        Coefficient \\( \\beta \\) of Relaxation Zone weight function, by default 1
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
    """
    xmlfile = Path(xmlfile)
    # Check if the xml exist
    if not os.path.exists(xmlfile):
        if xmlfile.endswith(".xml"):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlfile)

        xmlfile = f"{xmlfile}.xml"
        if not os.path.exists(xmlfile):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlfile)

    tree = ET.parse(xmlfile)
    # Add to `special` section
    elem_exec = xml_get_or_create_subelement(tree, "execution")
    elem_special = xml_get_or_create_subelement(elem_exec, "special")

    rz_elem = xml_get_or_create_subelement(elem_special, "relaxationzones")
    rz1d = rz_elem.find("rzwaves_external_1d")

    if not overwrite and rz1d is not None:
        print(
            "*WARNING* [xml file] `rzwaves_external_1d` already exist in xml. Exitting without modifying the xml."
        )
        return

    if overwrite and rz1d is not None:
        rz_elem.remove(rz1d)

    rz1d = ET.SubElement(rz_elem, "rzwaves_external_1d")
    ET.SubElement(
        rz1d,
        "depth",
        attrib={
            "value": str(depth),
            "comment": "Water depth. It is necessary for drift correction (def=0)",
        },
    )
    ET.SubElement(
        rz1d,
        "swl",
        attrib={
            "value": str(swl),
            "comment": "Still water level (free-surface). It is necessary for drift correction (def=0)",
        },
    )
    ET.SubElement(
        rz1d,
        "filesvel",
        attrib={
            "value": velfile_prefix,
            "comment": "Main name of files with velocity to use",
        },
    )
    ET.SubElement(
        rz1d,
        "filesvelx",
        attrib={
            "initial": str(files_init),
            "count": str(files_count),
            "comment": "First file and count to use",
        },
    )
    ET.SubElement(
        rz1d,
        "usevelz",
        attrib={
            "value": str(usevelz).lower(),
            "comment": "Use velocity in Z or not (def=false)",
        },
    )
    ET.SubElement(
        rz1d,
        "movedata",
        attrib={
            "x": str(movedata[0]),
            "y": str(movedata[1]),
            "z": str(movedata[2]),
            "comment": "Movement of data in CSV files",
        },
    )
    ET.SubElement(
        rz1d,
        "smooth",
        attrib={
            "value": str(smooth),
            "comment": "Smooth motion level (def=0)",
        },
    )
    ET.SubElement(
        rz1d,
        "center",
        attrib={
            "x": str(center[0]),
            "y": str(center[1]),
            "z": str(center[2]),
            "comment": "Central point of application",
        },
    )
    ET.SubElement(
        rz1d,
        "width",
        attrib={
            "value": str(width),
            "comment": "Width for generation",
        },
    )
    ET.SubElement(
        rz1d,
        "function",
        attrib={
            "psi": str(psi),
            "beta": str(beta),
            "comment": "Coefficients in function for velocity (def. psi=0.9, beta=1)",
        },
    )
    ET.SubElement(
        rz1d,
        "driftcorrection",
        attrib={
            "value": str(drift_corr),
            "comment": "Coefficient of drift correction applied in velocity X. 0:Disabled, 1:Full correction (def=0)",
        },
    )
    print("[xml file] `special` section updated. `rzwaves_external_1d` added.")

    ET.indent(tree, " " * 4)
    tree.write(xmlfile)


def _run_elevation(
    dirin: Path,
    config: pd.DataFrame,
    ptels_list: List[List[float]],
    overwrite_fs: bool,
    binpath: str = None,
) -> None:
    """Checks if there is an old elevation configuration that matches. If not uses
    measuretool to find elevation.
    """
    config_fpath = dirin / "measuretool/RZ_config.csv"
    freesurface_fname = "RZ_freesurf"
    freesurface_fpath = dirin / f"measuretool/{freesurface_fname}_Elevation.csv"

    exclude_list = ["all"]
    include_list = ["fluid"]
    disable_hvars = ["all"]
    enable_hvars = ["eta"]
    old_config = None
    if os.path.exists(config_fpath):
        old_config = np.loadtxt(config_fpath, delimiter=";", skiprows=1)
    # Check if need to update elevation files
    if (
        overwrite_fs
        or not os.path.exists(freesurface_fpath)
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
            kclimit=0.2,
            enable_hvars=enable_hvars,
            disable_hvars=disable_hvars,
            binpath=binpath,
            print_options=True,
        )

        _clean_raw_vel_data(dirin)


def _run_vel_raw(
    dirin: Path,
    layers: Tuple[int, int],
    df_fs: pd.DataFrame,
    xs: np.ndarray,
    yloc: float,
    zs_per_time: List[np.ndarray],
    tlim: Tuple[float, float],
    overwrite_vel: bool,
    binpath: str = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Runs the raw measuretool to get raw velocity if needed and creates the
    arrays to be used for the interpolation of the velocities.

    Returns
    -------
    interp_pnts : List of np.ndarrays
        A list of with as many elements as the number of layers in the x direction
        (`layers[0]`). Each element is an 2d array of the interpolation points of
        size `(len(zs_per_time) * layers[1], 2)`. Check notes.
    interp_velx : List of np.ndarrays
        A list of with as many elements as the number of layers in the x direction
        (`layers[0]`). Each element is an 1d array  of the x-velocity for each point
        of `inter_pnts`. Size: (`len(zs_per_time) * layers[1]`,). Check notes.
    interp_velz : List of np.ndarrays
        A list of with as many elements as the number of layers in the x direction
        (`layers[0]`). Each element is an 1d array  of the z-velocity for each point
        of `inter_pnts`. Size: (`len(zs_per_time) * layers[1]`,). Check notes.

    Notes
    -----
    Interpolation point array format:
    [np.array([t0,z0],[t0,z1],...,[t0,zn],[t1,z0],...,[tn,zn]), <- x0
     np.array([t0,z0],[t0,z1],...,[t0,zn],[t1,z0],...,[tn,zn]), <- x1
     ...,
     np.array([t0,z0],[t0,z1],...,[t0,zn],[t1,z0],...,[tn,zn])] <- xn

    where xi is value of x at the ith layer, zi is value of x at the ith layer and
    ti is the time that the ith time-step.

    Similarly for the velocity arrays
    """
    rawdata_fname = lambda p: f"RZ_data_raw_Part{p}"
    rawdata_fpath = (
        lambda p: dirin / f"measuretool/rawveldata/{rawdata_fname(p)}_Vel.csv"
    )

    exclude_list = ["all"]
    include_list = ["fluid"]
    disable_vars = ["all"]
    enable_vars = ["vel"]

    tmin, tmax = tlim
    xlayers, zlayers = layers
    sz0 = len(zs_per_time) * zlayers
    interp_pnts = [np.zeros((sz0, 2)) for _ in range(xlayers)]
    interp_velx = [np.zeros(sz0) for _ in range(xlayers)]
    interp_velz = [np.zeros(sz0) for _ in range(xlayers)]
    for i, zs in enumerate(zs_per_time):
        if df_fs.Time[i] < tmin or df_fs.Time[i] > tmax:
            continue

        if overwrite_vel or not os.path.exists(rawdata_fpath(i)):
            run_measuretool(
                dirin,
                savecsv=rawdata_fname(i),
                dirout=dirin / "measuretool/rawveldata",
                file_nums=(i,),
                pt_list=[
                    (xs[r], yloc, z) for r in range(zs.shape[0]) for z in zs[r, :]
                ],
                # ptls_list=ptls_list,
                include_types=include_list,
                exclude_types=exclude_list,
                enable_vars=enable_vars,
                disable_vars=disable_vars,
                binpath=binpath,
                print_options=False,
            )
        df_vel = pd.read_csv(rawdata_fpath(i), header=1, sep=";")
        time = df_vel.loc[0, "Time [s]"]
        df_velx = df_vel.loc[:, df_vel.columns.str.contains("x")]
        df_velz = df_vel.loc[:, df_vel.columns.str.contains("z")]

        for j in range(xlayers):
            tidx0 = i * zlayers
            tidxf = tidx0 + zlayers
            vidx0 = j * zlayers
            vidxf = vidx0 + zlayers
            interp_pnts[j][tidx0:tidxf, 0] = time
            interp_pnts[j][tidx0:tidxf, 1] = zs[j, :]
            interp_velx[j][tidx0:tidxf] = df_velx.iloc[0, vidx0:vidxf].to_numpy()
            interp_velz[j][tidx0:tidxf] = df_velz.iloc[0, vidx0:vidxf].to_numpy()

    # Remove unnessesary rows
    tidx0 = np.argmax(tmin <= df_fs.Time) * zlayers
    tidxf = (np.argmax(df_fs.Time >= tmax) or len(df_fs.Time)) * zlayers
    for j in range(xlayers):
        interp_pnts[j] = interp_pnts[j][tidx0:tidxf, :]
        interp_velx[j] = interp_velx[j][tidx0:tidxf]
        interp_velz[j] = interp_velz[j][tidx0:tidxf]

    return interp_pnts, interp_velx, interp_velz


def _run_interpolation(
    interp_pnts: List[np.ndarray],
    interp_velx: List[np.ndarray],
    interp_velz: List[np.ndarray],
    layers: Tuple[int, int],
    tlim: Tuple[float, float],
    dt: float,
    df_fs: pd.DataFrame,
    dists: Tuple[float, float],
) -> Tuple[List[interpolate.CubicSpline], List[np.ndarray], List[np.ndarray]]:
    """Interpolates and returns the raw velocity data at specified dt. The array
    format is the same with more dense time data. Also, returns the free surface cubic spline for each layer in the x direction.

    Returns
    -------
    fsurf_splines : List of scipy's cubic splines
        The free surface for each layer in x direction
    time_series : numpy array
        The time data used in the interpolation
    interp_xi : List of numpy arrays
        The points where the velocity was interpolated
    velx : List of numpy arrays
        The interpolated x-velocity for each layer in x direction
    velz : List of numpy arrays
        The interpolated z-velocity for each layer in x direction
    """
    xlayers, zlayers = layers
    tmin, tmax = tlim
    first_dist, last_dist = dists

    time_series = np.arange(tmin, tmax + dt, dt)
    szi = len(time_series) * zlayers
    interp_xi = [np.zeros((szi, 2)) for _ in range(xlayers)]
    velx: List[np.ndarray] = [None] * xlayers
    velz: List[np.ndarray] = [None] * xlayers
    fsurf_splines = [None] * xlayers
    for i in range(xlayers):
        fsurf_splines[i] = interpolate.CubicSpline(
            df_fs["Time"], df_fs[f"Elevation_{i}"]
        )

        for j, time in enumerate(time_series):
            idx0 = j * zlayers
            idxf = idx0 + zlayers
            interp_xi[i][idx0:idxf, 0] = time
            interp_xi[i][idx0:idxf, 1] = np.linspace(
                last_dist, fsurf_splines[i](time) - first_dist, zlayers
            )

        velx[i] = interpolate.griddata(
            interp_pnts[i], interp_velx[i], interp_xi[i], method="cubic"
        )
        velz[i] = interpolate.griddata(
            interp_pnts[i], interp_velz[i], interp_xi[i], method="cubic"
        )
        # Fill nans using linear interpolation
        for j in range(zlayers):
            subsetx = interp_xi[i][j::zlayers, 0]
            subsetvelx = velx[i][j::zlayers]
            subsetvelz = velz[i][j::zlayers]
            maskx = np.isnan(subsetvelx)
            maskz = np.isnan(subsetvelz)
            subsetvelx[maskx] = np.interp(
                subsetx[maskx], subsetx[~maskx], subsetvelx[~maskx]
            )
            subsetvelz[maskz] = np.interp(
                subsetx[maskz], subsetx[~maskz], subsetvelz[~maskz]
            )
            velx[i][j::zlayers] = subsetvelx
            velz[i][j::zlayers] = subsetvelz
    return fsurf_splines, time_series, interp_xi, velx, velz


def _clean_raw_vel_data(dirin: Union[str, Path]) -> None:
    """Cleans old vel files."""
    dirin
    rawdata_fname = lambda p: f"RZ_data_raw_Part{p}"
    rawdata_fpath = (
        lambda p: dirin / f"measuretool/rawveldata/{rawdata_fname(p)}_Vel.csv"
    )

    for f in glob.glob(
       str(dirin / f"measuretool/rawveldata/{rawdata_fname('*')}_PointsDef.vtk")
    ):
        os.remove(f)
    for f in glob.glob(str(rawdata_fpath("*"))):
        os.remove(f)
