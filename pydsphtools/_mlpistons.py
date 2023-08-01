"""The imlpementation of the functions that allows for couple between two DualSPHysics
simulations using the Multi-Layer Pistons approach of DualSPHysics.
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
    "mlpistons2d_from_dsph",
    "mlpistons1d_from_dsph",
    "write_mlpiston2d_xml",
    "write_mlpiston1d_xml",
]


def mlpistons2d_from_dsph(
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
    """
    # File and directory names
    xmlfile = os.path.abspath(xmlfile)
    xmldir, _ = os.path.split(xmlfile)
    config_fpath = f"{dirin}/measuretool/MLPiston2D_config.csv"
    rawdata_fname = "MLPiston2D_data_raw"
    rawdata_fpath = f"{dirin}/measuretool/{rawdata_fname}_Vel.csv"
    freesurface_fname = "MLPiston2D_freesurf"
    freesurface_fpath = f"{dirin}/measuretool/{freesurface_fname}_Elevation.csv"

    dp = get_dp(dirin)

    # Save configuration
    config = pd.DataFrame(
        {"NoLayers_y": [ylayers], "NoLayers_z": [zlayers], "xLocation": [xloc]}
    )

    # Find the free surface for each column (y-direction)
    y0 = yrange[0] + 3 * dp
    yf = yrange[1] - 3 * dp
    ylen = yf - y0
    dy = ylen / ylayers
    z0 = zrange[0]
    zf = zrange[1]
    dz = (zf - z0) / 1000
    ptels_list = [[xloc, 0, xloc], [y0, dy, yf], [z0, dz, zf]]
    exclude_list = ["all"]
    include_list = ["fluid"]
    disable_hvars = ["all"]
    enable_hvars = ["eta"]

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
            highest_point = elevation - 3 * dp
            # dz = highest_point / zlayers
            grid_z[:, j] = np.linspace(highest_point, dp, zlayers)
        points_per_time.append((grid_y, grid_z))

    # Find velocity data from the points and create the dataframe
    outfiles_dir = f"{xmldir}/{dirout}"
    if not os.path.exists(outfiles_dir):
        os.mkdir(outfiles_dir)

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
            free_surface = zs.max() + 3 * dp
            outfiles[key].iloc[i, 0] = time
            outfiles[key].iloc[i, 1 : zlayers + 1] = zs - free_surface
            outfiles[key].iloc[i, zlayers + 1 :] = df_vel.iloc[0, j::ylayers]

    # Save the input files
    written = True
    for i in range(len(ys)):
        fname = f"{outfiles_dir}/{file_prefix}_y{i:02d}.csv"
        key = _get_key_fmt(xloc, ys[i])
        if os.path.exists(fname):
            written = False
            break

        with open(fname, "a") as f:
            f.write(f"{key}\n")
            outfiles[key].to_csv(f, sep=";", index=False)
    if written:
        print(f'Input csv files created in directory "{outfiles_dir}".')

    velfiles = (f"{outfiles_dir}/{file_prefix}_y{i:02d}.csv" for i in range(ylayers))
    write_mlpiston2d_xml(
        xmlfile, mkbound, velfiles, ys, smoothz=smoothz, smoothy=smoothy
    )

    # Clean-up
    if os.path.exists(rawdata_fpath):
        os.remove(rawdata_fpath)
    if os.path.exists(f"{dirin}/measuretool/{rawdata_fname}_PointsDef.vtk"):
        os.remove(f"{dirin}/measuretool/{rawdata_fname}_PointsDef.vtk")


def write_mlpiston2d_xml(
    xmlfile: str,
    mkbound: int,
    velfiles: Iterable[str],
    yvals: Iterable[float],
    *,
    smoothz: int = 0,
    smoothy: int = 0,
) -> None:
    """Modifies the xml file to add the nessesary fields (in "motion" and "special") for
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
    """
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


def mlpistons1d_from_dsph(
    xmlfile: str,
    dirin: str,
    xloc: float,
    yloc: float,
    zrange: Tuple[float, float],
    layers: int,
    mkbound: int,
    *,
    smooth: int = 0,
    file_prefix: str = "MLPiston1D_SPH_velx",
    dirout: str = "MLPiston1D",
    binpath: str = None,
    overwrite: bool = False,
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
    """
    # File and directory names
    xmlfile = os.path.abspath(xmlfile)
    xmldir, _ = os.path.split(xmlfile)
    config_fpath = f"{dirin}/measuretool/MLPiston1D_config.csv"
    freesurface_fname = "MLPiston1D_freesurf"
    freesurface_fpath = f"{dirin}/measuretool/{freesurface_fname}_Elevation.csv"
    rawdata_fname = lambda p: f"MLPiston1D_data_raw_Part{p}"
    rawdata_fpath = (
        lambda p: f"{dirin}/measuretool/rawveldata/{rawdata_fname(p)}_Vel.csv"
    )

    dp = get_dp(dirin)
    first_dist = 4 * dp  # Distance of first poit from free surface
    last_dist = 2 * dp  # Distance of last poit from free surface

    tmax = tmax or get_var(dirin, "PhysicalTime", float)
    if tmax <= tmin:
        raise InvalidTimeInterval(tmin, tmax)

    # Save configuration
    config = pd.DataFrame(
        {"yLocation": [yloc], "NoLayers": [layers], "xLocation": [xloc]}
    )

    # Find the free surface for each column (y-direction)
    ptels_list = [[xloc, 0, xloc], [yloc, 0, yloc], [zrange[0], 0.0001, zrange[1]]]
    exclude_list = ["all"]
    include_list = ["fluid"]
    disable_hvars = ["all"]
    enable_hvars = ["eta"]

    # Check if there are old config files
    old_config = None
    if os.path.exists(config_fpath):
        old_config = np.loadtxt(config_fpath, delimiter=";", skiprows=1)
    # Check if need to update elevation files
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
        # Cleaning old vel files.
        for f in glob.glob(
            f"{dirin}/measuretool/rawveldata/{rawdata_fname('*')}_PointsDef.vtk"
        ):
            os.remove(f)
        for f in glob.glob(rawdata_fpath("*")):
            os.remove(f)

    # Read elevations
    df_fs = pd.read_csv(freesurface_fpath, header=3, sep=";")
    # remove units, eg `Vel [m/s^2]` -> `Vel`
    _sub_map = lambda x: re.sub(r"\ \[[A-Za-z\^0-9/]*\]$", "", x)
    df_fs.columns = df_fs.columns.map(_sub_map)

    # Create the point where the velocities will be calculated
    points_per_time = []
    for _, row in df_fs.iterrows():
        row = row.drop(["Time", "Part"])
        elevation = row.iloc[0]
        highest_point = elevation - first_dist
        zs = np.linspace(highest_point, last_dist, layers)
        points_per_time.append(zs)

    # Find velocity data from the points
    outfiles_dir = f"{xmldir}/{dirout}"
    if not os.path.exists(outfiles_dir):
        os.mkdir(outfiles_dir)

    disable_vars = ["all"]
    enable_vars = ["vel"]

    columns = pd.Series(
        [
            "time",
            *(f"pz_{i}" for i in range(layers)),
            *(f"vel_{i}" for i in range(layers)),
        ]
    )

    # Interpolation array format
    # [[t0,z0],[t0,z1],...,[t0,zn],[t1,z0],...,[tn, zn]]
    sz0 = len(points_per_time) * layers
    interp_pnts = np.zeros((sz0, 2))
    interp_vals = np.zeros(sz0)
    for i, zs in enumerate(points_per_time):
        if df_fs.Time[i] < tmin or df_fs.Time[i] > tmax:
            continue

        if overwrite or not os.path.exists(rawdata_fpath(i)):
            run_measuretool(
                dirin,
                savecsv=rawdata_fname(i),
                dirout=f"{dirin}/measuretool/rawveldata",
                file_nums=(i,),
                pt_list=[(xloc, yloc, z) for z in zs],
                include_types=include_list,
                exclude_types=exclude_list,
                enable_vars=enable_vars,
                disable_vars=disable_vars,
                binpath=binpath,
            )
        df_vel = pd.read_csv(rawdata_fpath(i), header=1, sep=";")
        time = df_vel.loc[0, "Time [s]"]
        df_vel = df_vel.loc[:, df_vel.columns.str.contains("x")]

        idx = i * layers
        interp_pnts[idx : idx + layers, 0] = time
        interp_pnts[idx : idx + layers, 1] = zs
        interp_vals[idx : idx + layers] = df_vel.iloc[0, :].to_numpy()

    # Cubic spline interpolation of velocity at 0.025 timesteps
    fsurf_spline = interpolate.CubicSpline(df_fs["Time"], df_fs["Elevation_0"])
    time_series = np.arange(tmin, tmax + dt, dt)
    szi = len(time_series) * layers
    interp_xi = np.zeros((szi, 2))

    _get_key_fmt = lambda x, y: f"px:;{x};py:;{y}"
    df = pd.DataFrame(columns=columns, index=range(len(time_series)))
    outfile = {_get_key_fmt(xloc, yloc): df.copy()}

    for i, time in enumerate(time_series):
        idx = i * layers
        interp_xi[idx : idx + layers, 0] = time
        interp_xi[idx : idx + layers, 1] = np.linspace(
            fsurf_spline(time) - first_dist, last_dist, layers
        )
    vels = interpolate.griddata(interp_pnts, interp_vals, interp_xi, method="cubic")
    # Fill nans using linear interpolation
    for i in range(layers):
        subsetx = interp_xi[i::layers, 0]
        subsety = vels[i::layers]
        mask = np.isnan(subsety)
        subsety[mask] = np.interp(subsetx[mask], subsetx[~mask], subsety[~mask])
        vels[i::layers] = subsety

    # Prepare output dataframe
    for i, time in enumerate(time_series):
        idx = i * layers
        key = _get_key_fmt(xloc, yloc)
        outfile[key].iloc[i, 0] = time - tmin
        outfile[key].iloc[i, 1 : layers + 1] = interp_xi[
            idx : idx + layers, 1
        ] - fsurf_spline(time)
        outfile[key].iloc[i, layers + 1 :] = vels[idx : idx + layers]

    # Save the input file
    fname = f"{outfiles_dir}/{file_prefix}_y00.csv"
    key = _get_key_fmt(xloc, yloc)
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, "a") as f:
        f.write(f"{key}\n")
        outfile[key].to_csv(f, sep=";", index=False)
    np.savetxt(f"{outfiles_dir}/freesurface_spline.csv", fsurf_spline(time_series))
    print(f'Csv input files created in directory "{outfiles_dir}".')

    velfile = f"{outfiles_dir}/{file_prefix}_y00.csv"
    write_mlpiston1d_xml(xmlfile, mkbound, velfile, smooth=smooth)

    # Clean-up
    if cleanup:
        for f in glob.glob(
            f"{dirin}/measuretool/rawveldata/{rawdata_fname('*')}_PointsDef.vtk"
        ):
            os.remove(f)
        for f in glob.glob(rawdata_fpath("*")):
            os.remove(f)


def write_mlpiston1d_xml(
    xmlfile: str,
    mkbound: int,
    velfile: str,
    *,
    smooth: int = 0,
) -> None:
    """Modifies the xml file to add the nessesary fields (in "motion" and "special") for
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
    """
    # Check if the xml exist
    if not os.path.exists(xmlfile):
        if xmlfile.endswith(".xml"):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlfile)

        xmlfile = f"{xmlfile}.xml"
        if not os.path.exists(xmlfile):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlfile)

    xmldir, _ = os.path.split(xmlfile)
    fpath_rel = os.path.relpath(velfile, xmldir)
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
    if mlayer_elem.find("piston1d") is not None:
        print(
            "*WARNING* [xml file]`piston1d` already exist in xml. Exitting without modifying the xml."
        )
        return
    piston1d = ET.SubElement(mlayer_elem, "piston1d")
    ET.SubElement(
        piston1d,
        "mkbound",
        attrib={"value": str(mkbound), "comment": "Mk-Bound of selected particles"},
    )
    ET.SubElement(
        piston1d,
        "filevelx",
        attrib={"value": fpath_rel, "comment": "File name with X velocity"},
    )
    ET.SubElement(
        piston1d,
        "smooth",
        attrib={"value": str(smooth), "comment": "Smooth motion level (def=0)"},
    )
    print("[xml file] `special` section updated. `piston1d` added.")

    ET.indent(tree, " " * 4)
    tree.write(xmlfile)
