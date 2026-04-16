"""Implementation of module for post-processing DualSPHysics simulations."""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from ._main import get_partfiles
from ._io import Bi4File


def compute_floating_motion(
    dirout: str | Path,
    mkbound: int,
    *,
    vreszone: int = -1,
    savefile: str = None,
    create_dirs: bool = True,
    angle_seq: str = "xyz",
    max_part: int = -1,
    float_fmt: str = "%.12e",
    verbose: bool = True,
    vtk_filenames: str = None,
    vtk_dir: str | Path = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute rigid-body motion of a floating object from DualSPHysics
    `Part_*.bi4` or `*.vtk` files.

    The motion is reconstructed by tracking floating particles and computing,
    at each timestep:

    - the center of mass (COM)
    - the rotation relative to the initial configuration

    The rotation is obtained using a least-squares rigid transformation
    (Kabsch algorithm) based on Singular Value Decomposition (SVD), which
    provides a robust estimate even if particle ordering changes during the
    simulation.

    For 3D simulations, the rotation is returned as Euler angles using the
    specified sequence (via `scipy.spatial.transform.Rotation`).
    For 2D simulations, only the in-plane rotation (about the y-axis) is
    computed and returned as a single angle (pitch).

    Parameters
    ----------
    dirout : str | Path
        Path to the output directory of the simulation and associated files
        (e.g. `CaseDamBreak_out`).
    mkbound : int
        Identifier of the floating body (MkBound) to track.
    vreszone : int, optional
        The ID if the variable resolution zone if is used. If negative it is
        ignored. Default, `-1`.
    savefile : str, optional
        If provided, results are saved to this file in text format
        (semicolon-separated). Default, `None`.
    create_dirs : bool, optional
        If provided, creates directories (if necessary) before saving results
        to a file. Ingored if `savefile` is `None`, Default, `True`.
    angle_seq : str, optional
        Euler angle sequence used for 3D rotations (e.g. `"xyz"`, `"zyx"`,
        `"XYZ"`, ...). Passed directly to SciPy. Ignored for 2D simulations.
        Default, `"xyz"`.
    max_part : int, optional
        Maximum Part index to process (e.g. 100 → up to Part_0100). If negative,
        all available files are processed. Default, `-1`.
    float_fmt : str, optional
        Floating-point format used when saving to file (e.g. `"%.8f"`). Defualt,
        `"%.12e"`.
    verbose : bool, optional
        If True, prints progress and diagnostic information. Defualt, `True`.
    vtk_filenames : str, optional
        The pattern of the VTK files of the floating (e.g. `"PartBoulder"` if
        `PartBoulder_*.vtk` are to be used). If provided VTK file will be used
        instead of BI4. Default, `None`.
    vtk_dir : str | Path, optional
        The directory of the VTK files relative to `dirout`. Only used if
        `vtk_filenames` is not `None`. Default, `None`.

    Returns
    -------
    parts : np.ndarray of shape (N,)
        Part indices (Cpart values) corresponding to each timestep.
    times : np.ndarray of shape (N,)
        Simulation time [s] for each timestep.
    coms : np.ndarray of shape (N, 3) or (N, 2)
        Center of mass coordinates:
        - (x, y, z) in 3D
        - (x, z) in 2D
    angles : np.ndarray of shape (N, 3) or (N, 1)
        Rotation angles in degrees:
        - (roll, pitch, yaw) in 3D
        - (pitch,) in 2D (rotation about y-axis)

    Notes
    -----
    - Particle correspondence between timesteps is ensured using particle IDs
      unless boundaryvtk file is used.
    - The rotation is computed relative to the initial configuration
      (`Part_0000` or first file in directory).
    - In 2D mode, only the (x, z) plane is considered.
    """
    dirout = Path(dirout)

    sim2d = _is_sim2d(dirout)

    parts, times, coms, angles = _compute_float_motion(
        dirout,
        mkbound,
        vreszone,
        sim2d,
        angle_seq,
        max_part,
        verbose,
        vtk_filenames,
        vtk_dir,
    )

    if sim2d:
        coms = coms[:, [0, 2]]
        angles = angles[:, [1]]

    if savefile:
        if sim2d:
            columns = [
                "part",
                "time [s]",
                "center.x [m]",
                "center.z [m]",
                "pitch [deg]",
            ]
            header = f"# MkBound={mkbound}."
        else:
            columns = [
                "part",
                "time [s]",
                "center.x [m]",
                "center.y [m]",
                "center.z [m]",
                "roll [deg]",
                "pitch [deg]",
                "yaw [deg]",
            ]
            header = (
                f"# MkBound={mkbound}. Euler angles sequence = '{angle_seq}' (SciPy)\n"
            )

        header += ";".join(columns)
        out = np.concatenate(
            (
                parts.reshape((1, len(parts))).T,
                times.reshape((1, len(times))).T,
                coms,
                angles,
            ),
            axis=1,
        )

        savefile = Path(savefile)
        if create_dirs:
            savefile.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            savefile,
            out,
            fmt=["%d"] + [float_fmt] * (len(columns) - 1),
            delimiter=";",
            header=header,
            comments="",
        )

    return parts, times, coms, angles


def _read_vtk(filepath: Path, mk: int):
    """Internal helper to extract positions and IDs from VTK files."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    mesh = reader.GetOutput()

    point_data = mesh.GetPointData()
    # n_points = mesh.GetNumberOfPoints()

    # Extract positions
    pos = vtk_to_numpy(mesh.GetPoints().GetData()).copy()
    idp_array = point_data.GetArray("Idp")
    idp = vtk_to_numpy(idp_array).copy() if idp_array is not None else None
    mk_array = point_data.GetArray("Mk")
    mk_vals = vtk_to_numpy(mk_array).copy() if mk_array is not None else None

    # Extract TimeStep from field data
    field_data = mesh.GetFieldData()
    time_array = field_data.GetArray("TimeStep")
    time = float(vtk_to_numpy(time_array)[0]) if time_array is not None else 0.0

    # Extract part number from filename
    match = re.search(r"_(\d+)", filepath.stem)
    cpart = int(match.group(1)) if match else 0

    if mk_vals is not None:
        mask = mk_vals == mk
        pos = pos[mask]
        if idp is not None:
            idp = idp[mask]

    return (cpart, time, idp, pos)


def _calculate_angles(x0, xn, sim2d, angle_seq):
    """Calculates rigid-body rotation using the Kabsch algorithm."""
    if sim2d:
        x0 = x0[:, [0, 2]]
        xn = xn[:, [0, 2]]

    h_mat = x0.T @ xn
    svd_res = np.linalg.svd(h_mat)
    u = svd_res[0]
    vh = svd_res[2]

    rmat = vh.T @ u.T

    det = np.linalg.det(rmat)
    if det < 0:
        vh[-1, :] = vh[-1, :] * -1
        rmat = vh.T @ u.T

    if sim2d:
        theta = np.arctan2(rmat[1, 0], rmat[0, 0])
        theta_deg = np.degrees(theta)
        euler_angles = np.array([0.0, theta_deg, 0.0])
    else:
        euler_angles = Rotation.from_matrix(rmat).as_euler(angle_seq, degrees=True)

    return euler_angles


def _compute_float_motion(
    dirout: Path,
    mkbound: int,
    vreszone: int,
    sim2d: bool,
    angle_seq: str,
    max_part: int,
    verbose: bool,
    vtk_filenames: str,
    vtk_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mk, beginp, npfloat = _load_float_info(dirout, mkbound, vreszone, verbose)

    use_vtk = bool(vtk_filenames)
    if use_vtk:
        search_dir = dirout
        if vtk_dir:
            search_dir /= vtk_dir

        vtk_list = list(search_dir.glob(f"{vtk_filenames}_*.vtk"))
        partfiles = sorted(vtk_list)
        nparts = len(partfiles)
        if max_part >= 0:
            pattern = re.compile(r"_(\d+)")
            filtered = []
            for pf in partfiles:
                match = pattern.search(Path(pf).stem)
                if match:
                    if int(match.group(1)) <= max_part:
                        filtered.append(pf)
            partfiles = filtered
            nparts = len(partfiles)
    else:
        search_dir = dirout
        if vreszone < 0:
            search_dir /= "data"
        else:
            search_dir /= f"data_vres{vreszone:02d}"
        partfiles = get_partfiles(search_dir)
        nparts = len(partfiles)

        if max_part >= 0:
            pattern = re.compile(r"Part_(\d+)")
            for count, partfile in enumerate(partfiles, 1):
                part = pattern.search(partfile)
                part = int(part.group(1))
                if part > max_part:
                    nparts = count - 1
                    break

    if verbose:
        print(f"Number of Part files found: {nparts}")
        print(f"Processing {Path(partfiles[0]).name} [{1 / nparts:4.0%}]")

    if use_vtk:
        vtk_data0 = _read_vtk(Path(partfiles[0]), mk)
        part_number = vtk_data0[0]
        timestep = vtk_data0[1]
        idp0 = vtk_data0[2]
        pos0 = vtk_data0[3]
    else:
        part0 = Bi4File(partfiles[0])
        part_number = part0.get_value_by_name("Cpart").value
        timestep = part0.get_value_by_name("TimeStep").value

        idp0 = part0.get_array_by_name("Idp")
        if not idp0:
            raise Exception(f"Array 'Idp' for found in '{part0.filepath}'.")

        pos_array_name = "Pos"
        pos0 = part0.get_array_by_name(pos_array_name)
        if not pos0:
            pos_array_name = "Posd"
            pos0 = part0.get_array_by_name(pos_array_name)

        if not pos0:
            raise Exception(f"Array 'Pos' or 'Posd' for found in '{part0.filepath}'.")

        idx_sort = np.argsort(idp0.data)
        idx_start = beginp
        idx_end = idx_start + npfloat
        idp0 = idp0[idx_sort][idx_start:idx_end]
        pos0 = pos0[idx_sort][idx_start:idx_end]

    com0 = np.mean(pos0, axis=0)

    parts = np.zeros(nparts, dtype=np.int32)
    times = np.zeros(nparts)
    coms = np.zeros((nparts, 3))
    angles = np.zeros((nparts, 3))

    parts[0] = part_number
    times[0] = timestep
    coms[0] = com0
    angles[0] = (0.0, 0.0, 0.0)

    for i in range(1, nparts):
        partfile = Path(partfiles[i])

        if verbose:
            print(f"Processing {partfile.name} [{(i + 1) / nparts:4.0%}]")

        if use_vtk:
            vtk_datan = _read_vtk(partfile, mkbound)
            part_number = vtk_datan[0]
            timestep = vtk_datan[1]
            idpn = vtk_datan[2]
            posn = vtk_datan[3]
        else:
            partn = Bi4File(partfile)
            part_number = partn.get_value_by_name("Cpart").value
            timestep = partn.get_value_by_name("TimeStep").value

            idpn = partn.get_array_by_name("Idp")
            posn = partn.get_array_by_name(pos_array_name)

        if idpn is not None and idp0 is not None:
            sort_idx = np.argsort(idpn)
            idpn_sorted = idpn[sort_idx]
            indices = np.searchsorted(idpn_sorted, idp0)

            valid_mask = (indices < len(idpn_sorted)) & (idpn_sorted[indices] == idp0)
            if not np.all(valid_mask):
                missing = np.sum(~valid_mask)
                print(f"Warning: {missing} particles missing at timestep {part_number}")

            matched_pos0 = pos0[valid_mask]
            matched_posn = posn[sort_idx[indices[valid_mask]]]
        else:
            matched_pos0 = pos0
            matched_posn = posn

        com0 = np.mean(matched_pos0, axis=0)
        comn = np.mean(matched_posn, axis=0)

        x0 = matched_pos0 - com0
        xn = matched_posn - comn

        # Encapsulated call to handle the angle calculations
        euler_angles = _calculate_angles(x0, xn, sim2d, angle_seq)

        parts[i] = part_number
        times[i] = timestep
        coms[i] = comn
        angles[i] = euler_angles

    return parts, times, coms, angles


def _is_sim2d(dirout: Path) -> bool:
    """Check if the simulation is 2D by parsing the execution constants
    from the output XML file.
    """
    xml_files = list(dirout.glob("*.xml"))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        case_node = root.find("case")
        if case_node is None:
            continue

        app_attr = case_node.get("app")
        if app_attr is None or not app_attr.startswith("GenCase"):
            continue

        execution_node = root.find("execution")
        constants_node = execution_node.find("constants")
        data2d_node = constants_node.find("data2d")

        value_str = data2d_node.get("value")
        value_lower = value_str.lower()

        return value_lower == "true"


def _load_float_info(
    dirout: Path, mkbound: int, vreszone: int, verbose: bool
) -> tuple[int, int, int]:
    """Loads floating particle information (begin index and count) by parsing
    the simulation's output XML file.
    """
    glob = "*.xml"
    if vreszone >= 0:
        glob = f"*_vres{vreszone:02d}{glob}"
    xml_files = list(dirout.glob(glob))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {dirout}")

    xml_file = xml_files[0]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    execution_node = root.find("execution")
    if execution_node is None:
        raise ValueError(f"Could not find <execution> tag in {xml_file.name}")

    particles_node = execution_node.find("particles")
    if particles_node is None:
        raise ValueError(
            f"Could not find <particles> tag under <execution> in {xml_file.name}"
        )

    # Search all <floating> tags for the one with the matching mkbound
    for floating in particles_node.findall("floating"):
        mk_str = floating.get("mkbound")

        if mk_str is not None and int(mk_str) == mkbound:
            mk = int(floating.get("mk"))
            beginp = int(floating.get("begin"))
            npfloat = int(floating.get("count"))

            if verbose:
                print(
                    f"Loaded float info for mkbound {mkbound}: begin={beginp}, count={npfloat}"
                )

            return mk, beginp, npfloat

    raise ValueError(
        f"Floating block with mkbound={mkbound} not found in {xml_file.name}"
    )
