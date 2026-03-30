import re
from pathlib import Path
from packaging.version import Version
import numpy as np
from scipy.spatial.transform import Rotation

from ._main import get_partfiles
from ._io import Bi4File


def compute_floating_motion(
    diroutdata: str | Path,
    mkbound: int,
    savefile: str = None,
    angle_seq: str = "xyz",
    max_part: int = -1,
    float_fmt: str = "%.12e",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f"""Compute rigid-body motion of a floating object from DualSPHysics
    ``Part_*.bi4`` files.

    The motion is reconstructed by tracking floating particles belonging to a
    given ``mkbound`` and computing, at each timestep:

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
    diroutdata : str | Path
        Path to the directory containing `Part_*.bi4` and associated files
        (e.g. `PartInfo.ibi4`, `PartFloatInfo.ibi4`).
    mkbound : int
        Identifier of the floating body (MkBound) to track.
    savefile : str, optional
        If provided, results are saved to this file in text format
        (semicolon-separated). Default, `{savefile}`.
    angle_seq : str, optional
        Euler angle sequence used for 3D rotations (e.g. `"xyz"`, `"zyx"`,
        `"XYZ"`, ...). Passed directly to SciPy. Ignored for 2D simulations.
        Default, `"{angle_seq}"`.
    max_part : int, optional
        Maximum Part index to process (e.g. 100 → up to Part_0100). If negative,
        all available files are processed. Default, `{max_part}`.
    float_fmt : str, optional
        Floating-point format used when saving to file (e.g. `"%.8f"`). Defualt,
        `"{float_fmt}"`.
    verbose : bool, optional
        If True, prints progress and diagnostic information. Defualt, `{verbose}`.

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
    - Particle correspondence between timesteps is ensured using particle IDs.
    - The rotation is computed relative to the initial configuration
      (Part_0000).
    - In 2D mode, only the (x, z) plane is considered.
    """
    diroutdata = Path(diroutdata)
    sim2d = _is_sim2d(diroutdata)
    parts, times, coms, angles = _compute_float_motion(
        diroutdata, mkbound, sim2d, angle_seq, max_part, verbose
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
            header = f"# MkBound={mkbound}. Euler angles sequence = '{angle_seq}' (SciPy)\n"

        print(sim2d, columns)
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
        np.savetxt(
            savefile,
            out,
            fmt=["%d"] + [float_fmt] * (len(columns) - 1),
            delimiter=";",
            header=header,
            comments="",
        )

    return parts, times, coms, angles


def _is_sim2d(diroutdata: Path) -> bool:
    diroutdata = diroutdata
    partinfofile = diroutdata / "PartInfo.ibi4"
    partinfo = Bi4File(partinfofile)
    return partinfo.get_value_by_name("Data2d").value


def _compute_float_motion(
    diroutdata: Path,
    mkbound: int,
    sim2d: bool,
    angle_seq: str,
    max_part: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    legacy = _is_legacy_version(diroutdata)
    beginp, npfloat = _load_float_info(diroutdata, mkbound, legacy, verbose)

    partfiles = get_partfiles(diroutdata)
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
        print(f"Number of Part_*.bi4 files found: {nparts}")
        print(f"Processing {Path(partfiles[0]).name} [{1 / nparts:4.0%}]")

    part0 = Bi4File(partfiles[0])
    part_number = part0.get_value_by_name("Cpart").value
    timestep = part0.get_value_by_name("TimeStep").value

    # Need to sort idp
    idp0 = part0.get_array_by_name("Idp")
    pos0 = part0.get_array_by_name("Pos")

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

    for i, partfile in enumerate(partfiles[1:nparts], 1):
        partfile = Path(partfile)

        if verbose:
            print(f"Processing {partfile.name} [{(i + 1) / nparts:4.0%}]")

        partn = Bi4File(partfile)
        part_number = partn.get_value_by_name("Cpart").value
        timestep = partn.get_value_by_name("TimeStep").value

        idpn = partn.get_array_by_name("Idp")
        posn = partn.get_array_by_name("Pos")

        sort_idx = np.argsort(idpn)
        idpn_sorted = idpn[sort_idx]
        indices = np.searchsorted(idpn_sorted, idp0)

        valid_mask = (indices < len(idpn_sorted)) & (idpn_sorted[indices] == idp0)
        if not np.all(valid_mask):
            missing = np.sum(~valid_mask)
            print(f"Warning: {missing} particles missing at timestep {part_number}")

        matched_pos0 = pos0[valid_mask]
        matched_posn = posn[sort_idx[indices[valid_mask]]]

        com0 = np.mean(matched_pos0, axis=0)
        comn = np.mean(matched_posn, axis=0)

        x0 = matched_pos0 - com0
        xn = matched_posn - comn

        if sim2d:
            x0 = x0[:, [0, 2]]
            xn = xn[:, [0, 2]]

        h_mat = x0.T @ xn
        u, _, vh = np.linalg.svd(h_mat)

        Rmat = vh.T @ u.T

        if np.linalg.det(Rmat) < 0:
            vh[-1, :] *= -1
            Rmat = vh.T @ u.T

        if sim2d:
            theta = np.arctan2(Rmat[1, 0], Rmat[0, 0])
            theta_deg = np.degrees(theta)
            euler_angles = np.array([0.0, theta_deg, 0.0])
        else:
            euler_angles = Rotation.from_matrix(Rmat).as_euler(angle_seq, degrees=True)

        parts[i] = part_number
        times[i] = timestep
        coms[i] = comn
        angles[i] = euler_angles

    return parts, times, coms, angles


def _is_legacy_version(diroutdata: Path):
    diroutdata = diroutdata
    partinfofile = diroutdata / "PartInfo.ibi4"
    partinfo = Bi4File(partinfofile)
    appname = partinfo.get_value_by_name("AppName").value
    ver_str_match = re.search(r"v(\d+.\d+.\d+)", appname)
    if not ver_str_match:
        raise Exception(f"Could not find error in '{partinfofile}'")

    version = Version(ver_str_match.group(1))
    return version < Version("5.4")


def _load_float_info(
    diroutdata: Path, mkbound: int, legacy: bool, verbose: bool
) -> tuple[int, int]:
    if legacy:
        floatinfofile = diroutdata / "PartFloat.fbi4"
        floatinginfo = Bi4File(floatinfofile)
        mkbounds = floatinginfo.get_array_by_name("mkbound")
        mkbounds_mask = mkbounds.data == mkbound

        beginp = floatinginfo.get_array_by_name("begin")[mkbounds_mask][0]
        npfloat = floatinginfo.get_array_by_name("count")[mkbounds_mask][0]

    else:
        floatinfofile = diroutdata / "PartFloatInfo.ibi4"
        floatinginfo = Bi4File(floatinfofile)
        mkbounds = floatinginfo.get_array_by_name("MkBound")
        mkbounds_mask = mkbounds.data == mkbound

        beginp = floatinginfo.get_array_by_name("Beginp")[mkbounds_mask][0]
        npfloat = floatinginfo.get_array_by_name("Countp")[mkbounds_mask][0]

    if verbose:
        print(f"Loaded Floating Configuration from {floatinfofile.name}")

    return beginp, npfloat
