Module pydsphtools.postprocess
==============================
A module post-processing tools.

Functions
---------

`compute_floating_motion(diroutdata: str | pathlib.Path, mkbound: int, savefile: str = None, angle_seq: str = 'xyz', max_part: int = -1, float_fmt: str = '%.12e', verbose: bool = True) ‑> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]`
:   Compute rigid-body motion of a floating object from DualSPHysics
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
        (semicolon-separated). Default, `None`.
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