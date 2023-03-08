from typing import Union, Tuple, Optional

import numpy as np
from scipy import optimize

from pydsphtools.main import RAD2DEG, DEG2RAD


def find_wavenumber(
    omega: Union[float, np.ndarray], depth: float
) -> Union[float, np.ndarray]:
    """Solves the dispersion relation for a given angular frequency and depth
    and finds the wavenumber.

    Parameters
    ----------
    omega : float or numpy.ndarray
        The angular frequency. Either a float or a numpy array may be passed.
    depth : float
        The water depth.

    Returns
    -------
    float or numpy.ndarray
        The solution to the dispersion equation (the wavenumber). The angular
        frequency is a float then the wavenumber will be a float as well. If
        a numpy array is passed then the wavenumber will be a numpy array for
        with the solution for each element of the angular frequency array.

    Notes
    -----
    The dispersion equation:

    ..math:: \\omega^2 = gk*\\tanh(kh)
    """

    def func(wavenumber: float, omega: float, depth: float) -> float:
        return omega * omega - 9.81 * wavenumber * np.tanh(wavenumber * depth)

    def fprime(wavenumber: float, _: float, depth: float) -> float:
        tanh = np.tanh(wavenumber * depth)
        return -9.81 * tanh - 9.81 * wavenumber * depth * (1.0 - tanh * tanh)

    x0 = 1.0 if isinstance(omega, float) else np.ones(omega.size)
    return optimize.newton(func, x0, fprime=fprime, args=(omega, depth))


def find_celerity(
    wavenumber: Union[float, np.ndarray], depth: float
) -> Union[float, np.ndarray]:
    """Calculates the celerity for a given wavenumber.

    Parameters
    ----------
    wavenumber : float or numpy array
        The wavenumber.
    depth : float
        The water depth.

    Returns
    -------
    float or numpy array
        The culculated celerity. If a numpy array is passed in `wavenumber`
        a numppy array is returned.
    """
    return np.sqrt(9.81 * np.tanh(wavenumber * depth) / wavenumber)


def ricker_spectrum(
    omega: Union[float, np.ndarray], Ar: float, T: float, a: float, m: float
) -> Union[float, np.ndarray]:
    """A more general ricker spectrum implementation based on O.Kimmoun and L.Brosset (2010).

    Parameters
    ----------
    omega : float or numpy array
        The angular frequency.
    Ar : float
        The parameter that controls the amplitude.
    T : float
        Shape and peak frequency parameter 1.
    a : float
        Shape and peak frequency parameter 2.
    m : float
        Shape and peak frequency parameter 3.

    Returns
    -------
    float or numpy array
        The amplitude spectrum for given angular frequency or frequencies.

    Notes
    -----
    The spectrum is calculated using the equation:

    ..math: A_r \\sqrt{T} (1 - \\alpha(\\omega_m T - 1)) e^{-\\omega^m T}

    The peak frequency is given by:

    ..math: \\omega_p = \\left( \\frac{1 + 2\\alpha}{\\alpha T} \\right)^\\frac{1}{m}
    """
    return Ar * np.sqrt(T) * np.exp(-(omega**m) * T) * (1 - a * (omega**m * T - 1))


def ricker_spectrum_simple(
    omega: Union[float, np.ndarray], omegap: float
) -> Union[float, np.ndarray]:
    """A simple ricker spectrum implementation. The spectrum is the same as
    the generalized ricker, `ricker_spectrum`, spectrum with the parameters equal to:
    - Ar = (4 / π)^0.5
    - m = 2
    - a = -1
    - T = (1 / omegap)^2

    Parameters
    ----------
    omega : float or numpy array
        The angular frequency.
    omegap : float
        The peak angular frequency

    Returns
    -------
    float or numpy array
        The amplitude spectrum for given angular frequency or frequencies.

    Notes
    -----
    The spectrum is calculated using the equation:

    ..math: \\frac{2}{\\sqrt{\\pi}} \\frac{\\omega^2}{\\omega_p^3} e^\\frac{-\\omega^2}{\\omega_p^2}
    """
    SQRT_PI = 1.7724538509055159
    omega2 = omega**2
    return 2 * omega2 * np.exp(-omega2 / omegap**2) / (SQRT_PI * omegap**3)


def ricker_wavelet_simple(
    t: Union[float, np.ndarray], omegap: float
) -> Union[float, np.ndarray]:
    """The theoretical wavelet from ricker spectrum.

    Parameters
    ----------
    t : float or numpy array
        The parameter 't' (usually time) of the wavelet.
    omegap : float
        The peak angular frequency

    Returns
    -------
    float or numpy array
        The shape of the wavelet.
    """
    omegap2 = omegap**2
    t2 = t**2
    return (1.0 - 0.5 * omegap2 * t2) * np.exp(-0.25 * omegap2 * t2)


def wavemaker_transfer_func(
    wavenumber: Union[float, np.ndarray],
    depth: float,
    wv_type: str = "flap",
    *,
    hinge: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """For a given wavenumber and depth calculates the stroke to wave height
    ratio for either a piston or flap type wavemaker.

    Parameters
    ----------
    wavenumber : float or np.ndarray
        The wavenumber. Either a float or a numpy array may be passed.
    depth : float
        The water depth.
    wv_type : str, optional
        The type of the wavemaker, either "piston" or "flap". By default "flap"
    hinge : float, optional
        The distance to the bottom of the wavemaker from the still water free surface. If
        `None` is passed, `hinge` is assumed to be equal to `depth`. By default `None`.

    Returns
    -------
    float or np.ndarray
        The height to stroke ratio (H/S), ie the transfer function, of the wavemaker. If
        the wavenumber is passed as a numpy array the return will also be a number array
        with the same dimensions.

    Raises
    ------
    Exception
        Raises exception if:
        - The hinge is less than or equal to zero.
        - An unknown wavemaker type is passed to `wv_type`.

    Notes
    -----
    Equation calculated:

    For piston type wavemaker:

    ..math:: \\left( \\frac{H}{S} \\right)_{piston} = \\frac{2[\\cosh(2kh) - 1]}{2kh + \\sinh(2kh)}

    For flap type wavemaker:

    ..math:: \\left( \\frac{H}{S} \\right)_{flap} = 4\\frac{\\sinh(kh)}{kd}\\frac{\\cosh[k(h - d)] + kd\\sinh(kh) - \\cosh(kh)}{2kh + \\sinh(2kh)}

    where :math: `k` is the wavenumber, :math: `h` is the depth and :math: `d` is the hinge.
    """
    kh = depth * wavenumber
    kh2 = kh * 2.0

    if hinge and hinge < 0:
        raise Exception(f"`hidge` ({hinge}) cannot be less than or equal to zero.")

    if wv_type.lower() == "piston":
        return 2.0 * (np.cosh(kh2) - 1.0) / (kh2 + np.sinh(kh2))

    if wv_type.lower() == "flap":
        d = depth if hinge is None else hinge
        kd = kh if hinge is None else wavenumber * d

        return (
            4.0
            * (np.sinh(kh) / kd)
            * (np.cosh(wavenumber * (depth - d)) + kd * np.sinh(kh) - np.cosh(kh))
            / (kh2 + np.sinh(kh2))
        )

    raise Exception(f"Unknown wavemaker type `{wv_type}`. Expected 'flap' or 'piston'")


def generate_ricker_signal(
    focus_loc: float,
    amplitude: float,
    peak_frequency: float,
    wv_type: str,
    *,
    filepath: str = None,
    angle_units: str = "rad",
    nwaves: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the wavemaker signal from a ricker spectrum to be used in
    a DualSPHysics simulation. The signal (numpy array) is returned and
    saved to a file.

    Parameters
    ----------
    focus_loc : float
        The location of the focusing from the wavemaker.
    amplitude : float
        The amplitude of the focused wave
    peak_frequency : float
        The peak frequency of the ricker spectrum
    wv_type : str, optional
        The type of the wavemaker, either "piston" or "flap". By default "flap"
    filepath : str, optional
        The name of the output file. The path may also be passed. By default "output"
    angle_units : str, optional
        The angle units that will be used, either "rad" or "deg" for the output of a
        flap waverider is used. By default "rad"
    nwaves : int, optional
        The number of waves (ie discrete frequencies) that will be used for the
        calculation. By default, 5000

    Returns
    -------
    np.ndarray
        The time series of the signal.
    np.ndarray
        The generated signal of the wavemaker.

    Raises
    ------
    Exception
        Raises exception if:
        - Unknown angle units are passed to `angle_units`.
        - An unknown wavemaker type is passed to `wv_type`.
    """
    wv_type = wv_type.lower()
    angle_units = angle_units.lower()
    if angle_units not in ("rad", "deg"):
        raise Exception(
            f"The `angle_units` can either be 'rad' or 'deg'. '{angle_units}' was found."
        )

    wp = 2.0 * np.pi * peak_frequency
    depth = 0.35
    freq = np.linspace(1e-6, peak_frequency * 5, nwaves)
    omega = 2.0 * np.pi * freq
    wavenumbers = find_wavenumber(omega, depth)
    spectrum = 2.0 * amplitude * ricker_spectrum_simple(omega, wp)
    height_stroke = wavemaker_transfer_func(wavenumbers, depth, wv_type=wv_type)

    slowest_wave = wavenumbers[-1]
    slowes_speed = find_celerity(slowest_wave, depth)
    xf = focus_loc
    tf = xf / slowes_speed

    time = np.linspace(0, tf, nwaves)
    signal = np.zeros(nwaves)
    domega = omega[1] - omega[0]
    stroke = spectrum / height_stroke
    max_elev = (spectrum * domega).sum()

    if wv_type == "flap":
        stroke /= depth

    for i in range(nwaves):
        signal[i] = (
            stroke * np.cos(omega * time[i] + wavenumbers * xf - omega * tf) * domega
        ).sum()

    # Modification of the signal for initial time using ramp function
    t0 = 5e-2 * tf
    idx_t0 = np.argmax(time > t0)
    signal[:idx_t0] = signal[:idx_t0] * time[:idx_t0] / t0

    # Generate output files per case
    if filepath is None:
        filepath = "output" + (".dat" if wv_type == "piston" else ".csv")

    if wv_type == "piston":
        output = np.stack((time, signal))
        np.savetxt(filepath, output.T, delimiter=" ")

    elif wv_type == "flap":
        signal = np.arctan(signal)
        if angle_units == "deg":
            signal = signal * RAD2DEG
        output = np.stack((time, signal))
        np.savetxt(
            filepath,
            output.T,
            delimiter=";",
            header=f"Time(s);Angle({angle_units})",
            comments="#",
        )

    print(f"Focusing should happen at xf={xf:.2f} m and tf={tf:.2f} sec")
    print(f"Predicted elevation {max_elev:.2f} m")
    return time, signal
