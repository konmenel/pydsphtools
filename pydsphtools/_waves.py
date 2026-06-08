"""Contains the implementation of functions useful when working with oceaning waves."""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
from typing import Union, Tuple, Optional, Sequence

import numpy as np
from scipy import optimize
from ._main import RAD2DEG


def period2omega(period: float | Sequence[float]) -> float | Sequence[float]:
    """Transforms period to omega (ω = 2π / T).

    Parameters
    ----------
    period : float or array-like
        The period or array of periods.

    Returns
    -------
    float or array-like
        The omega or array of omegas for each period.
    """
    period = np.asarray(period)
    return 2.0 * np.pi / period


def omega2period(omega: float | Sequence[float]) -> float | Sequence[float]:
    """Transforms omega to period (Τ = 2π / ω).

    Parameters
    ----------
    omega : float or array-like
        The omega or array of omegas.

    Returns
    -------
    float or array-like
        The period or array of periods for each omega.
    """
    omega = np.asarray(omega)
    return 2.0 * np.pi / omega


def wavelength2wavenumber(
    wavelength: float | Sequence[float],
) -> float | Sequence[float]:
    """Transforms wavelength to wavenumber (k = 2π / Λ).

    Parameters
    ----------
    wavelength : float or array-like
        The wavelength or array of wavelengths.

    Returns
    -------
    float or array-like
        The wavenumber or array of wavenumbers for each wavelength.
    """
    wavelength = np.asarray(wavelength)
    return 2.0 * np.pi / wavelength


def wavenumber2wavelength(
    wavenumber: float | Sequence[float],
) -> float | Sequence[float]:
    """Transforms wavenumber to wavelength (Λ = 2π / k).

    Parameters
    ----------
    wavenumber : float or array-like
        The wavenumber or array of wavenumbers.

    Returns
    -------
    float or array-like
        The wavelength or array of wavelengths for each wavenumber.
    """
    wavenumber = np.asarray(wavenumber)
    return 2.0 * np.pi / wavenumber


def find_wavenumber(
    omega: Union[float, Sequence[float]], depth: float
) -> Union[float, np.ndarray]:
    """Solves the dispersion relation for a given angular frequency and depth
    and finds the wavenumber.

    Parameters
    ----------
    omega : float or numpy array-like
        The angular frequency. Either a float or an array-like may be passed.
    depth : float
        The water depth.

    Returns
    -------
    float or numpy array
        The solution to the dispersion equation (the wavenumber). The angular
        frequency is a float then the wavenumber will be a float as well. If
        a numpy array is passed then the wavenumber will be a numpy array for
        with the solution for each element of the angular frequency array.

    Notes
    -----
    The dispersion equation:

    .. math::
      \\omega^2 = gk*\\tanh(kh)

    where \\( k \\) is the wavenumber, \\( h \\) is the depth and \\( \\omega \\)
    is the angular frequency.
    """

    def _func(wavenumber, omega, depth):
        return omega * omega - 9.81 * wavenumber * np.tanh(wavenumber * depth)

    def _fprime(wavenumber, _, depth):
        tanh = np.tanh(wavenumber * depth)
        return -9.81 * tanh - 9.81 * wavenumber * depth * (1.0 - tanh * tanh)

    omega = np.asarray(omega, dtype=np.float64)
    x0 = np.ones(omega.shape)
    ret = optimize.newton(_func, x0, fprime=_fprime, args=(omega, depth))
    if ret.size == 1:
        return float(ret)
    return ret


def find_celerity(
    wavenumber: Union[float, Sequence[float]], depth: float
) -> Union[float, np.ndarray]:
    """Calculates the celerity for a given wavenumber.

    Parameters
    ----------
    wavenumber : float or numpy array-like
        The wavenumber.
    depth : float
        The water depth.

    Returns
    -------
    float or numpy array
        The culculated celerity. If an array-like is passed in `wavenumber`
        a numppy array is returned.

    Notes
    -----
    The celerity \\( c \\) is calulated from:

    .. math::
      c = \\sqrt{\\frac{g*\\tanh(kh)}{k}}

    where \\( k \\) is the wavenumber and \\( h \\) is the depth.
    """
    return np.sqrt(9.81 * np.tanh(wavenumber * depth) / wavenumber)


def ricker_spectrum(
    omega: Union[float, np.ndarray], Ar: float, T: float, a: float, m: float
) -> Union[float, np.ndarray]:
    """A more general ricker spectrum implementation based on O.Kimmoun and L.Brosset
    (2010).

    Parameters
    ----------
    omega : float or numpy array-like
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

    .. math::
      A_r \\sqrt{T} (1 - \\alpha(\\omega^m T - 1)) e^{-\\omega^m T}

    The peak frequency is given by:

    .. math::
      \\omega_p = \\left( \\frac{1 + 2\\alpha}{\\alpha T} \\right)^\\frac{1}{m}
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
    omega : float or numpy array-like
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

    .. math::
      \\frac{2}{\\sqrt{\\pi}} \\frac{\\omega^2}{\\omega_p^3}
      e^\\frac{-\\omega^2}{\\omega_p^2}
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
    t : float or numpy array-like
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
    hinge: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """For a given wavenumber and depth calculates the stroke to wave height
    ratio for either a piston or flap type wavemaker.

    Parameters
    ----------
    wavenumber : float or numpy array-like
        The wavenumber. Either a float or a numpy array-like may be passed.
    depth : float
        The water depth.
    wv_type : str, optional
        The type of the wavemaker, either "piston" or "flap". By default "flap"
    hinge : float, optional
        The distance to the bottom of the wavemaker from the still water free surface.
        If `None` is passed, `hinge` is assumed to be equal to `depth`. By default,
        `None`.

    Returns
    -------
    float or numpy array
        The height to stroke ratio (H/S), ie the transfer function, of the wavemaker. If
        the wavenumber is passed as a numpy array the return will also be a number array
        with the same dimensions.

    Raises
    ------
    Exception

        - The hinge is less than or equal to zero.

        - An unknown wavemaker type is passed to `wv_type`.

    Notes
    -----
    Equation calculated:

    For piston type wavemaker:

    .. math::
      \\left( \\frac{H}{S} \\right)_{piston} = \\frac{2[\\cosh(2kh) - 1]}{2kh +
      \\sinh(2kh)}

    For flap type wavemaker:

    .. math::
      \\left( \\frac{H}{S} \\right)_{flap} = 4\\frac{\\sinh(kh)}{kd}
      \\frac{\\cosh[k(h - d)] + kd\\sinh(kh) - \\cosh(kh)}{2kh + \\sinh(2kh)}

    where \\( k \\) is the wavenumber, \\( h \\) is the depth and \\( d \\)
    is the hinge.
    """
    kh = depth * wavenumber
    kh2 = kh * 2.0

    if hinge is not None and hinge <= 0:
        raise Exception(f"`hinge` ({hinge}) cannot be less than or equal to zero.")

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
    depth: float,
    amplitude: float,
    peak_frequency: float,
    wv_type: str,
    *,
    filepath: str = None,
    hinge: Optional[float] = None,
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
    depth : float
        The water depth
    amplitude : float
        The amplitude of the focused wave
    peak_frequency : float
        The peak frequency of the ricker spectrum
    wv_type : str, optional
        The type of the wavemaker, either "piston" or "flap". By default "flap"
    filepath : str, optional
        The name of the output file. The path may also be passed. By default "output"
    hinge: float, optional
        The distance to the bottom of the wavemaker from the still water free surface.
        If `None` is passed, `hinge` is assumed to be equal to `depth`. By default
        `None`.
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

        - The hinge is less than or equal to zero.

        - Unknown angle units are passed to `angle_units`.

        - An unknown wavemaker type is passed to `wv_type`.
    """
    wv_type = wv_type.lower()
    angle_units = angle_units.lower()
    if angle_units not in ("rad", "deg"):
        raise Exception(
            (
                "The `angle_units` can either be 'rad' or 'deg'"
                f" but '{angle_units}' was found."
            )
        )

    wp = 2.0 * np.pi * peak_frequency
    freq = np.linspace(1e-6, peak_frequency * 5, nwaves)
    omega = 2.0 * np.pi * freq
    wavenumbers = find_wavenumber(omega, depth)
    spectrum = 2.0 * amplitude * ricker_spectrum_simple(omega, wp)
    height_stroke = wavemaker_transfer_func(wavenumbers, depth, wv_type, hinge)

    slowest_wave = wavenumbers[-1]
    slowest_speed = find_celerity(slowest_wave, depth)
    xf = focus_loc
    tf = xf / slowest_speed

    time = np.linspace(0, tf, nwaves)
    signal = np.zeros(nwaves)
    domega = omega[1] - omega[0]
    stroke = spectrum / height_stroke
    max_elev = (spectrum * domega).sum()

    if wv_type == "flap":
        stroke /= depth if hinge is None else hinge

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


def jonswap_spectrum_frequency(
    Hs: float,
    fp: float,
    freqs: Optional[np.ndarray] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    nfreqs: int = 100,
    gamma: float = 3.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the JONSWAP spectrum in the frequency domain for a given 
    significant wave height and peak frequency.

    Parameters
    ----------
    Hs : float
        The significant wave height [m].
    fp : float
        The peak frequency [Hz].
    freqs : numpy array-like, optional
        The frequencies at which to evaluate the spectrum. If None, `freq_range` and
        `nfreqs` are used to generate the frequency array.
    freq_range : tuple, optional
        A tuple (f_min, f_max) defining the frequency range. If None, default is (0.2*fp, 5.0*fp).
    nfreqs : int, optional
        The number of frequency points to generate if `freqs` is None. Default is 100.
    gamma : float, optional
        The peak enhancement factor. Default is 3.3.

    Returns
    -------
    np.ndarray
        The array of frequencies.
    np.ndarray
        The spectral density values for each frequency (S(f)).
    """
    if freqs is None:
        if freq_range is None:
            freq_range = (0.2 * fp, 5.0 * fp)
        freqs = np.linspace(freq_range[0], freq_range[1], nfreqs)
    else:
        freqs = np.asarray(freqs)

    # alpha = 0.0081
    alpha = 1.0 # Does not matter since it is scaled at the end
    sigma = np.where(freqs <= fp, 0.07, 0.09)
    r = np.exp(-((freqs - fp) ** 2) / (2.0 * sigma**2 * fp**2))

    # Standard JONSWAP Formulation S(f)
    spectrum = (
        (alpha * 9.81**2 / ((2.0 * np.pi) ** 4 * freqs**5))
        * np.exp(-1.25 * (fp / freqs) ** 4)
        * gamma**r
    )

    # Scale to match targeted significant wave height
    m0_calculated = np.trapezoid(spectrum, freqs)
    m0_target = (Hs / 4.0) ** 2
    spectrum = spectrum * (m0_target / m0_calculated)

    return freqs, spectrum


def jonswap_spectrum_period(
    Hs: float,
    Tp: float,
    periods: Optional[np.ndarray] = None,
    period_range: Optional[Tuple[float, float]] = None,
    nperiods: int = 100,
    gamma: float = 3.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the JONSWAP spectrum in the period domain for a given 
    significant wave height and peak period.

    Parameters
    ----------
    Hs : float
        The significant wave height [m].
    Tp : float
        The peak period [s].
    periods : numpy array-like, optional
        The periods at which to evaluate the spectrum. If None, `period_range` and
        `nperiods` are used to generate the periods.
    period_range : tuple, optional
        A tuple (T_min, T_max) defining the period range. If None, default is (0.2*Tp, 25*Tp).
    nperiods : int, optional
        The number of period points to generate if `periods` is None. Default is 100.
    gamma : float, optional
        The peak enhancement factor. Default is 3.3.

    Returns
    -------
    np.ndarray
        The array of periods.
    np.ndarray
        The spectral density values for each period (S(T)).
    """
    if periods is None:
        if period_range is None:
            period_range = (0.2 * Tp, 25.0 * Tp)
        periods = np.linspace(period_range[0], period_range[1], nperiods)
    else:
        periods = np.asarray(periods)

    freqs = 1.0 / periods
    fp = 1.0 / Tp

    # alpha = 0.0081
    alpha = 1.0 # Does not matter since it is scaled at the end
    sigma = np.where(freqs <= fp, 0.07, 0.09)
    r = np.exp(-((freqs - fp) ** 2) / (2.0 * sigma**2 * fp**2))

    spectrum_f = (
        (alpha * 9.81**2 / ((2.0 * np.pi) ** 4 * freqs**5))
        * np.exp(-1.25 * (fp / freqs) ** 4)
        * gamma**r
    )
    
    # Transform Spectral Density to Period domain: S(T) = S(f) * f^2
    spectrum = spectrum_f * (freqs**2)

    # Scale to match targeted significant wave height
    m0_calculated = np.trapezoid(spectrum, periods)
    m0_target = (Hs / 4.0) ** 2
    spectrum = spectrum * (m0_target / m0_calculated)

    return periods, spectrum

def free_surface_elevation(
    periods: np.ndarray,
    spectrum: np.ndarray,
    x: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    depth: float,
    second_order: bool = False,
    random_seed: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Calculates the free surface elevation from a given spectrum.

    Parameters
    ----------
    periods : numpy array
        The periods corresponding to the spectrum values [s].
    spectrum : numpy array
        The amplitude spectrum values [m²·s].
    x : float or numpy array-like
        The spatial position(s) [m].
    t : float or numpy array-like
        The time(s) [s].
    depth : float
        The water depth [m].
    second_order : bool, optional
        If True, includes second-order wave theory corrections. *NOTE*: Not tested! By default False.
    random_seed : int, optional
        Seed for reproducible random phase generation. By default None.

    Returns
    -------
    float or numpy array
        The free surface elevation at given position(s) and time(s).

    Notes
    -----
    Uses first-order (Airy) wave theory by default:

    .. math::
      \\eta(x, t) = \\sum_n A_n \\cos(k_n x - \\omega_n t + \\phi_n)

    where \\( A_n \\) is derived from the input spectrum, \\( k_n \\) is the
    wavenumber, \\( \\omega_n \\) is the angular frequency, and \\( \\phi_n \\) is
    a random phase.

    Second-order theory includes corrections from the Stokes expansion.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    periods = np.asarray(periods)
    spectrum = np.asarray(spectrum)

    omega = period2omega(periods)
    wavenumbers = find_wavenumber(omega, depth)

    dperiod = periods[1] - periods[0]
    phases = np.random.uniform(0, 2.0 * np.pi, len(periods))
    # Wave amplitudes from spectrum (A = sqrt(2 * S(T) * dT))
    amplitudes = np.sqrt(2.0 * spectrum * dperiod)

    # Convert coordinates to arrays and map out clean 2D meshes
    x = np.asarray(x)
    t = np.asarray(t)
    xv, tv = np.meshgrid(x, t, indexing='ij')

    eta = np.zeros_like(xv, dtype=np.float64)
    
    # First-order elevation
    for omega_i, k_i, amp_i, phase_i in zip(omega, wavenumbers, amplitudes, phases):
        eta += amp_i * np.cos(k_i * xv - omega_i * tv + phase_i)

    # Second-order corrections (Stokes expansion)
    if second_order:
        for omega_i, k_i, amp_i, phase_i in zip(omega, wavenumbers, amplitudes, phases):
            cosh_factor = np.cosh(2.0 * k_i * depth) / np.sinh(k_i * depth)
            second_order_amp = (
                (1.0 / 8.0)
                * (k_i * amp_i) ** 2
                * cosh_factor
                / np.sinh(k_i * depth) ** 2
            )

            eta += second_order_amp * np.cos(
                2.0 * (k_i * xv - omega_i * tv + phase_i)
            )

    if eta.size == 1:
        return float(eta.flat[0])
    return eta


def velocity_2d(
    periods: np.ndarray,
    spectrum: np.ndarray,
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    depth: float,
    second_order: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calculates the 2D velocity field (horizontal and vertical) from a given spectrum.

    Parameters
    ----------
    periods : numpy array
        The periods corresponding to the spectrum values [s].
    spectrum : numpy array
        The amplitude spectrum values [m²·s].
    x : float or numpy array-like
        The horizontal spatial position(s) [m].
    z : float or numpy array-like
        The vertical position(s) relative to the still water surface (z=0).
        Negative values are below the surface.
    t : float or numpy array-like
        The time(s) [s].
    depth : float
        The water depth (positive value) [m].
    second_order : bool, optional
        If True, includes second-order wave theory corrections. *NOTE*: Not tested! By default False.
    random_seed : int, optional
        Seed for reproducible random phase generation. By default None.

    Returns
    -------
    tuple of (u, w)
        u : float or numpy array
            The horizontal velocity component [m/s].
        w : float or numpy array
            The vertical velocity component [m/s].

    Notes
    -----
    Uses first-order (Airy) wave theory by default:

    .. math::
      u(x, z, t) = \\sum_n A_n \\omega_n \\frac{\\cosh(k_n(z+h))}{\\sinh(k_n h)}
      \\cos(k_n x - \\omega_n t + \\phi_n)

    .. math::
      w(x, z, t) = \\sum_n A_n \\omega_n \\frac{\\sinh(k_n(z+h))}{\\sinh(k_n h)}
      \\sin(k_n x - \\omega_n t + \\phi_n)

    where \\( h \\) is the water depth, \\( z \\) is the elevation relative to
    the still water surface, and other symbols have their usual meanings.

    Second-order theory includes corrections from the Stokes expansion.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    periods = np.asarray(periods)
    spectrum = np.asarray(spectrum)

    omega = period2omega(periods)
    wavenumbers = find_wavenumber(omega, depth)

    dperiod = periods[1] - periods[0]
    phases = np.random.uniform(0, 2.0 * np.pi, len(periods))
    # Wave amplitudes from spectrum
    amplitudes = np.sqrt(2.0 * spectrum * dperiod)

    # Create explicit 3D grid grids across all spatial and time domains
    x = np.asarray(x)
    z = np.asarray(z)
    t = np.asarray(t)
    xv, zv, tv = np.meshgrid(x, z, t, indexing='ij')

    # Convert z coordinates relative to the ocean bed
    z_bottom = zv + depth

    u = np.zeros_like(xv, dtype=np.float64)
    w = np.zeros_like(xv, dtype=np.float64)

    # First-order velocities
    for omega_i, k_i, amp_i, phase_i in zip(omega, wavenumbers, amplitudes, phases):
        sinh_kh = np.sinh(k_i * depth)
        cosh_kh_plus_z = np.cosh(k_i * z_bottom)
        sinh_kh_plus_z = np.sinh(k_i * z_bottom)

        arg = k_i * xv - omega_i * tv + phase_i

        u += amp_i * omega_i * (cosh_kh_plus_z / sinh_kh) * np.cos(arg)
        w += amp_i * omega_i * (sinh_kh_plus_z / sinh_kh) * np.sin(arg)

    # Second-order corrections
    if second_order:
        for omega_i, k_i, amp_i, phase_i in zip(omega, wavenumbers, amplitudes, phases):
            sinh_kh = np.sinh(k_i * depth)
            cosh_2kh_plus_z = np.cosh(2.0 * k_i * z_bottom)
            sinh_2kh_plus_z = np.sinh(2.0 * k_i * z_bottom)

            u_2nd_factor = (
                (3.0 / 8.0) * (k_i * amp_i) ** 2 * cosh_2kh_plus_z / sinh_kh**4
            )
            w_2nd_factor = (
                (3.0 / 8.0) * (k_i * amp_i) ** 2 * sinh_2kh_plus_z / sinh_kh**4
            )

            arg = 2.0 * (k_i * xv - omega_i * tv + phase_i)

            u += u_2nd_factor * np.cos(arg)
            w += w_2nd_factor * np.sin(arg)

    if u.size == 1:
        return float(u.flat[0]), float(w.flat[0])
    return u, w
