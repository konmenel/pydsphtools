Module pydsphtools.waves
========================
Module containing functions useful when working with oceaning waves.

Functions
---------

`find_celerity(wavenumber: float | Sequence[float], depth: float) ‑> float | numpy.ndarray`
:   Calculates the celerity for a given wavenumber.
    
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
    The celerity \( c \) is calulated from:
    
    .. math::
      c = \sqrt{\frac{g*\tanh(kh)}{k}}
    
    where \( k \) is the wavenumber and \( h \) is the depth.

`find_wavenumber(omega: float | Sequence[float], depth: float) ‑> float | numpy.ndarray`
:   Solves the dispersion relation for a given angular frequency and depth
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
      \omega^2 = gk*\tanh(kh)
    
    where \( k \) is the wavenumber, \( h \) is the depth and \( \omega \)
    is the angular frequency.

`free_surface_elevation(periods: numpy.ndarray, spectrum: numpy.ndarray, x: float | numpy.ndarray, t: float | numpy.ndarray, depth: float, second_order: bool = False, random_seed: Optional[int] = None) ‑> float | numpy.ndarray`
:   Calculates the free surface elevation from a given spectrum.
    
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
      \eta(x, t) = \sum_n A_n \cos(k_n x - \omega_n t + \phi_n)
    
    where \( A_n \) is derived from the input spectrum, \( k_n \) is the
    wavenumber, \( \omega_n \) is the angular frequency, and \( \phi_n \) is
    a random phase.
    
    Second-order theory includes corrections from the Stokes expansion.

`generate_ricker_signal(focus_loc: float, depth: float, amplitude: float, peak_frequency: float, wv_type: str, *, filepath: str = None, hinge: Optional[float] = None, angle_units: str = 'rad', nwaves: int = 5000) ‑> Tuple[numpy.ndarray, numpy.ndarray]`
:   Generates the wavemaker signal from a ricker spectrum to be used in
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

`jonswap_spectrum_frequency(Hs: float, fp: float, freqs: Optional[numpy.ndarray] = None, freq_range: Optional[Tuple[float, float]] = None, nfreqs: int = 100, gamma: float = 3.3) ‑> Tuple[numpy.ndarray, numpy.ndarray]`
:   Calculates the JONSWAP spectrum in the frequency domain for a given 
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

`jonswap_spectrum_period(Hs: float, Tp: float, periods: Optional[numpy.ndarray] = None, period_range: Optional[Tuple[float, float]] = None, nperiods: int = 100, gamma: float = 3.3) ‑> Tuple[numpy.ndarray, numpy.ndarray]`
:   Calculates the JONSWAP spectrum in the period domain for a given 
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

`omega2period(omega: float | Sequence[float]) ‑> float | Sequence[float]`
:   Transforms omega to period (Τ = 2π / ω).
    
    Parameters
    ----------
    omega : float or array-like
        The omega or array of omegas.
    
    Returns
    -------
    float or array-like
        The period or array of periods for each omega.

`period2omega(period: float | Sequence[float]) ‑> float | Sequence[float]`
:   Transforms period to omega (ω = 2π / T).
    
    Parameters
    ----------
    period : float or array-like
        The period or array of periods.
    
    Returns
    -------
    float or array-like
        The omega or array of omegas for each period.

`ricker_spectrum(omega: float | numpy.ndarray, Ar: float, T: float, a: float, m: float) ‑> float | numpy.ndarray`
:   A more general ricker spectrum implementation based on O.Kimmoun and L.Brosset
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
      A_r \sqrt{T} (1 - \alpha(\omega^m T - 1)) e^{-\omega^m T}
    
    The peak frequency is given by:
    
    .. math::
      \omega_p = \left( \frac{1 + 2\alpha}{\alpha T} \right)^\frac{1}{m}

`ricker_spectrum_simple(omega: float | numpy.ndarray, omegap: float) ‑> float | numpy.ndarray`
:   A simple ricker spectrum implementation. The spectrum is the same as
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
      \frac{2}{\sqrt{\pi}} \frac{\omega^2}{\omega_p^3}
      e^\frac{-\omega^2}{\omega_p^2}

`ricker_wavelet_simple(t: float | numpy.ndarray, omegap: float) ‑> float | numpy.ndarray`
:   The theoretical wavelet from ricker spectrum.
    
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

`velocity_2d(periods: numpy.ndarray, spectrum: numpy.ndarray, x: float | numpy.ndarray, z: float | numpy.ndarray, t: float | numpy.ndarray, depth: float, second_order: bool = False, random_seed: Optional[int] = None) ‑> Tuple[float | numpy.ndarray, float | numpy.ndarray]`
:   Calculates the 2D velocity field (horizontal and vertical) from a given spectrum.
    
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
      u(x, z, t) = \sum_n A_n \omega_n \frac{\cosh(k_n(z+h))}{\sinh(k_n h)}
      \cos(k_n x - \omega_n t + \phi_n)
    
    .. math::
      w(x, z, t) = \sum_n A_n \omega_n \frac{\sinh(k_n(z+h))}{\sinh(k_n h)}
      \sin(k_n x - \omega_n t + \phi_n)
    
    where \( h \) is the water depth, \( z \) is the elevation relative to
    the still water surface, and other symbols have their usual meanings.
    
    Second-order theory includes corrections from the Stokes expansion.

`wavelength2wavenumber(wavelength: float | Sequence[float]) ‑> float | Sequence[float]`
:   Transforms wavelength to wavenumber (k = 2π / Λ).
    
    Parameters
    ----------
    wavelength : float or array-like
        The wavelength or array of wavelengths.
    
    Returns
    -------
    float or array-like
        The wavenumber or array of wavenumbers for each wavelength.

`wavemaker_transfer_func(wavenumber: float | numpy.ndarray, depth: float, wv_type: str = 'flap', hinge: Optional[float] = None) ‑> float | numpy.ndarray`
:   For a given wavenumber and depth calculates the stroke to wave height
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
      \left( \frac{H}{S} \right)_{piston} = \frac{2[\cosh(2kh) - 1]}{2kh +
      \sinh(2kh)}
    
    For flap type wavemaker:
    
    .. math::
      \left( \frac{H}{S} \right)_{flap} = 4\frac{\sinh(kh)}{kd}
      \frac{\cosh[k(h - d)] + kd\sinh(kh) - \cosh(kh)}{2kh + \sinh(2kh)}
    
    where \( k \) is the wavenumber, \( h \) is the depth and \( d \)
    is the hinge.

`wavenumber2wavelength(wavenumber: float | Sequence[float]) ‑> float | Sequence[float]`
:   Transforms wavenumber to wavelength (Λ = 2π / k).
    
    Parameters
    ----------
    wavenumber : float or array-like
        The wavenumber or array of wavenumbers.
    
    Returns
    -------
    float or array-like
        The wavelength or array of wavelengths for each wavenumber.