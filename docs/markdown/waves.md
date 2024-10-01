Module pydsphtools.waves
========================
Module containing functions useful when working with oceaning waves.

Functions
---------

`find_celerity(wavenumber: Union[float, Sequence[float]], depth: float) ‑> Union[float, numpy.ndarray]`
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

`find_wavenumber(omega: Union[float, Sequence[float]], depth: float) ‑> Union[float, numpy.ndarray]`
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
        Raises exception if:
        - The hinge is less than or equal to zero.
        - Unknown angle units are passed to `angle_units`.
        - An unknown wavemaker type is passed to `wv_type`.

`ricker_spectrum(omega: Union[float, numpy.ndarray], Ar: float, T: float, a: float, m: float) ‑> Union[float, numpy.ndarray]`
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

`ricker_spectrum_simple(omega: Union[float, numpy.ndarray], omegap: float) ‑> Union[float, numpy.ndarray]`
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

`ricker_wavelet_simple(t: Union[float, numpy.ndarray], omegap: float) ‑> Union[float, numpy.ndarray]`
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

`wavemaker_transfer_func(wavenumber: Union[float, numpy.ndarray], depth: float, wv_type: str = 'flap', hinge: Optional[float] = None) ‑> Union[float, numpy.ndarray]`
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
        Raises exception if:
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