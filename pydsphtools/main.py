"""Copyright (C) 2023 Constantinos Menelaou <https://github.com/konmenel>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@author: Constantinos Menelaou
@github: https://github.com/konmenel
@year: 2023
"""
import io
import re
import pathlib
from typing import Callable, TypeVar, Union, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from pydsphtools.exceptions import NotFoundInOutput

# BUG: There is a bug with the way DualSPHysics creates the `Run.csv`. It replaces all `,` with `;`
#   which causes the Shifting(_txt_,_num_,_num_,_txt_) in the configure section to become
#   Shifting(_txt_;_num_;_num_;_txt_). This causes a parsing bug with `pandas.read_csv` (any csv parser really)

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

_R = TypeVar("_R")


class RE_PATTERNS:
    """Just constant class to store useful regex. Will move into a file if it grows too large."""

    # pattern to capture any number (eg 1.23, -1523, -12.3e-45)
    NUMBER = r"[\-\+]?\d+\.?\d*[Ee]?[\+\-]?\d*"
    # Pattern to captures the chrono floating section of the output. Returns the
    # "ID" and "name" of the chorno object floating
    FLOATING = r"Body_(?P<ID>\d+) \"(?P<name>\w*)\" -  type: Floating"


def read_and_fix_csv(dirout: Union[str, pathlib.Path]) -> io.StringIO:
    """Fixed the bug with the csv where if shifting is present in the `Run.csv` it has key that are
    `;` separated, e.g. Shifting(_txt_;_num_;_num_;_txt_). This causes a parsing bug with `pandas.read_csv`
    (any csv parser really). This function reads the data in fixes the bug in memory.

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.

    Returns
    -------
    io.StringIO
        The corrected IO object to be used instead of the file.

    Examples
    --------
    >>> stream = read_and_fix_csv(dirout=".")
    >>> df = pd.read_csv(stream, sep=";")
    """
    with open(f"{dirout}/Run.csv") as file:
        ORIG_RE = r"Shifting\((\w*);({0});({0});(\w*)\)".format(RE_PATTERNS.NUMBER)
        REPL_RE = r"Shifting(\1:\2:\3:\4)"
        txt = file.read()
        txt = re.sub(ORIG_RE, REPL_RE, txt)
        return io.StringIO(txt)


def get_dp(dirout: Union[str, pathlib.Path]) -> float:
    """Gets the inital particle distance of the simulation, aka `Dp`.

    Parameters
    ----------
    dirout : str, path object or file-like object
        The output directory of the simulation.

    Returns
    -------
    float
        The value of `Dp`.

    Raises
    ------
    NotFoundInOutput
        If `Dp` is not pressent in `Run.out`.
    """
    try:
        stream = read_and_fix_csv(dirout)
        df = pd.read_csv(stream, sep=";")
        return df["Dp"][0]

    except FileNotFoundError:
        with open(f"{dirout}/Run.out") as file:
            for line in file:
                pattern = r"Dp=({0})".format(RE_PATTERNS.NUMBER)
                number = re.match(pattern, line)
                if number:
                    return float(number.groups()[0])

        raise NotFoundInOutput("Variable `Dp`")


def get_usr_def_var(
    dirout: Union[str, pathlib.Path], var: str, dtype: Callable[[str], _R] = float
) -> _R:
    """Finds and parses the value of any user defined variable from the simulation output.

    Parameters
    ----------
    dirout : str, path object or file-like object
        The output directory of the simulation.
    var : str
        The name of the user defined variable.
    dtype : Callable[[str], R], optional
        The return type of the function. The return type will be the same as the
        return type of the callable passed. The callable should accept a string as
        the input. E.g. if `int` is used the return type will be in `int`. By default `float`.

    Returns
    -------
    R
        The value of the variable, By default `float`.

    Raises
    ------
    NotFoundInOutput
        If the user defined variable is not pressent in `Run.out`.
    """

    with open(f"{dirout}/Run.out") as file:
        for line in file:
            if not line.startswith("XML-Vars (uservars + ctes)"):
                continue

            pattern = r"{0}=\[({1})\]".format(var, RE_PATTERNS.NUMBER)
            mass = re.search(pattern, line)
            if mass:
                return dtype(mass.groups()[0])

    raise NotFoundInOutput(f"User defined variable `{var}`")


def get_chrono_mass(dirout: Union[str, pathlib.Path], bname: str) -> float:
    """Finds the mass of a floating chrono body.

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.
    bname : str
        The name of the body.

    Returns
    -------
    float
        The mass of the object

    Raises
    ------
    NotFoundInOutput
        If the either the chorno section or a chrono body with the
        specified name doesn't exist.
    """
    FLOATING_RE = re.compile(RE_PATTERNS.FLOATING)
    PATTERN = r"Mass.........: ({0})".format(RE_PATTERNS.NUMBER)

    with open(f"{dirout}/Run.out") as file:
        found_floating = False
        for line in file:
            if not found_floating:
                res = FLOATING_RE.search(line)
                if res and res.group("name") == bname:
                    found_floating = True
                continue

            mass = re.search(PATTERN, line)
            if mass:
                return float(mass.groups()[0])

    raise NotFoundInOutput(f'Chrono floating mass for "{bname}"')


def get_chrono_inertia(dirout: Union[str, pathlib.Path], bname: str) -> np.ndarray:
    """Finds the inertia tensor of a floating chrono body (only diagonal elements).

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.
    bname : str
        The name of the body.

    Returns
    -------
    np.ndarray
        1D array with the digonal elements of inertia tensor of the body.

    Raises
    ------
    NotFoundInOutput
        If the either the chorno section or a chrono body with the
        specified name doesn't exist.
    """
    FLOATING_RE = re.compile(RE_PATTERNS.FLOATING)
    PATTERN = r"Inertia......: \(({0}),({0}),({0})\)".format(RE_PATTERNS.NUMBER)

    with open(f"{dirout}/Run.out") as file:
        found_floating = False
        for line in file:
            if not found_floating:
                res = FLOATING_RE.search(line)
                if res and res.group("name") == bname:
                    found_floating = True
                continue

            inertia = re.search(PATTERN, line)
            if inertia:
                return np.array([float(i) for i in inertia.groups()])

    raise NotFoundInOutput(f'Chrono floating mass for "{bname}"')


def get_chrono_property(
    dirout: Union[str, pathlib.Path], bname: str, pname: str
) -> Union[float, np.ndarray, str]:
    """Finds and returns any property for a specified chrono floating body.

    Parameters
    ----------
    dirout : str or path object
        The output directory of the simulation.
    bname : str
        The name of the body.
    pname : str
        The name of the property. Should be a property that can be found at
        `Body_XXXX "<bname>" -  type: Floating` section of the `Run.out` file.

    Returns
    -------
    float | np.ndarray | str
        If the property can be interpreted as a number it will return a `float`.
        Else if it can be interpreted as an array it will return an `numpy.array`.
        In any other case it will return the `str` of the property

    Raises
    ------
    NotFoundInOutput
        If the either the chorno section or a chrono body with the specified name
        or property with the specified name doesn't exist.
    """
    FLOATING_RE = re.compile(RE_PATTERNS.FLOATING)
    PATTERN = r"{0}\.*: (.+)".format(pname)
    ELEM_PAT = r"\(({0}),({0}),({0})\)".format(RE_PATTERNS.NUMBER)

    with open(f"{dirout}/Run.out") as file:
        found_floating = False
        for line in file:
            if not found_floating:
                res = FLOATING_RE.search(line)
                if res and res.group("name") == bname:
                    found_floating = True
                continue

            res = re.search(PATTERN, line)
            if res:
                value = res.groups()[0]
                try:
                    return float(value)

                except ValueError as e:
                    if value[0] == "(":
                        elems = re.search(ELEM_PAT, value)
                        elems = elems.groups()
                        return np.array([float(i) for i in elems])

                    return value

    raise NotFoundInOutput(f'Property "{pname}" for chrono body "{bname}"')


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

    def func(wave_number: float, omega: float, depth: float) -> float:
        return omega * omega - 9.81 * wave_number * np.tanh(wave_number * depth)

    def fprime(wave_number: float, _: float, depth: float) -> float:
        tanh = np.tanh(wave_number * depth)
        return -9.81 * tanh - 9.81 * wave_number * depth * (1.0 - tanh * tanh)

    x0 = 1.0 if isinstance(omega, float) else np.ones(omega.size)
    return optimize.newton(func, x0, fprime=fprime, args=(omega, depth))


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


def ricker_spectrum_simple(
    omega: Union[float, np.ndarray], omegap: float
) -> Union[float, np.ndarray]:
    """A simple ricker spectrum implementation.

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
    """
    SQRT_PI = 1.7724538509055159
    omega2 = omega**2
    return 2 * omega2 * np.exp(-omega2 / omegap**2) / (SQRT_PI * omegap**3)


def ricker_wavelet(
    t: Union[float, np.ndarray], omegap: float
) -> Union[float, np.ndarray]:
    """The theoretical wavelet from ricker spectrum

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
    """
    return Ar * np.sqrt(T) * np.exp(-(omega**m) * T) * (1 - a * (omega**m * T - 1))


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
            - The an unknown wavemaker type is passed to `wv_type`.

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
        return 2.0 * (np.cosh(kh2) - 1.0) / (np.sinh(kh2) + kh2)

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
    filepath: str = None,
    angle_units: str = "rad",
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        The generated signal of the wavemaker.

    Raises
    ------
    Exception
        Raises exception if:
        - Unknown angle units are passed to `angle_units`.
        - An unknown wavemaker type is passed to `wv_type`.
    """
    if angle_units.lower() not in ("rad", "deg"):
        raise Exception(
            f"The `angle_units` can either be 'rad' or 'deg'. '{angle_units}' was found."
        )
    wv_type = wv_type.lower()

    wp = 2 * np.pi * peak_frequency
    depth = 0.35
    freq = np.linspace(1e-6, peak_frequency * 5, 5000)
    omega = 2.0 * np.pi * freq
    wavenumbers = find_wavenumber(omega, depth)
    spectrum = amplitude * ricker_spectrum_simple(omega, wp)
    height_stroke = wavemaker_transfer_func(wavenumbers, depth, wv_type=wv_type)

    slowest_wave = wavenumbers[-1]
    slowes_speed = find_celerity(slowest_wave, depth)
    xf = focus_loc
    tf = xf / slowes_speed

    time = np.linspace(0, tf, 5000)
    domega = omega[1] - omega[0]

    signal = np.zeros_like(time)
    stroke = spectrum / height_stroke

    if wv_type == "flap":
        stroke /= depth

    for i in range(len(time)):
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
    return signal


if __name__ == "__main__":
    pass
