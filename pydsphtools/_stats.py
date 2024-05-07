"""The implementation of basic statistics functions."""

# This file is part of PyDSPHtools. It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/konmenel/pydsphtools/blob/main/LICENSE. No part of PyDSPHtools,
# including this file, may be copied, modified, propagated, or distributed except
# according to the terms contained in the LICENSE file.
import numpy as np
from typing import Sequence, Optional


def agreement_idx(obs: Sequence[float], pred: Sequence[float], c: float = 2) -> float:
    """Calulcates the index of agreement (dr) following the paper by Willmott et al.[1].

    Parameters
    ----------
    obs : numpy array-like
        The observed values as an array-like object
    pred : numpy array-like
        The predicted values as an array-like object
    c : float, optional
        The scaling factor. By default 2

    Returns
    -------
    float
        The  value of the index of agreement

    References
    ----------
    [1] Cort J. Willmott, Scott M. Robeson, and Kenji Matsuura, "A refined index of
    model performance", International Journal of Climatology, Volume 32, Issue 13,
    pages 2088-2094, 15 November 2012,
    https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.2419.
    """
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ae = np.abs(obs - pred).sum()  # absolute error
    cad = c * np.abs(obs - obs.mean()).sum()  # absolute deviation scaled
    cond = int(ae <= cad)
    return (-1) ** cond * ((ae / cad) ** (2 * cond - 1) - 1)


def l1_norm(
    signal1: Sequence[float],
    signal2: Sequence[float],
    x: Optional[Sequence[float]] = None,
    dx: float = 1.0,
) -> float:
    """Calculates the L¹-norm between signal. See Notes for more details.

    Parameters
    ----------
    signal1 : array-like
        The first signal.
    signal2 : array-like
        The second signal
    x : array-like, optional
        The sample points corresponding to the y values. If x is None, the sample points
        are assumed to be evenly spaced dx apart. Default, None.
    dx : scalar, optional
        The spacing between sample points when x is None. Default, 1.

    Returns
    -------
    float
        The L¹-norm of the two signals.

    Notes
    -----
    The L¹-norm is calculated using the following formula:

    .. math::
      L^1 \\text{-norm} =  \\int_{x_0}^{x_f} \\left| f_1(x) - f_2(x) \\right| dx

    where \\( f_1(x) \\) is `signal1` and \\( f_2(x) \\) is `signal2`.
    """
    s1 = np.asarray(signal1, dtype=np.float64)
    s2 = np.asarray(signal2, dtype=np.float64)
    return np.trapz(np.abs(s1 - s2), x=x, dx=dx)


def rmse(
    y: Sequence[float],
    obs: Sequence[float],
) -> float:
    """Calculates Root Mean Square Error between two signals.

    Parameters
    ----------
    y : array-like
        The prediction signal.
    obs : array-like
        The observation signal.

    Returns
    -------
    float
        The RMSE of the two signals.

    Notes
    -----
    The RMSE is calculated using the following formula:

    .. math::
      RMSE = \\sqrt{\\frac{\\sum^N_{i=1} (y_i - y_{obs})^2}{N}}
    """
    assert len(y) == len(obs), "The two signals must have the same length."
    s1 = np.asarray(y, dtype=np.float64)
    s2 = np.asarray(obs, dtype=np.float64)
    return np.sqrt(np.sum((s1 - s2) ** 2) / len(s1))
