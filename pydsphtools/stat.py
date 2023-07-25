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
from .imports import *


def agreement_idx(obs: npt.ArrayLike, pred: npt.ArrayLike, c: float = 2) -> float:
    """Calulcates the index of agreement (dr) following the paper by Willmott et al. [1].

    References:
    [1] Cort J. Willmott, Scott M. Robeson, and Kenji Matsuura, "A refined index of model performance",
    International Journal of Climatology, Volume 32, Issue 13, pages 2088-2094, 15 November 2012,
    https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.2419.

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
    """
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ae = np.abs(obs - pred).sum()  # absolute error
    cad = c * np.abs(obs - obs.mean()).sum()  # absolute deviation scaled
    cond = int(ae <= cad)
    return (-1) ** cond * ((ae / cad) ** (2 * cond - 1) - 1)
