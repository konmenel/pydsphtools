import numpy as np
import numpy.typing as npt


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
    ae = np.abs(obs - pred).sum()               # absolute error
    cad = c * np.abs(obs - obs.mean()).sum()    # absolute deviation scaled
    cond = int(ae <= cad)
    return (-1) ** cond * ((ae / cad) ** (2*cond - 1) - 1)

