from enum import StrEnum
from functools import partial
from typing import Callable

import numpy as np
from scipy.special import eval_jacobi


# FIXME: This is not a very elegant solution. IT would be better to find an explicit
# formula for the conversion from Wyant/Fringe single index j to n, m rather than
# populating a lookup table.
def _n_m_to_j_wyant(n, m):
    return int((1 + (n + np.abs(m)) / (2)) ** 2 - 2 * np.abs(m) + (1 - np.sign(m)) / 2)


_wyant_j_to_n_m_lookup_table = {}

for n in range(100):
    for m in reversed(range(-n, n + 1)):
        if (n - abs(m)) % 2 == 0:
            j = int(_n_m_to_j_wyant(n, m))
            _wyant_j_to_n_m_lookup_table[j - 1] = (n, m)
_wyant_j_to_n_m_lookup_table = dict(sorted(_wyant_j_to_n_m_lookup_table.items()))
_fringe_j_to_n_m_lookup_table = {
    idx + 1: val for idx, val in _wyant_j_to_n_m_lookup_table.items()
}


class ZernikeOrder(StrEnum):
    """Enumeration of conventions for ordering Zernike polynomials with a single index.

    There are different ways to order the Zernike polynomials with a single index. The
    most common conventions are the OSA or ANSI convention [1, 2], the Noll
    convention [3], the Fringe/University of Arizona/Air Force convention [4],
    and the Wyant convention [5].

    References
    ----------
    [1] Noll, R. J. (1976).
        Zernike polynomials and atmospheric turbulence*.
        J. Opt. Soc. Am., 66, 207–211.
        https://opg.optica.org/josa/fulltext.cfm?uri=josa-66-3-207&id=56041
    [2] Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., & Webb, R. (2000).
        Standards for Reporting the Optical Aberrations of Eyes.
        Optics InfoBase Conference Papers.
    [3] ANSI Z80.28-2017.
        American National Standard for Ophthalmics - Methods for Reporting Optical
        Aberrations Of Eyes.
    [4] Yen, A. (2021).
        Straightforward path to Zernike polynomials.
        Journal of Micro/Nanopatterning, Materials and Metrology, 20(2).
        https://doi.org/10.1117/1.JMM.20.2.020501
    [5] Wikipedia contributors. (2025).
        Zernike polynomials.
        In Wikipedia, The Free Encyclopedia.
        https://en.wikipedia.org/wiki/Zernike_polynomials
    """

    ANSI = "ANSI"
    """Ordering according to OSA and ANSI Z80.28-2017."""
    NOLL = "NOLL"
    """Noll ordering."""
    FRINGE = "FRINGE"
    """Fringe/University of Arizona/Air Force  ordering."""
    WYANT = "WYANT"
    """Wyant's ordering."""


FIRST_INDEX_J: dict[ZernikeOrder, int] = {
    ZernikeOrder.ANSI: 0,
    ZernikeOrder.NOLL: 1,
    ZernikeOrder.FRINGE: 1,
    ZernikeOrder.WYANT: 0,
}
"""First index j of the Zernike polynomials for each ordering scheme (1 or 0)."""


class ZernikeNorm(StrEnum):
    """Enumeration of Zernike conventions for normalization.

    There are two common conventions for normalizing Zernike polynomials. The Noll
    convention [1] normalizes the polynomials with a factor sqrt(n+1), while the OSA
    convention [2, 3] normalizes the polynomials to have unity variance.

    References
    ----------
    [1] Noll, R. J. (1976).
        Zernike polynomials and atmospheric turbulence*.
        J. Opt. Soc. Am., 66, 207-211.
        https://opg.optica.org/josa/fulltext.cfm?uri=josa-66-3-207&id=56041
    [2] Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., & Webb, R. (2000).
        Standards for Reporting the Optical Aberrations of Eyes.
        Optics InfoBase Conference Papers.
        https://doi.org/10.3928/1081-597x-20020901-30
    [3] Lakshminarayanan, V., & Fleck, A. (2011).
        Zernike polynomials: A guide.
        Journal of Modern Optics (Vol. 58, Issue 7).
        https://doi.org/10.1080/09500340.2011.554896
    """

    NOLL = "NOLL"
    """Noll normalization [1], sqrt(n+1)."""
    OSA = "OSA"
    """OSA normalization [2], sqrt(2n+2) for m=0 and sqrt(n+1) for m!=0."""


def j_to_n_m(j: int, order: ZernikeOrder) -> tuple[int, int]:
    """Map the single index j to the pair of indices (n, m).

    Parameters
    ----------
    j : int
        Single index of the Zernike polynomial.
    order : ZernikeOrder
        Ordering scheme for the Zernike polynomials.

    Returns
    -------
    tuple of int
        Degree and azimuth of the Zernike polynomial.

    Notes
    -----
    Some of the implementations were taken from the hcipy library:
    https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py
    """
    if order not in FIRST_INDEX_J:
        raise ValueError(f"Unknown ordering scheme {order=}.")
    if j < FIRST_INDEX_J[order]:
        raise ValueError(
            f"Index {j=} must be greater or equal to {FIRST_INDEX_J[order]} for {order}."
        )
    match order:
        case ZernikeOrder.NOLL:
            n = int(np.sqrt(2 * j - 1) + 0.5) - 1
            if n % 2:
                m = 2 * int((2 * (j + 1) - n * (n + 1)) // 4) - 1
            else:
                m = 2 * int((2 * j + 1 - n * (n + 1)) // 4)
            m = m * (-1) ** (j % 2)
        case ZernikeOrder.ANSI:
            n = int((np.sqrt(8 * j + 1) - 1) / 2)
            m = 2 * j - n * (n + 2)
        case ZernikeOrder.WYANT:
            if j >= len(_wyant_j_to_n_m_lookup_table):
                raise ValueError(f"Fringe index {j=} is out of range.")
            n, m = _wyant_j_to_n_m_lookup_table[j]
        case ZernikeOrder.FRINGE:
            if j > len(_fringe_j_to_n_m_lookup_table):
                raise ValueError(f"Fringe index {j=} is out of range.")
            n, m = _fringe_j_to_n_m_lookup_table[j]
    return n, m


def radial(rho: np.ndarray, n: int, m: int) -> np.ndarray:
    """Compute the radial part of the Zernike polynomial.

    Parameters
    ----------
    rho : np.ndarray
        Normalized adial coordinate.
    n, m : int
        Degree and azimuth of the Zernike polynomial.

    Returns
    -------
    np.ndarray
        Value of the Zernike polynomial at the given coordinates.
    """
    m = np.abs(m)
    r = (
        (-1) ** ((n - m) // 2)
        * rho**m
        * eval_jacobi((n - m) // 2, m, 0, 1 - 2 * rho**2)
    )
    r[np.isnan(rho)] = np.nan
    return r


def zernike_term(
    rho: np.ndarray, theta: np.ndarray, n: int, m: int, norm: ZernikeNorm | None = None
) -> np.ndarray:
    """Compute the Zernike polynomial of degree n and azimuth m.

    rho : np.ndarray
        Normalized radial coordinate.
    theta : np.ndarray
        Azimuthal coordinate.
    n, m : int
        Degree and azimuth of the Zernike polynomial.
    norm : ZernikeNorm, optional
        Normalization scheme for the Zernike polynomial. Default is None, which means
        no normalization.

    Returns
    -------
    np.ndarray
        Value of the Zernike polynomial at the given coordinates.
    """
    match norm:
        case None:
            norm = 1
        case ZernikeNorm.NOLL:
            norm = np.sqrt(n + 1)
        case ZernikeNorm.OSA:
            norm = np.sqrt((2 * n + 2) / (1 + int(m == 0)))
        case _:
            raise ValueError(f"Unknown normalization scheme {norm=}.")
    r = radial(rho, n, m)
    r = r * norm
    if m == 0:
        return r
    elif m < 0:
        return r * np.sin(np.abs(m) * theta)
    else:
        return r * np.cos(m * theta)


class ZernikePolynomial:
    """Class for evaluating Zernike polynomials.

    Parameters
    ----------
    coeffs : dict
        Coefficients of the Zernike polynomial. The keys of the dict are the single
        indices j of the Zernike polynomials according to the given ordering scheme. If
        the ordering scheme starts at 1, index 0 (if present) is ignored.
    order : ZernikeOrder or str
        Ordering scheme for the Zernike polynomials. Default is ZernikeOrder.NOLL. See
        `ZernikeOrder` for possible values.
    norm : ZernikeNorm, str or None (optional)
        Normalization scheme for the Zernike polynomials. Default is None, which means
        no normalization. See `ZernikeNorm` for possible values.

    Attributes
    ----------
    coeffs : dict
        Coefficients of the Zernike polynomial with the single index j as keys.
    order : ZernikeOrder or str
        Ordering scheme for the Zernike polynomials. See `ZernikeOrder` for possible
        values.
    norm : ZernikeNorm, str or None
        Normalization scheme for the Zernike polynomials. See `ZernikeNorm` for possible
        values.
    zernike_funcs : dict
        Dict of Zernike polynomial functions for each coefficient indexed by  the single
        index j.
    """

    def __init__(
        self,
        coeffs: dict[int, float],
        order: ZernikeOrder = ZernikeOrder.NOLL,
        norm: ZernikeNorm | None = None,
    ) -> None:
        self.coeffs = coeffs
        self.order = order
        self.norm = norm
        self.zernike_funcs: dict[int, Callable] = {}
        for j in self.coeffs:
            n, m = j_to_n_m(j, order)
            self.zernike_funcs[j] = partial(zernike_term, n=n, m=m, norm=self.norm)

    def zern_terms(self, rho: np.ndarray, theta: np.ndarray) -> dict[int, np.ndarray]:
        """Evaluate the Zernike polynomial terms at the given coordinates.

        Parameters
        ----------
        rho :  ndarray
            1 dimensional array of normalized radial coordinates.
        theta : ndarray
            1 dimensional array of azimuthal coordinates. Must have the same length as
            `rho`.

        Returns
        -------
        dict
            Dictionary with the Zernike polynomial terms for all coordinates and for
            each coefficient. The keys are the single indices j depending on the
            ordering scheme.
        """
        terms = {}
        for j in self.coeffs:
            terms[j] = self.coeffs[j] * self.zernike_funcs[j](rho, theta)
        return terms

    def zern_sum(self, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the sum of the Zernike polynomials at the given coordinates.

        Parameters
        ----------
        rho :  ndarray
            1 dimensional array of normalized radial coordinates.
        theta : ndarray
            1 dimensional array of azimuthal coordinates. Must have the same length as
            `rho`.

        Returns
        -------
        ndarray
            n dimensional array of the sum of the Zernike polynomial terms at the n
            coordinates.
        """
        return np.sum(np.array(list(self.zern_terms(rho, theta).values())), axis=0)
