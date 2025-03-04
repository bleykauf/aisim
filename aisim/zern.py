from enum import StrEnum
from functools import partial

import numpy as np
import numpy.typing as npt
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
j_to_n_m_lookup_table = dict(sorted(_wyant_j_to_n_m_lookup_table.items()))


class ZernikeOrder(StrEnum):
    """Enumeration of conventions for ordering Zernike polynomials with a single index."""

    ANSI = "ANSI"
    """Ordering according to ANSI Z80.28-2017."""
    NOLL = "NOLL"
    """Noll's ordering."""
    FRINGE = "FRINGE"
    """Fringe ordering."""
    WYANT = "WYANT"
    """Wyant's ordering."""
    SHS = "SHS"
    """Should be equal to Wyant's ordering."""


class ZernikeNorm(StrEnum):
    """Enumeration of Zernike conventions for normalization.

    References
    ----------
    https://opg.optica.org/abstract.cfm?URI=VSIA-2000-SuC1
    https://opg.optica.org/view_article.cfm?pdfKey=18218be4-47da-44fc-a4a09e5ef667c048_56041
    """

    OSA = "OSA"
    """OSA normalization, sqrt(2n+2) for m=0 and sqrt(n+1) for m!=0."""

    NOLL = "NOLL"
    """Noll's normalization sqrt(n+1)."""


def j_to_n_m(j: int, order: ZernikeOrder) -> tuple[int, int]:
    """Map the single index j to the pair of indices (n, m).

    References
    ----------
    https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py
    """
    match order:
        case ZernikeOrder.NOLL:
            if j < 1:
                raise ValueError(f"Noll's index {j=} must be greater than 0.")
            n = int(np.sqrt(2 * j - 1) + 0.5) - 1
            if n % 2:
                m = 2 * int((2 * (j + 1) - n * (n + 1)) // 4) - 1
            else:
                m = 2 * int((2 * j + 1 - n * (n + 1)) // 4)
            m = m * (-1) ** (j % 2)
        case ZernikeOrder.ANSI:
            if j < 0:
                raise ValueError(f"ANSI index {j=} can not be negative.")
            n = int((np.sqrt(8 * j + 1) - 1) / 2)
            m = 2 * j - n * (n + 2)
        case ZernikeOrder.WYANT | ZernikeOrder.FRINGE:
            if order == ZernikeOrder.WYANT:
                if j < 0:
                    raise ValueError(f"Wyant index {j=} can not be negative.")
                if j >= len(_wyant_j_to_n_m_lookup_table):
                    raise ValueError(f"Wyant index {j=} is out of range.")
            else:
                if j < 1:
                    raise ValueError(f"Fringe index {j=} must be greater than 0.")
                if j > len(_wyant_j_to_n_m_lookup_table):
                    raise ValueError(f"Fringe index {j=} is out of range.")
                j = j - 1
            n, m = _wyant_j_to_n_m_lookup_table[j]
        case _:
            raise ValueError(f"Unknown ordering scheme {order=}.")
    return n, m


def radial(rho: np.ndarray, n: int, m: int) -> np.ndarray:
    """Compute the radial part of the Zernike polynomial."""
    m = np.abs(m)
    if (n - m) % 2:
        r = np.ones_like(rho)
    else:
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
    """Compute the Zernike polynomial of degree n and azimuth m."""
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
    if m == 0:
        return r * norm
    elif m < 0:
        return r * np.sin(np.abs(m) * theta) * norm
    else:
        return r * np.cos(m * theta) * norm


class ZernikePolynomial:
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        order: ZernikeOrder = ZernikeOrder.NOLL,
        norm: ZernikeNorm | None = None,
    ) -> None:
        self.coeffs = np.asarray(coeffs)
        self.order = order
        self.norm = norm
        self.zernike_funcs = []
        for j in range(len(self.coeffs)):
            if self.order == ZernikeOrder.NOLL or self.order == ZernikeOrder.FRINGE:
                j += 1  # Indices start at 1
            n, m = j_to_n_m(j, order)
            self.zernike_funcs.append(partial(zernike_term, n=n, m=m, norm=self.norm))

    def zern_terms(self, rho, theta):
        return np.array(
            [c * f(rho, theta) for c, f in zip(self.coeffs, self.zernike_funcs)]
        )

    def zern_sum(self, rho, theta):
        return np.sum(self.zern_terms(rho, theta), axis=0)


def zern_explicit(rho, theta, coeff):
    """
    Calculate the sum of the first 36 Zernike polynomials.

    Parameters
    ----------
    rho, theta : float or array of float
        Polar coordinates of the position where the sum of Zernike polynomials
        should be calculated.
    coeff : array
        first 36 Zernike coefficients
    r_beam : float
        radius of the wavefront

    Returns
    -------
    values : float or array of float
        value(s) of the wavefront at the probed position
    """
    # precalculating values
    # powers of rho
    rho2 = rho * rho
    rho3 = rho2 * rho
    rho4 = rho3 * rho
    rho5 = rho4 * rho
    rho6 = rho5 * rho
    rho7 = rho6 * rho
    rho8 = rho7 * rho
    rho9 = rho8 * rho
    # cos and sin of n*theta
    costh = np.cos(theta)
    sinth = np.sin(theta)
    cos2th = np.cos(2 * theta)
    sin2th = np.sin(2 * theta)
    cos3th = np.cos(3 * theta)
    sin3th = np.sin(3 * theta)
    cos4th = np.cos(4 * theta)
    sin4th = np.sin(4 * theta)

    coeff = np.array(coeff)

    zern_vals = (
        coeff[0] * np.ones(rho.shape)
        + coeff[1] * rho * costh
        + coeff[2] * rho * sinth
        + coeff[3] * (2 * rho2 - 1)
        + coeff[4] * rho2 * cos2th
        + coeff[5] * rho2 * sin2th
        + coeff[6] * (3 * rho3 - 2 * rho) * costh
        + coeff[7] * (3 * rho3 - 2 * rho) * sinth
        + coeff[8] * (6 * rho4 - 6 * rho2 + 1)
        + coeff[9] * rho3 * cos3th
        + coeff[10] * rho3 * sin3th
        + coeff[11] * (4 * rho4 - 3 * rho2) * cos2th
        + coeff[12] * (4 * rho4 - 3 * rho2) * sin2th
        + coeff[13] * (10 * rho5 - 12 * rho3 + 3 * rho) * costh
        + coeff[14] * (10 * rho5 - 12 * rho3 + 3 * rho) * sinth
        + coeff[15] * (20 * rho6 - 30 * rho4 + 12 * rho2 - 1)
        + coeff[16] * rho4 * cos4th
        + coeff[17] * rho4 * sin4th
        + coeff[18] * (5 * rho5 - 4 * rho3) * cos3th
        + coeff[19] * (5 * rho5 - 4 * rho3) * sin3th
        + coeff[20] * (15 * rho6 - 20 * rho4 + 6 * rho2) * cos2th
        + coeff[21] * (15 * rho6 - 20 * rho4 + 6 * rho2) * sin2th
        + coeff[22] * (35 * rho7 - 60 * rho5 + 30 * rho3 - 4 * rho) * costh
        + coeff[23] * (35 * rho7 - 60 * rho5 + 30 * rho3 - 4 * rho) * sinth
        + coeff[24] * (70 * rho8 - 140 * rho6 + 90 * rho4 - 20 * rho2 + 1)
        + coeff[25] * rho5 * np.cos(5 * theta)
        + coeff[26] * rho5 * np.sin(5 * theta)
        + coeff[27] * (6 * rho6 - 5 * rho4) * cos4th
        + coeff[28] * (6 * rho6 - 5 * rho4) * sin4th
        + coeff[29] * (21 * rho7 - 30 * rho5 + 10 * rho3) * cos3th
        + coeff[30] * (21 * rho7 - 30 * rho5 + 10 * rho3) * sin3th
        + coeff[31] * (56 * rho8 - 105 * rho6 + 60 * rho4 - 10 * rho2) * cos2th
        + coeff[32] * (56 * rho8 - 105 * rho6 + 60 * rho4 - 10 * rho2) * sin2th
        + coeff[33]
        * (126 * rho9 - 280 * rho7 + 210 * rho5 - 60 * rho3 + 5 * rho)
        * costh
        + coeff[34]
        * (126 * rho9 - 280 * rho7 + 210 * rho5 - 60 * rho3 + 5 * rho)
        * sinth
        + coeff[35]
        * (252 * rho**10 - 630 * rho8 + 560 * rho6 - 210 * rho4 + 30 * rho2 - 1)
    )
    return zern_vals
