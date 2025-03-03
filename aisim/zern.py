from enum import StrEnum
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.special import eval_jacobi


class ZernikeConvention(StrEnum):
    """Enumeration of Zernike conventions for ordering and normalization."""

    ANSI = "ANSI"
    """Convention according to Z80.28-2017, i.e ANSI's indexing and normalization
    sqrt(2n+2) for m=0 and sqrt(n+1) for m!=0.
    """
    ISO = "ISO"
    """
    Convention according to ISO 24157, i.e. Noll's indexing and normalization sqrt(n+1).
    """
    ISO_NAIVE = "NAIVE"
    """
    Naive implementation of ISO 24157, i.e. Noll's indexing and normalization sqrt(n+1).
    """


def j_to_nm(j: int, convention: ZernikeConvention) -> tuple[int, int]:
    """Map the single index j to the pair of indices (n, m).

    References
    ----------
    https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py
    """
    match convention:
        case ZernikeConvention.ISO:
            if j < 1:
                raise ValueError(f"Noll's index {j=} must be greater than 0.")
            n = int(np.sqrt(2 * j - 1) + 0.5) - 1
            if n % 2:
                m = 2 * int((2 * (j + 1) - n * (n + 1)) // 4) - 1
            else:
                m = 2 * int((2 * j + 1 - n * (n + 1)) // 4)
            m = m * (-1) ** (j % 2)
        case ZernikeConvention.ANSI:
            if j < 0:
                raise ValueError(f"ANSI index {j=} can not be negative.")
            n = int((np.sqrt(8 * j + 1) - 1) / 2)
            m = 2 * j - n * (n + 2)
        case _:
            raise ValueError(f"Unknown convention {convention=}.")
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
    rho: np.ndarray, theta: np.ndarray, n: int, m: int, convention: ZernikeConvention
) -> np.ndarray:
    """Compute the Zernike polynomial of degree n and azimuth m."""
    match convention:
        case ZernikeConvention.ISO:
            norm = np.sqrt(n + 1)
        case ZernikeConvention.ANSI:
            norm = np.sqrt((2 * n + 2) / (1 + int(m == 0)))
        case _:
            raise ValueError(f"Unknown convention {convention=}.")
    r = radial(rho, n, m)
    if m == 0:
        return r * norm
    elif m < 0:
        return r * np.sin(m * theta) * norm
    else:
        return r * np.cos(m * theta) * norm


class ZernikePolynomial:
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        convention: ZernikeConvention = ZernikeConvention.ISO,
    ) -> None:
        self.convention = convention
        self.coeffs = np.asarray(coeffs)
        self.zernike_funcs = []
        for j in range(len(self.coeffs)):
            if self.convention == ZernikeConvention.ISO:
                j += 1  # Noll's index starts from 1
            n, m = j_to_nm(j, convention)
            self.zernike_funcs.append(
                partial(zernike_term, n=n, m=m, convention=convention)
            )

    def zern_terms(self, rho, theta):
        return np.array(
            [c * f(rho, theta) for c, f in zip(self.coeffs, self.zernike_funcs)]
        )

    def zern_sum(self, rho, theta):
        return np.sum(self.zern_terms(rho, theta), axis=0)


def zern_iso_naive(rho, theta, coeff):
    """
    Calculate the sum of the first 36 Zernike polynomials.

    Based on ISO24157:2008.

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
        + coeff[31] * (56 * rho8 - 105 * rho6 + 60 * rho4 - 10 * rho2) * costh
        + coeff[32] * (56 * rho8 - 105 * rho6 + 60 * rho4 - 10 * rho2) * sinth
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
