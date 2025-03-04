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


FIRST_INDEX_J = {
    ZernikeOrder.ANSI: 0,
    ZernikeOrder.NOLL: 1,
    ZernikeOrder.FRINGE: 1,
    ZernikeOrder.WYANT: 0,
}
"""First index j of the Zernike polynomials for each ordering scheme."""


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
        case ZernikeOrder.WYANT | ZernikeOrder.FRINGE:
            j = j - FIRST_INDEX_J[order]
            if j >= len(_wyant_j_to_n_m_lookup_table):
                raise ValueError(f"Fringe index {j=} is out of range.")
            n, m = _wyant_j_to_n_m_lookup_table[j]
    return n, m


def radial(rho: np.ndarray, n: int, m: int) -> np.ndarray:
    """Compute the radial part of the Zernike polynomial."""
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
