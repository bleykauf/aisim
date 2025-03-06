"""Classes and functions related to the interferometry lasers."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from . import convert
from .zern import FIRST_INDEX_J, ZernikeNorm, ZernikeOrder, ZernikePolynomial


class Wavevectors:
    """
    Class that defines the wave vectors of the two Ramen beams.

    Parameters
    ----------
    k1, k2 : float
        1D wave vectors (wavenumber) in z-direction of the two Raman beams  in rad/m,
        defaults to 2*pi/780e-9

    Attributes
    ----------
    k1, k2 : float
        1D wave vectors (wavenumber) in z-direction of the two Raman beams in rad/m
    """

    def __init__(self, k1=8055366, k2=-8055366):
        self.k1 = k1
        self.k2 = k2

    def doppler_shift(self, atoms):
        """
        Calculate the Doppler shifts for an atomic ensemble.

        Parameters
        ----------
        atoms : AtomicEnsemble
            an atomic enemble with a finite velocity in the z direction

        Returns
        -------
        dopler_shift : ndarray
            Doppler shift of each atom in the ensemble in rad/s
        """
        # calculate two photon detuning for atoms' velocity (-v*k_eff)
        velocity_z = atoms.velocity[:, 2]
        doppler_shift = -velocity_z * (self.k1 - self.k2)
        return doppler_shift


class IntensityProfile:
    """
    Class that defines a Gaussian intensity profile in terms of the Rabi frequency.

    Parameters
    ----------
    r_profile : float
        1/e² radius of the Gaussian intensity profile in m
    center_rabi_freq : float
        Rabi frequency at center of intensity profile in rad/s
    r_beam : float (optional)
        Beam radius in m. Can be set if the intensity profile is limited by an aperture.
        Rabi frequency will be set to 0 outside of the beam.

    Attributes
    ----------
    r_profile : float
        1/e² radius of the Gaussian intensity profile in m
    center_rabi_freq : float
        Rabi frequency at center of intensity profile in rad/s
    r_beam : float or None
        Beam radius in m. If set, Rabi frequency will be set to 0 outside of the beam.

    """

    def __init__(self, r_profile, center_rabi_freq, r_beam=None):
        self.r_profile = r_profile
        self.center_rabi_freq = center_rabi_freq
        self.r_beam = r_beam

    def get_rabi_freq(self, pos):
        """
        Rabi frequency at a position of a Gaussian beam.

        Parameters
        ----------
        pos : (n × 2) array or (n × 3) array
            positions of the n atoms in two or three dimensions (x, y, [z]).

        Returns
        -------
        rabi_freqs : array of float
            the Rabi frequencies for the n positions. If ``r_beam`` is set, the Rabi
            frequency will be set to 0 outside of the beam.
        """
        r_sq = pos[:, 0] ** 2 + pos[:, 1] ** 2
        rabi_freqs = self.center_rabi_freq * np.exp(-2 * r_sq / (self.r_profile**2))
        if self.r_beam is not None:
            rabi_freqs[r_sq > self.r_beam] = 0.0
        return rabi_freqs


class Wavefront:
    """
    Class that defines a wavefront.

    Parameters
    ----------
    r_wf : float
        radius of the wavefront data in m
    coeff : dict
        Dictionary of the Zernike coefficients. The keys are the Zernike polynomial
        single indices j and the values are the coefficients depends on the Zernike
        order and normalization.
    r_beam : float (optional)
        Beam radius in m. Can be set if the beam is smaller than the wavefront data.
        Values outside of the beam will be set to NaN.
    zern_order : ZernikeOrder or str
        Ordering scheme for the Zernike polynomials. See `ZernikeOrder` for possible
        values.
    zern_norm : ZernikeNorm, str or None
        Normalization scheme for the Zernike polynomials. See `ZernikeNorm` for possible
        values.

    Attributes
    ----------
    r_wf : float
        radius of the wavefront data in m
    coeff : dict
        Dictionary of the Zernike coefficients. The keys are the Zernike polynomial
        single indices j and the values are the coefficients depends on the Zernike
        order and normalization.
    r_beam : float or None
        Beam radius in m. If set, values outside of the beam will be set to NaN.
    zern_order : ZernikeOrder or str
        Ordering scheme for the Zernike polynomials.
    zern_norm : ZernikeNorm, str or None
        Normalization scheme for the Zernike polynomials.
    """

    def __init__(
        self,
        r_wf: float,
        coeff: dict[int, float],
        r_beam: float | None = None,
        zern_order: ZernikeOrder = ZernikeOrder.WYANT,
        zern_norm: ZernikeNorm | None = None,
    ) -> None:
        self.r_wf = r_wf
        self.coeff = coeff
        self.r_beam = r_beam
        self.zern_order = zern_order
        self.zern_norm = zern_norm

    def get_value(self, pos: np.ndarray) -> np.ndarray:
        """
        Get the wavefront at a position.

        Parameters
        ----------
        pos : n × 3 array
            array of position vectors (x, y, z) where the wavefront is probed

        Returns
        -------
        wf : nd array
            The value of the wavefront at the positions
        """
        pos = convert.cart2pol(pos)
        rho = pos[:, 0]
        rho = rho / self.r_wf
        rho[rho > 1] = np.nan
        theta = pos[:, 1]
        zern = ZernikePolynomial(self.coeff, self.zern_order, self.zern_norm)
        values = zern.zern_sum(rho, theta)
        if self.r_beam is not None:
            values[rho > self.r_beam / self.r_wf] = np.nan
        return values

    def plot(
        self,
        ax: plt.Axes | None = None,
        cmap: str | Colormap = "RdBu",
        levels: int = 100,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the wavefront data.

        Parameters
        ----------
        ax : Axis , optional
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.
        cmap : str or Colormap
            Colormap for the plot
        level : int
            Number of levels for the contour plot
        **kwargs
            Additional keyword arguments for the plot function

        Returns
        -------
        fig, ax : tuple of plt.Figure and plt.Axes
            The figure and axis of the plot
        """
        azimuths = np.radians(np.linspace(0, 360, 180))
        zeniths = np.linspace(0, self.r_wf, 50)
        rho, theta = np.meshgrid(zeniths, azimuths)
        z = np.zeros_like(rho)

        n_dim, m_dim = rho.shape
        pos = np.array([rho.flatten(), theta.flatten(), z.flatten()]).T
        values = self.get_value(convert.pol2cart(pos))

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        else:
            fig = ax.figure

        theta = theta.reshape(n_dim, m_dim)
        rho = rho.reshape(n_dim, m_dim)
        values = values.reshape(n_dim, m_dim)
        contour = ax.contourf(theta, rho, values, cmap=cmap, levels=levels, **kwargs)
        cbar = plt.colorbar(contour)
        cbar.set_label(r"Aberration / $\lambda$", rotation=90)
        plt.tight_layout()

        return fig, ax

    def plot_coeff(
        self, ax: plt.Axes | None = None, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the coefficients as a bar chart.

        Parameters
        ----------
        ax : Axis , optional
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.

        Returns
        -------
        fig, ax : tuple of plt.Figure and plt.Axes
            The figure and axis of the plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.bar(list(self.coeff.keys()), list(self.coeff.values()), **kwargs)
        ax.set_xlabel("Zernike polynomial $j$")
        ax.set_ylabel(r"Zernike coefficient $Z_j$ / $\lambda$")
        ax.set_xlim(min(self.coeff.keys()) - 1, max(self.coeff.keys()) + 1)
        return fig, ax


def gen_wavefront(
    r_wf: float,
    std: float = 0.0,
    r_beam: float | None = None,
    n_zern: int = 36,
    zern_order: ZernikeOrder = ZernikeOrder.NOLL,
    zern_norm: ZernikeNorm | None = None,
    seed: int | None = None,
) -> Wavefront:
    """
    Create an artificial wavefront.

    Parameters
    ----------
    r_wf : float
        radius of the wavefront data in m
    std : float
        standard deviation of each Zernike polynomial coefficient in multiples of the
        wavelength.
    r_beam : float, optional
        Beam radius in m. Can be set if the beam is smaller than the wavefront data.
        Values outside of the beam will be set to NaN.
    n_zern : int
        number of Zernike polynomials to be used
    zern_order : ZernikeOrder or str
        Ordering scheme for the Zernike polynomials. See `ZernikeOrder` for possible
        values.
    zern_norm : ZernikeNorm, str or None
        Normalization scheme for the Zernike polynomials. See `ZernikeNorm` for possible
        values.
    seed : int, optional
        seed for the random number generator

    Returns
    -------
    wf : Wavefront
        artificial wavefront
    """
    if seed is not None:
        np.random.seed(seed)
    coeff = np.random.normal(0, std, size=n_zern)
    zern_coeff = {
        j: val for j, val in enumerate(coeff, start=FIRST_INDEX_J[zern_order])
    }
    return Wavefront(r_wf, zern_coeff, r_beam, zern_order, zern_norm)
