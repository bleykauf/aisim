"""Classes and functions related to the interferometry lasers."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from . import convert
from .zern import zern_iso_naive


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
        dopler_shift : 1d array
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
    coeff : list of float
        list of 36 Zernike coefficients in multiples of the wavelength
    r_beam : float (optional)
        Beam radius in m. Can be set if the beam is smaller than the wavefront data.
        Values outside of the beam will be set to NaN.

    Attributes
    ----------
    r_wf : float
        radius of the wavefront data in m
    coeff : list of float
        list of 36 Zernike coefficients in multiples of the wavelength
    r_beam : float or None
        Beam radius in m. If set, values outside of the beam will be set to NaN.
    """

    def __init__(
        self, r_wf: float, coeff: ArrayLike, r_beam: float | None = None
    ) -> None:
        self.r_wf = r_wf
        self.coeff = np.array(coeff)
        self.r_beam = r_beam

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
        theta = pos[:, 1]
        values = zern_iso_naive(rho, theta, coeff=self.coeff, r_wf=self.r_wf)
        if self.r_beam is not None:
            values[rho > self.r_beam] = np.nan
        return values

    def plot(self, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the wavefront data.

        Parameters
        ----------
        ax : Axis , optional
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.

        """
        azimuths = np.radians(np.linspace(0, 360, 180))
        zeniths = np.linspace(0, self.r_wf, 50)
        rho, theta = np.meshgrid(zeniths, azimuths)
        n_dim, m_dim = rho.shape
        z = np.zeros_like(rho)
        pos = np.array([rho.flatten(), theta.flatten(), z.flatten()]).T
        values = self.get_value(convert.pol2cart(pos))

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        else:
            fig = ax.figure

        theta = theta.reshape(n_dim, m_dim)
        rho = rho.reshape(n_dim, m_dim)
        values = values.reshape(n_dim, m_dim)
        contour = ax.contourf(theta, rho, values)
        cbar = plt.colorbar(contour)
        cbar.set_label(r"Aberration / $\lambda$", rotation=90)
        plt.tight_layout()

        return fig, ax

    def plot_coeff(self, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the coefficients as a bar chart.

        Parameters
        ----------
        ax : Axis , optional
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.bar(np.arange(len(self.coeff)), self.coeff)
        ax.set_xlabel("Zernike polynomial $i$")
        ax.set_ylabel(r"Zernike coefficient $Z_i$ / $\lambda$")
        return fig, ax


def gen_wavefront(
    r_wf: float, std: float = 0.0, r_beam: float | None = None
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

    Returns
    -------
    wf : Wavefront
        artificial wavefront
    """
    coeff = np.random.normal(0, std, size=36)
    return Wavefront(r_wf, coeff, r_beam)
