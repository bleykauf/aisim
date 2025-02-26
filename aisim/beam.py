"""Classes and functions related to the interferometry lasers."""

import matplotlib.pyplot as plt
import numpy as np

from . import convert


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

    def __init__(self, r_wf, coeff, r_beam=None):
        self.r_wf = r_wf
        self.coeff = coeff
        self.r_beam = r_beam

    def get_value(self, pos):
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
        values = self.zern_iso(rho, theta, coeff=self.coeff, r_wf=self.r_wf)
        if self.r_beam is not None:
            values[rho > self.r_beam] = np.nan
        return values

    def plot(self, ax=None):
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
        values = self.zern_iso(rho, theta, coeff=self.coeff, r_wf=self.r_wf)

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        else:
            fig = ax.figure

        contour = ax.contourf(theta, rho, values)
        cbar = plt.colorbar(contour)
        cbar.set_label(r"Aberration / $\lambda$", rotation=90)
        plt.tight_layout()

        return fig, ax

    def plot_coeff(self, ax=None):
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

    @staticmethod
    def zern_iso(rho, theta, coeff, r_wf):
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
        rho = rho / r_wf
        # Make sure rho outside of the beam are NaN
        rho[rho > 1] = np.nan
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


def gen_wavefront(r_wf, std=0, r_beam=None):
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
