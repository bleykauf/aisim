"""Module for functions related to atomic position and velocity distributions."""

import numpy as np
from scipy.interpolate import interp1d

from .convert import vel_from_temp


def position_dist_gaussian(n: int, std: float) -> np.ndarray:
    """Create a random position distribution from a Gaussian distribution.

    Parameters
    ----------
    n : int
        number of samples
    std : float
        standard deviation of the Gaussian distribution in meters
    """
    return np.random.normal(scale=std, size=n)


def velocity_dist_from_temp(n: int, temperature: float) -> np.ndarray:
    """Create a random velocity distribution from a given temperature.

    Parameters
    ----------
    n : int
        number of samples
    temperature : float
        temperature of the atomic ensemble in Kelvin

    Returns
    -------
    velocities : array
        n-dimensional array of the randomly selected velocities
    """
    return np.random.normal(scale=vel_from_temp(temperature), size=n)


def velocity_dist_for_box_pulse_velsel(
    n: int, pulse_duration: float, wavelenth: float = 780e-9, n_lobes: int = 3
) -> np.ndarray:
    """Create a random velocity distribution for a box pulse velocity selection.

    The velocity distribution is created by sampling the inverse cumulative distribution
    function of the Fourier transform of a box pulse with a uniform distribution. The
    box pulse is defined by its duration and the velocity distribution is given by
    |τ sinc(2*τ*v/λ)|^2 where τ is the pulse duration, λ the wavelength and v is the
    velocity of the atoms.

    Parameters
    ----------
    n : int
        number of samples
    pulse_duration : float
        duration of the box pulse in seconds
    wavelenth : float, optional
        wavelength of the light in meters (default 780e-9)
    n_lobes : int, optional
        number of lobes of the sinc function that are sampled (default 3)

    Returns
    -------
    velocities : array
        n-dimensional array of the randomly selected velocities
    """

    def box_pulse_ft(x: np.ndarray, pulse_duration: float) -> np.ndarray:
        """Fourier transform of a box pulse."""
        return np.abs(pulse_duration * np.sinc(pulse_duration * x) ** 2)

    # Based on https://stackoverflow.com/a/64288861/2750945
    x = np.linspace(-n_lobes / pulse_duration, n_lobes / pulse_duration, int(1e5))
    y_cumsum = np.cumsum(box_pulse_ft(x, pulse_duration))
    y_cumsum = y_cumsum - y_cumsum.min()
    f = interp1d(y_cumsum / y_cumsum.max(), x)

    return wavelenth * f(np.random.random(n)) / 2


def velocity_dist_for_gaussian_velsel(
    n: int, pulse_duration: float, wavelength: float = 780e-9
):
    """Create a random velocity distribution for a Gaussian pulse velocity selection.

    Parameters
    ----------
    n : int
        number of samples
    pulse_duration : float
        duration of the Gaussian pulse in seconds (1/sqrt(2) full width, i.e. 2*sigma)
    wavelength : float, optional
        wavelength of the light in meters (default 780e-9)

    Returns
    -------
    velocities : array
        n-dimensional array of the randomly selected velocities
    """

    pulse_duration = pulse_duration / 2  # Convert to sigma
    sigma_nu = 1 / (2 * np.pi * pulse_duration)

    return np.random.normal(scale=sigma_nu, size=n) * wavelength / 2
