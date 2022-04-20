"""Functions to convert various quantities."""

import numpy as np

mass = {"Rb87": 1.443161e-25}
"""Atomic mass in kg."""

kb = 1.3806488e-23
"""Boltzmann constant in J/K."""


def temp(sigma_v, species="Rb87"):
    """
    Calculate the temperature of an atomic cloud from its velocity spread.

    Parameters
    ----------
    sigma_v : float
        velocity spread (1 sigma) in meters per second
    species : str
        the atomic species, has to be a key in `mass`

    Returns
    -------
    temp : float
        temperature of the cloud in Kelvin

    Raises
    ------
    ValueError
        If negative velocity spread is passed
    """
    if sigma_v < 0:
        raise ValueError("sigma_v must be non-negative")
    return mass[species] * sigma_v**2 / kb


def vel_from_temp(temp, species="Rb87"):
    """
    Calculate the velocity spread (1 sigma) from the temperature of the cloud.

    Parameters
    ----------
    temp : float
        temperature of the cloud in Kelvin
    species : str
        the atomic species, has to be a key in `mass`

    Returns
    -------
    vel : float
        velocity spread (1 sigma)

    Raises
    ------
    ValueError
        If negative temperature is passed
    """
    if temp < 0:
        raise ValueError("temp must be non-negative")
    return np.sqrt(temp * kb / mass[species])


def cart2pol(cart):
    """
    Convert vectors in cartesian coordinates to polar coordinates.

    Parameters
    ----------
    cart : n × 3 array
        array of n vectors in cartesian coordinates (x, y, z)

    Returns
    -------
    pol : n × 3 array
        array of n vectors in polar coordinates (rho, theta, z)
    """
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Transpose to retain form (n × 3 array)
    return np.array([rho, theta, z]).T


def pol2cart(pol):
    """
    Convert vectors in polar coordinates to cartesian coordinates.

    Parameters
    ----------
    pol : n × 3 array
        array of n vectors in polar coordinates (rho, theta, z)

    Returns
    -------
    cart : n × 3 array
        array of n vectors in cartesian coordinates (x, y, z)
    """
    rho = pol[:, 0]
    theta = pol[:, 1]
    z = pol[:, 2]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    # Transpose to retain form (n × 3 array)
    return np.array([x, y, z]).T


def phase_to_grav(phase, T, keff):
    """
    Convert a phase shift to gravitational acceleration.

    Takes the phase shift measured in a Mach Zehnder atom interferometer and converts it
    to the corresponding gravtional accleration.

    Parameters
    ----------
    phase : float
        Interferometer phase
    T : float
        interferometer time
    keff : float
        effective wavenumber

    Returns
    -------
    float :
        gravitational acceleration in m/s²
    """
    return phase / keff / (T**2)
