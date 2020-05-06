import numpy as np
from . import const

def rad_to_grav(phase, T=260, keff=1.610574779769):
    """
    Converts a phase shift to the corresponding effect of in $g$ in $nm/s^2$.
    
    Parameters
    ----------
    phase : float
        Interferometer phase in radian

    """
    return phase/keff/(T**2)* 1e9

def temp(sigma_v: float, species: str='Rb87') -> float:
    """
    Calculates the temperature of an atomic cloud from its velocity spread.

    :param sigma_v: velocity spread (1 sigma) in meters per second
    :param species: the atomic species, e.g. 'Rb8'
    :returns: temperature of the cloud in microkelvin
    """
    return 1e6 * const.mass[species] * sigma_v ** 2 / const.kb

def vel_from_temp(temp, species='Rb87'):
    """
    Calculates the velocity spread (1 sigma) from the temperature of the cloud.
    
    :param temp: temperature of the cloud in microkelvin
    :param species: the atomic species, e.g. 'Rb8'
    :returns: velocity spread (1 sigma) in meters per second
    """
    return np.sqrt(1e-6 * temp * const.kb / const.mass[species])