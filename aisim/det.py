"""Classes and functions related to the detection system."""

import numpy as np

from . import convert


class Detector:
    """
    A generic detection zone.

    This is only a template without functionality. Deriving classes have to implement
    `_detected_idx`.

    Parameters
    ----------
    t_det : float
        time of the detection
    **kwargs :
        Additional arguments used by classes that inherit from this class. All keyworded
        arguments are stored as attribues.

    Attributes
    ----------
    t_det : float
        time of the detection
    ** kwargs :
        all keyworded arguments that are passed upon creation
    """

    def __init__(self, t_det, **kwargs):
        self.t_det = t_det
        # save all keyworded arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _detected_idx(self, atoms):
        """
        Return indices of the detected atoms.

        Determines wheter a position is within the detection zone and returns the
        indices of the phase space vectors of an atomic ensemble that are detected.

        Parameters
        ----------
        atoms : AtomicEnsemble
            the atomic ensemble

        Returns
        -------
        det_idx : nd array of bool
            boolean array for filtering an AtomicEnsemble; True if detected, otherwise
            False.
        """
        raise NotImplementedError()

    def detected_atoms(self, atoms):
        """
        Determine wheter a position is within the detection zone.

        Returns a new AtomicEnsemble object containing only the detected phase space
        vectors.

        Parameters
        ----------
        atoms : AtomicEnsemble
            the atomic ensemble

        Returns
        -------
        detected_atoms : AtomicEnsemble
            atomic ensemble containing only phase space vectors that are eventually
            detected
        """
        return atoms[self._detected_idx(atoms)]


class SphericalDetector(Detector):
    """
    A spherical detection zone.

    All atoms within a sphere of the specified radius will be detected.

    Parameters
    ----------
    t_det : float
        time of the detection
    r_det :
        radius of the spherical detection zone

    Attributes
    ----------
    t_det : float
        time of the detection
    ** kwargs :
        all keyworded arguments that are passed upon creation
    """

    def _detected_idx(self, atoms):
        det_pos = atoms.calc_position(self.t_det)
        x, y, z = det_pos[:, 0], det_pos[:, 1], det_pos[:, 2]
        rho = np.sqrt(x**2 + y**2 + z**2)
        return np.where(rho <= self.r_det, True, False)


class PolarDetector(Detector):
    """
    A spherical detection zone.

    All atoms within a circle with the specified radius within the x-y plane will be
    detected.

    Parameters
    ----------
    t_det : float
        time of the detection
    r_det :
        radius of the spherical detection zone

    Attributes
    ----------
    t_det : float
        time of the detection
    ** kwargs :
        all keyworded arguments that are passed upon creation
    """

    def _detected_idx(self, atoms):
        rho = convert.cart2pol(atoms.calc_position(self.t_det))[:, 0]
        return np.where(rho <= self.r_det, True, False)
