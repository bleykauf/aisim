"""Classes and functions related to the detection system"""

import numpy as np
from . import convert


class Detector():
    """
    Parameters
    ----------
    r_det : float
        radius of the detection area in the $x$, $y$ plane
    """

    def __init__(self, r_det, t_det):
        self.r_det = r_det
        self.t_det = t_det

    def detected_idx(self, atoms):
        """
        Determines wheter a position is within the detection zone and returns the indices of the
        phase space vectors of an atomic ensemble that are detected

        Parameters
        ----------
        atoms : AtomicEnsemble
            the atomic ensemble

        Returns
        -------
        det_idx : nd array of bool
            boolean array for filtering an AtomicEnsemble; True if detected, otherwise False.

        """

        # pylint: disable=unsubscriptable-object
        rho = convert.cart2pol(atoms.calc_position(self.t_det))[:, 0]
        return np.where(rho <= self.r_det, True, False)

    def detected_atoms(self, atoms):
        """
        Determines wheter a position is within the detection zone and returns a new AtomicEnsemble
        object containing only the detected phase space vectors.

        Parameters
        ----------
        atoms : AtomicEnsemble
            the atomic ensemble

        Returns
        -------
        detected_atoms : AtomicEnsemble
            atomic ensemble containing only phase space vectors that are eventually detected
        """
        return atoms[self.detected_idx(atoms)]
