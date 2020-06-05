"""High-level functions for simulating certain aspects of atom interferometers."""

import numpy as np
from . import convert


def wavefront_simulation(t, wf, atoms, det):
    """
    Simulation of the phase or gravity bias caused by wavefront aberrations in a Mach-Zehnder atom
    interferomter.

    Parameters
    ----------
    t : list of float
        times of the three interferometer pulses [t1, t2, t3].
    wf : Wavefront
        wavefront aberrations of the interferometry beam
    atoms : AtomicEnsemble
        atomic ensemble that probes the wavefront
    det : Detector
        describes the detection zone

    Returns
    -------
    weighted_awf : float
        Complex amplitude factor from which contrast and phase shift can be determined via `abs()` 
        and `np.angle()`
    """

    # unpack the times of detection and the interferometer pulses
    t1, t2, t3 = t

    atoms = det.detected_atoms(atoms)

    # calculate the imprinted phase for each "test atom" at each pulse. This is the computationally
    #  heavy part
    phi1 = wf.get_value(atoms.calc_position(t1))
    phi2 = wf.get_value(atoms.calc_position(t2))
    phi3 = wf.get_value(atoms.calc_position(t3))

    # calculate a complex amplitude factor for the Mach-Zehnder sequence and weight their
    # contribution to the signal
    awf = np.exp(1j * 2 * np.pi * (phi1 - 2*phi2 + phi3))
    weighted_awf = np.sum(atoms.weights * awf) / np.sum(atoms.weights)

    return weighted_awf
