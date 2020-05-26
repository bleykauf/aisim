import numpy as np
from . import convert


def mz_interferometer(t, wf, atoms, det):
    """
    Simulation of the phase or gravity bias caused by wavefront aberrations in a Mach-Zehnder atom
    interferomter.

    Parameters
    ----------
    t : list of float
        times three interferometer pulses [t1, t2, t3] in s.
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
        and `np.angle`
    """

    # unpack the times of detection and the interferometer pulses
    t1, t2, t3 = t

    atoms = det.detected_atoms(atoms)

    # calculate the imprinted phase for each "test atom" at each pulse. This is the computationally
    #  heavy part
    phi1 = wf.get_value(atoms.position(t1))
    phi2 = wf.get_value(atoms.position(t2))
    phi3 = wf.get_value(atoms.position(t3))

    # calculate a complex amplitude factor for the Mach-Zehnder sequence and weight their
    # contribution to the signal
    awf = np.exp(1j * 2 * np.pi * (phi1 - 2*phi2 + phi3))
    weighted_awf = np.sum(atoms.weights * awf) / np.sum(atoms.weights)

    return weighted_awf


def ai_time_propagation(intensity_profile, atoms, state_vectors, t0, tau, wf=None):
    """
    Calculates the change of an array of initial wave functions in the effective Raman
    two-level system according to the time propagator U as in [1].

    Parameters
    ----------
    beam_profile : BeamProfile
        Intensity profile of the interferometry lasers
    atoms : AtomicEnsemble
        atomic ensemble that probes the wavefront
    t0 : float
        time of pulse
    tau: float
        length of pulse
    wf : Wavefront (optional)
        wavefront aberrations of the interferometry beam

    Returns
    -------
    psi : Array with n entries of two complex probabillity amplitudes

    References
    ----------
    [1] Young, B. C., Kasevich, M., & Chu, S. (1997). Precision atom interferometry with light 
    pulses. In P. R. Berman (Ed.), Atom Interferometry (pp. 363–406). Academic Press. 
    https://doi.org/10.1016/B978-012092460-8/50010-2
    """

     # calculate Rabi frequency at atoms' positions
    Omega_R = intensity_profile.get_rabi_freq(atoms.position(t0))

    if wf is None:
        phase = 0
    else:
        phase = wf.get_value(atoms.position(t0))  # calculate phase at atoms' positions

    # calculate matrix elements
    U_ee = np.cos(Omega_R * tau / 2)
    U_eg = np.multiply(np.exp(-1j * phase), 1j * np.sin(Omega_R * tau / 2))
    U_ge = np.multiply(np.exp(+1j * phase), 1j * np.sin(Omega_R * tau / 2))
    U_gg = np.cos(Omega_R * tau / 2)
    propagator = np.array([[U_ee, U_eg], [U_ge, U_gg]], dtype='complex')
    # U*psi
    # FIXME: Vectorize this
    new_state_vectors = state_vectors.propagate(propagator)
    return new_state_vectors
