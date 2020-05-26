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


def ai_timepropagation(wf, atoms, t0, tau, psi0):
    """
    Calculates the change of an array of initial wave functions in the effective Raman
    two-level system according to the timepropagator U as in Berman et. al.

    Parameters
    ----------
    wf : Wavefront
        wavefront aberrations and intensity distribution of the interferometry beam
    atoms : AtomicEnsemble
        atomic ensemble that probes the wavefront
    t0 : time of pulse
    tau: length of pulse
    psi0: Array of initial wave function (Prbabillity amplitude in ground and excited state)

    Returns
    -------
    psi_return : Array with n entries of two complex probabillity amplitudes
    """

    assert atoms.phase_space_vectors.shape[0] == psi0.shape[0], \
        "Number of atoms has to equal number of initial wave functions."
    psi_return = np.zeros(psi0.shape, dtype='complex')  # initizalize return array
    Omega_R = wf.get_rabi_freq(atoms.position(t0))  # calculate Rabi frequency at atoms' positions
    phi = wf.get_value(atoms.position(t0))  # calculate phase at atoms' positions
    # alculate matrix elements
    U_ee = np.cos(Omega_R*tau/2)
    U_eg = np.multiply(np.exp(-1j*phi), 1j*np.sin(Omega_R*tau/2))
    U_ge = np.multiply(np.exp(+1j*phi), 1j*np.sin(Omega_R*tau/2))
    U_gg = np.cos(Omega_R*tau/2)
    U = np.array([[U_ee, U_eg], [U_ge, U_gg]], dtype='complex')
    # U*psi
    for i in range(0, psi0.shape[0]):
        psi_return[i, :, :] = np.matmul(U[:, :, i], psi0[i, :, :].T).T
    return psi_return
