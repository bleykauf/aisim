"""Functions to propagate atomic ensembles"""

import numpy as np
import copy


def free_evolution(atoms, dt):
    """
    Freely propagate an atomic ensemble for a certain time.

    Parameters
    ----------
    atoms : AtomicEnsemble
        the atomic ensemble to propagate
    dt : float
        duration of free propagation

    Returns
    atoms : AtomicEnsemble
        the initial atomic ensemble propagated by the timestep
    """

    atoms = copy.deepcopy(atoms)
    atoms.time += dt
    return atoms


def transition(atoms, intensity_profile, tau, wf=None):
    """
    Calculates the change of an array of initial wave functions in the effective Raman two-level 
    system according to the time propagator U as in [1].

    Parameters
    ----------
    atoms : AtomicEnsemble
        atomic ensemble that undergoes the transition
    beam_profile : IntensityProfile
        Intensity profile of the interferometry lasers
    tau: float
        length of pulse
    wf : Wavefront (optional)
        wavefront aberrations of the interferometry beam

    Returns
    -------
    atoms : AtomicEnsemble
        atomic ensemble with modified state vectorsand time

    References
    ----------
    [1] Young, B. C., Kasevich, M., & Chu, S. (1997). Precision atom interferometry with light 
    pulses. In P. R. Berman (Ed.), Atom Interferometry (pp. 363–406). Academic Press. 
    https://doi.org/10.1016/B978-012092460-8/50010-2
    """

    atoms = copy.deepcopy(atoms)

    # calculate Rabi frequency at atoms' positions
    Omega_R = intensity_profile.get_rabi_freq(atoms.position)

    if wf is None:
        phase = 0
    else:
        phase = wf.get_value(atoms.position)  # calculate phase at atoms' positions

    # calculate matrix elements
    U_ee = np.cos(Omega_R * tau / 2)
    U_eg = np.exp(-1j * phase) * 1j * np.sin(Omega_R * tau / 2)
    U_ge = np.exp(+1j * phase) * 1j * np.sin(Omega_R * tau / 2)
    U_gg = np.cos(Omega_R * tau / 2)
    propagator = np.array([[U_ee, U_eg], [U_ge, U_gg]], dtype='complex')
    propagator = np.transpose(propagator, (2, 0, 1))
    psi = np.zeros(atoms.state_vectors.shape, dtype='complex')  # initizalize return array

    # FIXME: Vectorize this
    for i in range(0, len(atoms)):
        # U*psi
        psi[i, :] = np.matmul(propagator[i][:, :], atoms.state_vectors[i][:].T).T
    atoms.state_vectors = psi
    atoms.time += tau
    return atoms
