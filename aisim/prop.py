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
    atoms.position += atoms.velocity * dt
    return atoms


def transition(atoms, intensity_profile, tau, wave_vectors=None, wf=None, phase_scan=0):
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
    wave_vectors: Wavevectors
        wave vectors of the two Raman beams for calculation
        of Doppler shifts
    wf : Wavefront , optional
        wavefront aberrations of the interferometry beam
    phase_scan : float
        effective phase for fringe scans

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

    # calculate the effective Rabi frequency at atoms' positions
    Omega_eff = intensity_profile.get_rabi_freq(atoms.position)

    if wf is None:
        phase = 0
    else:
        # calculate phase at atoms' positions
        phase = wf.get_value(atoms.position)

    phase += phase_scan

    if wave_vectors is None:
        delta = 0
    else:
        # calculate two photon detuning for atoms' velocity (-v*k_eff)
        delta = wave_vectors.doppler_shift(atoms)

    # calculate Rabi frequency
    Omega_R = np.sqrt(Omega_eff**2 + delta**2)
    # beginning of pulse t0
    t0 = atoms.time

    # calculate matrix elements
    sin_theta = Omega_eff/Omega_R
    cos_theta = -delta/Omega_R

    U_ee = np.cos(Omega_R * tau / 2) - 1j * \
        cos_theta * np.sin(Omega_R * tau / 2)
    U_ee *= np.exp(-1j * delta * tau/2)

    U_eg = np.exp(-1j * (delta*t0 + phase)) * -1j * \
        sin_theta * np.sin(Omega_R * tau / 2)
    U_eg *= np.exp(-1j * delta * tau/2)

    U_ge = np.exp(+1j * (delta*t0 + phase)) * -1j * \
        sin_theta * np.sin(Omega_R * tau / 2)
    U_ge *= np.exp(1j * delta * tau/2)

    U_gg = np.cos(Omega_R * tau / 2) + 1j * \
        cos_theta * np.sin(Omega_R * tau / 2)
    U_gg *= np.exp(1j * delta * tau/2)

    propagator = np.array([[U_ee, U_eg], [U_ge, U_gg]], dtype='complex')
    propagator = np.transpose(propagator, (2, 0, 1))
    # U*|Psi>
    atoms.state_vectors = np.conjugate(np.einsum('ijk,ki ->ij', propagator, np.conjugate(atoms.state_vectors).T))
    atoms = free_evolution(atoms, tau)
    return atoms
