"""Classes to propagate atomic ensembles."""

import copy

import numpy as np


class Propagator:
    """
    A generic propagator.

    This is just a template class without an implemented propagation matrix.

    Parameters
    ----------
    time_delta : float
        time that should be propagated
    **kwargs :
        Additional arguments used by classes that inherit from this class. All
        keyworded arguments are stored as attribues.
    """

    def __init__(self, time_delta, **kwargs):
        self.time_delta = time_delta

        # save all keyworded arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _prop_matrix(self, atoms):
        raise NotImplementedError()

    def propagate(self, atoms):
        """
        Propagate an atomic ensemble.

        Parameters
        ----------
        atoms : AtomicEnsemble
            atomic ensemble that should be is propagated
        """
        atoms = copy.deepcopy(atoms)
        # propagate position for half a time step
        atoms.position += atoms.velocity * self.time_delta / 2
        # U*|Psi>
        atoms.state_kets = np.einsum(
            "ijk,ikl ->ijl", self._prop_matrix(atoms), atoms.state_kets
        )
        # propagate for another half time step
        atoms.position += atoms.velocity * self.time_delta / 2
        # update internal time
        atoms.time += self.time_delta
        return atoms


class FreePropagator(Propagator):
    """
    Propagator implementing free propagation without light-matter interaction.

    Parameters
    ----------
    time_delta : float
        time that should be propagated
    """

    def _prop_matrix(self, atoms):
        n_levels = atoms.state_kets[0].shape[0]
        return np.repeat([np.eye(n_levels)], repeats=len(atoms), axis=0)


class TwoLevelTransitionPropagator(Propagator):
    """
    A time propagator of an effective Raman two-level system.

    Parameters
    ----------
    time_delta: float
        length of pulse
    intensity_profile : IntensityProfile
        Intensity profile of the interferometry lasers
    wave_vectors: Wavevectors
        wave vectors of the two Raman beams for calculation of Doppler shifts
    wf : Wavefront , optional
        wavefront aberrations of the interferometry beam
    phase_scan : float
        effective phase for fringe scans

    Notes
    -----
    The propagator is for example defined in  [1].

    References
    ----------
    [1] Young, B. C., Kasevich, M., & Chu, S. (1997). Precision atom interferometry with
    light pulses. In P. R. Berman (Ed.), Atom Interferometry (pp. 363–406). Academic
    Press.
    https://doi.org/10.1016/B978-012092460-8/50010-2
    """

    def __init__(
        self, time_delta, intensity_profile, wave_vectors=None, wf=None, phase_scan=0
    ):
        super().__init__(
            time_delta,
            intensity_profile=intensity_profile,
            wave_vectors=wave_vectors,
            wf=wf,
            phase_scan=phase_scan,
        )

    def _prop_matrix(self, atoms):
        # calculate the effective Rabi frequency at atoms' positions
        Omega_eff = self.intensity_profile.get_rabi_freq(atoms.position)
        if self.wf is None:
            phase = 0
        else:
            # calculate phase at atoms' positions
            phase = self.wf.get_value(atoms.position)

        phase += self.phase_scan

        if self.wave_vectors is None:
            delta = 0
        else:
            # calculate two photon detuning for atoms' velocity (-v*k_eff)
            delta = self.wave_vectors.doppler_shift(atoms)

        # calculate Rabi frequency
        Omega_R = np.sqrt(Omega_eff**2 + delta**2)
        # beginning of pulse t0
        t0 = atoms.time
        tau = self.time_delta

        # calculate matrix elements

        sin_theta = Omega_eff / Omega_R
        cos_theta = -delta / Omega_R

        u_ee = np.cos(Omega_R * tau / 2) - 1j * cos_theta * np.sin(Omega_R * tau / 2)
        u_ee *= np.exp(-1j * delta * tau / 2)

        u_eg = (
            np.exp(-1j * (delta * t0 + phase))
            * -1j
            * sin_theta
            * np.sin(Omega_R * tau / 2)
        )
        u_eg *= np.exp(-1j * delta * tau / 2)

        u_ge = (
            np.exp(+1j * (delta * t0 + phase))
            * -1j
            * sin_theta
            * np.sin(Omega_R * tau / 2)
        )
        u_ge *= np.exp(1j * delta * tau / 2)

        u_gg = np.cos(Omega_R * tau / 2) + 1j * cos_theta * np.sin(Omega_R * tau / 2)
        u_gg *= np.exp(1j * delta * tau / 2)

        u = np.array([[u_ee, u_eg], [u_ge, u_gg]], dtype="complex")
        u = np.transpose(u, (2, 0, 1))
        return u


class SpatialSuperpositionTransitionPropagator(TwoLevelTransitionPropagator):
    """
    An effective Raman two-level system.

    It is implemented as a time propagator as defined in [1]. In addition to
    class TwoLevelTransitionPropagator, this adds spatial superpositions
    in z-direction that occour at each pulse.

    Parameters
    ----------
    time_delta: float
        length of pulse
    intensity_profile : IntensityProfile
        Intensity profile of the interferometry lasers
    wave_vectors: Wavevectors
        wave vectors of the two Raman beams for calculation of Doppler shifts
    wf : Wavefront , optional
        wavefront aberrations of the interferometry beam
    phase_scan : float
        effective phase for fringe scans
    n_pulses : int
        overall number of intended light pulses in symmetric atom
        interferometry sequence. Each pulse adds two spatial eigenstates.
    n_pulse : int
        number of light pulse of symmetric atom-interferometry sequence.

    References
    ----------
    [1] Young, B. C., Kasevich, M., & Chu, S. (1997). Precision atom interferometry with
    light pulses. In P. R. Berman (Ed.), Atom Interferometry (pp. 363–406). Academic
    Press.
    https://doi.org/10.1016/B978-012092460-8/50010-2
    """

    def __init__(
        self,
        time_delta,
        intensity_profile,
        n_pulses,
        n_pulse,
        wave_vectors=None,
        wf=None,
        phase_scan=0,
    ):
        self.n_pulses = n_pulses
        self.n_pulse = n_pulse
        super().__init__(
            time_delta,
            intensity_profile=intensity_profile,
            wave_vectors=wave_vectors,
            wf=wf,
            phase_scan=phase_scan,
        )

    def _block_diag(self, u, num):
        """
        Fast generation of diagonal block matrices of internal state dynamics.

        Parameters
        ----------
        u : n × m × m array
            propagation matrices for the n atoms with m levels
        num : int
            number of times the internal propagation matrix is repeated in the diagonal
            block matrix

        Returns
        -------
        matrix : n × m*num × m*num
            block matrix mediating the interactions of the n atoms with m*num levels
        """
        # since u is a square matrix, row = cols = n_states
        n, m, m = u.shape
        matrix = np.zeros((n, num, m, num, m), dtype="complex")
        # Note that diag is a view of matrix, so matrix is changed. n is index for
        # atoms, m and k for the levels and i for num
        diag = np.einsum("nimik->nimk", matrix)
        for i in range(0, n):
            # diag[i, :] has shape (num, n_int, n_int), u[i] has shape (n_int, n_int),
            # i.e. so u[i] is copied num times in each loop
            diag[i, :] = u[i]
        # bring matrix into final form
        return matrix.reshape((n, m * num, m * num))

    def _index_shift(self):
        """
        Switch state indices to ensure that the correct states interact.

        Returns
        -------
        index_shift_matrix : 2*n_pulses × 2*n_pulses array
        """
        # Each pulse splits the initial state into two states (2 level system),
        # so we end up with 2*n_pulses output states
        index_shift_matrix = np.eye(2 * self.n_pulses)
        for i in range(0, len(index_shift_matrix)):
            if i % 2 == 0:
                # shift the index of one of the two interacting states;
                # applying this matrix n_pulses times results again in a unity matrx
                index_shift_matrix[i, :] = np.roll(index_shift_matrix[i, :], 2)
        return index_shift_matrix

    def _prop_matrix(self, atoms):
        assert (
            atoms.state_kets.shape[1] == 2 * self.n_pulses
        ), "Number of states must be twice the number of pulses."

        u_two_level = super()._prop_matrix(atoms)
        u = self._block_diag(u_two_level, self.n_pulses)
        # Create two index shift matrices. First, positions of the state vectors are
        # switched such that the two interacting states are next to each other and are
        # coupled by the blocks of _block_diag. After the application of the propagation
        # matrix, these states are put back into their initial position.
        shift_forth = np.linalg.matrix_power(self._index_shift(), self.n_pulse)
        shift_back = np.linalg.matrix_power(
            self._index_shift(), self.n_pulses - self.n_pulse
        )
        return np.einsum("ij,njk,kl->nil", shift_back, u, shift_forth)
