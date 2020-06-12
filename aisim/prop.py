"""Classes to propagate atomic ensembles"""

import numpy as np
import copy

class Propagator():
    def __init__(self, time_delta, **kwargs):
        """
        Parameters
        ----------
        time_delta : float
            time that should be propagated
        **kwargs :
            Additional arguments used by classes that inherit from this class. All keyworded
            arguments are stored as attribues.
        """
        # FIXME: find a better name
        self.time_delta = time_delta

        # save all keyworded arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def prop_matrix(self, atoms):
        raise NotImplementedError()

    def propagate(self, atoms, **kwargs):
        atoms = copy.deepcopy(atoms)
        atoms.time += self.time_delta
        atoms.position += atoms.velocity * self.time_delta
        # U*|Psi>
        atoms.state_kets = np.einsum('ijk,ikl ->ijl', self.prop_matrix(atoms, **kwargs), atoms.state_kets )
        return atoms


class FreePropagator(Propagator):
    def prop_matrix(self, atoms):
        n_levels = atoms.state_kets[0].shape[0]
        return np.repeat([np.eye(n_levels)], repeats=len(atoms), axis=0)


class TwoLevelTransitionPropagator(Propagator):

    def __init__(self, time_delta, intensity_profile, wave_vectors=None, wf=None, phase_scan=0):
        """
        Implements an effective Raman two-level system as a time propagator as defined in [1].

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

        References
        ----------
        [1] Young, B. C., Kasevich, M., & Chu, S. (1997). Precision atom interferometry with light 
        pulses. In P. R. Berman (Ed.), Atom Interferometry (pp. 363–406). Academic Press. 
        https://doi.org/10.1016/B978-012092460-8/50010-2
        """
        super().__init__(time_delta, intensity_profile=intensity_profile, 
        wave_vectors=wave_vectors, wf=wf, phase_scan=phase_scan)

    def prop_matrix(self, atoms):
        # calculate the effective Rabi frequency at atoms' positions
        # pylint: disable=no-member
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

        sin_theta = Omega_eff/Omega_R
        cos_theta = -delta/Omega_R

        u_ee = np.cos(Omega_R * tau / 2) - 1j * \
            cos_theta * np.sin(Omega_R * tau / 2)
        u_ee *= np.exp(-1j * delta * tau/2)

        u_eg = np.exp(-1j * (delta*t0 + phase)) * -1j * \
            sin_theta * np.sin(Omega_R * tau / 2)
        u_eg *= np.exp(-1j * delta * tau/2)

        u_ge = np.exp(+1j * (delta*t0 + phase)) * -1j * \
            sin_theta * np.sin(Omega_R * tau / 2)
        u_ge *= np.exp(1j * delta * tau/2)

        u_gg = np.cos(Omega_R * tau / 2) + 1j * \
            cos_theta * np.sin(Omega_R * tau / 2)
        u_gg *= np.exp(1j * delta * tau/2)

        u = np.array([[u_ee, u_eg], [u_ge, u_gg]], dtype='complex')
        u = np.transpose(u, (2, 0, 1))
        return u

class SpatialSuperpositionTransitionPropagator(TwoLevelTransitionPropagator):

    def __init__(self, time_delta, n_pulses, intensity_profile=None, wave_vectors=None, wf=None, phase_scan=0):
        """
        Implements an effective Raman two-level system as a time propagator as defined in [1].
        In addition to class TwoLevelTransitionPropagator, this adds spatial superpositions
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
            number of intended light pulses in symmetric atom-interferometry sequennce.
            Each pulse adds two spatial eigenstates.

        References
        ----------
        [1] Young, B. C., Kasevich, M., & Chu, S. (1997). Precision atom interferometry with light 
        pulses. In P. R. Berman (Ed.), Atom Interferometry (pp. 363–406). Academic Press. 
        https://doi.org/10.1016/B978-012092460-8/50010-2
        """
        self.n_pulses = n_pulses
        super().__init__(time_delta, intensity_profile=intensity_profile, 
        wave_vectors=wave_vectors, wf=wf, phase_scan=phase_scan)
        
    def _block_diag(self, u, num):
        '''
        Fast generation of diagonal block matrices as described

        Parameters
        ----------
        u : numpy array
            Shape is n_atoms x n_int x n_int, were n_atoms is the
            number of atoms and n_int the number of internal states
        num : int
            number of times the internal propagation matrix is
            stacked into the diagonal block matrix
        '''
        n, rows, cols = u.shape
        result = np.zeros((n, num, rows, num, cols), dtype="complex")
        diag = np.einsum('hijik->hijk', result)
        for i in range(0, n):
            diag[i, :] = u[i]
        return result.reshape((n, rows * num, cols * num))


    def _twoLeveltransition(self, atoms):
        # Internal function for two-level transitions
        return super().prop_matrix(atoms)

    def prop_matrix(self, atoms):
        u_two_level = self._twoLeveltransition(atoms)
        u = self._block_diag(u_two_level, self.n_pulses)
        return u
    
class SpatialSuperpositionFreePropagator(Propagator):

    def __init__(self, time_delta, n_pulses):
        """
        Free propagator of SpatialSuperpositionTransitionPropagator class

        Parameters
        ----------
        time_delta: float
            length of pulse
        n_pulses : int
            number of intended light pulses in symmetric atom-interferometry sequennce.
            Each pulse adds two spatial eigenstates.
        """
        self.n_pulses = n_pulses
        self.time_delta = time_delta

    def prop_matrix(self, atoms):
        freeprop_matrix = np.zeros((2*self.n_pulses,2*self.n_pulses))
        for i in range(1,len(freeprop_matrix)):
            if i%2 == 1:
                freeprop_matrix[i,i] = 1
            else:
                freeprop_matrix[i-2,i] = 1
        return np.repeat(np.array([freeprop_matrix]), len(atoms), axis=0)