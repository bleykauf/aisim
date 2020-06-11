import pytest
import aisim as ais
import numpy as np

def test_AtomicEnsemble_methods():
    '''
    This tests the AtomicEnsemble class' methods for
    consistency.
    It creates ensembles of atoms with random internal
    and external states.
    Methods of AtomicEnsemble that are tested:
        state_kets, state_bras, density_matrices,
        density_matrix
    '''
    def atomic_ensemble_test_function(atoms):
        # Testing AtomicEnsemble properties
        assert atoms.state_kets.shape == (n_atom, n_int, 1)
        assert atoms.state_bras.shape == (n_atom, 1, n_int)
        assert atoms.phase_space_vectors.shape == (n_atom, 6)
        assert atoms.phase_space_vectors.shape == (n_atom, 6)
        assert (atoms.phase_space_vectors[:,0:3] == atoms.position).all()
        assert (atoms.calc_position(0) == atoms.position).all()
        assert (atoms.phase_space_vectors[:,3:6] == atoms.velocity).all()
        # Test duality of Kets and Bras
        np.testing.assert_array_almost_equal(atoms.state_kets, np.conjugate(
            np.transpose(atoms.state_bras, (0, 2, 1))))
        # Test whether the trace of every density matrix is equal to one
        np.testing.assert_almost_equal(
            np.trace(atoms.density_matrices, axis1=1, axis2=2), 1)
        # Test whether every single atoms density matrix satisfies rho^2 = rho (condition for pure states)
        np.testing.assert_array_almost_equal(np.matmul(
            atoms.density_matrices, atoms.density_matrices), atoms.density_matrices)

    # Test AtomicEnsemble from very general randomly generated states
    n_atom = 1000
    n_ints = [2, 3, 5, 10, 20, 100]
    for n_int in n_ints:
        random_phase_space_vectors = np.random.rand(n_atom, 6)
        random_kets = np.random.rand(n_atom, n_int)
        # normalize random ket
        norm = np.einsum('ij,ij->i', random_kets, random_kets)
        norm_random_kets = np.einsum('ij,i->ij', random_kets, 1/norm**(1/2))
        # apply random phase shifts
        random_phases = 2*np.pi*np.random.rand(n_atom, n_int)
        norm_random_kets = np.exp(1j*random_phases) * norm_random_kets
        norm_random_kets = np.reshape(norm_random_kets, (n_atom, n_int, 1))
        random_atoms = ais.AtomicEnsemble(
            random_phase_space_vectors, norm_random_kets)
        atomic_ensemble_test_function(random_atoms)

    # Test AtomicEnsemble from create_random_ensemble_from_gaussian_distribution method
    pos_params = {
        'mean_x': 1.0,
        'std_x': 1.0,
        'mean_y': 1.0,
        'std_y': 1.0,
        'mean_z': 1.0,
        'std_z': 1.0,
    }
    vel_params = {
        'mean_vx': 1.0,
        'std_vx': ais.convert.vel_from_temp(3.0e-6),
        'mean_vy': 1.0,
        'std_vy': ais.convert.vel_from_temp(3.0e-6),
        'mean_vz': 1.0,
        'std_vz': ais.convert.vel_from_temp(0*.2e-6),
    }
    atoms = ais.create_random_ensemble_from_gaussian_distribution(
        pos_params, vel_params, n_atom, state_kets=norm_random_kets )
    atomic_ensemble_test_function(atoms)
