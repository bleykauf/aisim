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
    N_atom = 1000
    N_ints = [2, 3, 5, 10, 20, 100]
    for N_int in N_ints:
        # Generate random states
        random_phase_space_vectors = np.random.rand(N_atom, 6)
        random_kets = np.random.rand(N_atom, N_int)
        # normalize random ket
        norm = np.einsum('ij,ij->i', random_kets, random_kets)
        norm_random_kets = np.einsum('ij,i->ij', random_kets, 1/norm**(1/2))
        # apply random phase shifts
        random_phases = 2*np.pi*np.random.rand(N_atom, N_int)
        norm_random_kets = np.exp(1j*random_phases) * norm_random_kets
        norm_random_kets = np.reshape(norm_random_kets, (N_atom, N_int, 1))
        random_atoms = ais.AtomicEnsemble(
            random_phase_space_vectors, norm_random_kets)
        # Testing AtomicEnsemble properties
        # Test duality of Kets and Bras
        np.testing.assert_array_almost_equal(random_atoms.state_kets, np.conjugate(
            np.transpose(random_atoms.state_bras, (0, 2, 1))))
        # Test whether the trace of every density matrix is equal to one
        np.testing.assert_almost_equal(
            np.trace(random_atoms.density_matrices, axis1=1, axis2=2), 1)
        # Test whether every single atoms density matrix satisfies rho^2 = rho (condition for pure states)
        np.testing.assert_array_almost_equal(np.matmul(
            random_atoms.density_matrices, random_atoms.density_matrices), random_atoms.density_matrices)
