from functools import partial

import numpy as np
import pytest  # noqa

import aisim as ais


def generate_random_atoms(n_atom, n_int):
    random_phase_space_vectors = np.random.rand(n_atom, 6)
    random_kets = np.random.rand(n_atom, n_int)
    # normalize random ket
    norm = np.einsum("ij,ij->i", random_kets, random_kets)
    norm_random_kets = np.einsum("ij,i->ij", random_kets, 1 / norm ** (1 / 2))
    # apply random phase shifts
    random_phases = 2 * np.pi * np.random.rand(n_atom, n_int)
    norm_random_kets = np.exp(1j * random_phases) * norm_random_kets
    norm_random_kets = np.reshape(norm_random_kets, (n_atom, n_int, 1))
    random_atoms = ais.AtomicEnsemble(random_phase_space_vectors, norm_random_kets)
    return random_atoms


def test_atomic_ensemble_methods():
    # This tests the AtomicEnsemble class' methods for consistency. It creates ensembles
    # of atoms with random internal and external states.
    # Methods of AtomicEnsemble that are tested:
    #     state_kets, state_bras, density_matrices, density_matrix
    def atomic_ensemble_test_function(atoms):
        # Testing AtomicEnsemble properties
        assert atoms.state_kets.shape == (n_atom, n_int, 1)
        assert atoms.state_bras.shape == (n_atom, 1, n_int)
        assert atoms.phase_space_vectors.shape == (n_atom, 6)
        assert atoms.phase_space_vectors.shape == (n_atom, 6)
        assert (atoms.phase_space_vectors[:, 0:3] == atoms.position).all()
        assert (atoms.calc_position(0) == atoms.position).all()
        assert (atoms.phase_space_vectors[:, 3:6] == atoms.velocity).all()
        # Test duality of Kets and Bras
        np.testing.assert_array_almost_equal(
            atoms.state_kets, np.conjugate(np.transpose(atoms.state_bras, (0, 2, 1)))
        )
        # Test whether the trace of every density matrix is equal to one
        np.testing.assert_almost_equal(
            np.trace(atoms.density_matrices, axis1=1, axis2=2), 1
        )
        # Test whether every single atoms density matrix satisfiesrho^2 = rho (condition
        # for pure states)
        np.testing.assert_array_almost_equal(
            np.matmul(atoms.density_matrices, atoms.density_matrices),
            atoms.density_matrices,
        )
        # test for the density matrix (mixed states)
        np.testing.assert_almost_equal(np.trace(atoms.density_matrix), 1)
        # test whether the sum of all state occupation is 1
        m_levels = np.arange(atoms.state_kets.shape[1])  # number of levels
        state_occupation = 0
        for m in m_levels:
            state_occupation += atoms.state_occupation(m)
        np.testing.assert_almost_equal(state_occupation, 1)
        # Test properties of the fidelity function
        # fidelity of density matrix with itself is 1
        np.testing.assert_almost_equal(1, atoms.fidelity(atoms.density_matrix))

    # Test AtomicEnsemble from very general randomly generated states
    n_atom = 1000
    n_ints = [1, 2, 3, 100]
    for n_int in n_ints:
        random_atoms = generate_random_atoms(n_atom, n_int)
        atomic_ensemble_test_function(random_atoms)

        # test slicing/selecting
        # selecting a single atom
        one_atom = random_atoms[0]
        assert len(one_atom) == 1
        if len(random_atoms) > 5:
            sliced_atoms = random_atoms[0:5]
            assert len(sliced_atoms) == 5

    # Create an ensemble of atoms
    atoms = ais.create_random_ensemble(
        n_atom,
        mean_x=1.0,
        mean_y=1.0,
        mean_z=1.0,
        x_dist=partial(ais.dist.position_dist_gaussian, std=1.0),
        y_dist=partial(ais.dist.position_dist_gaussian, std=1.0),
        z_dist=partial(ais.dist.position_dist_gaussian, std=1.0),
        vx_dist=partial(ais.dist.velocity_dist_from_temp, temperature=3.0e-6),
        vy_dist=partial(ais.dist.velocity_dist_from_temp, temperature=30.0e-6),
        vz_dist=partial(ais.dist.velocity_dist_from_temp, temperature=0.2e-6),
        seed=1,
        state_kets=random_atoms.state_kets,
    )

    atomic_ensemble_test_function(atoms)


def test_fidelity():
    for n_atom in [1, 3, 1000]:
        for n_int in [2, 3, 10, 50]:
            # Generate random density matrices
            rho_a = generate_random_atoms(n_atom, n_int).density_matrix
            rho_b = generate_random_atoms(n_atom, n_int).density_matrix
            # Test that return is float
            assert isinstance(ais.atoms._fidelity(rho_a, rho_b), float)
            # Test symmetry F(rhoA,rhoB) == F(rhoB,rhoA)
            np.testing.assert_almost_equal(
                ais.atoms._fidelity(rho_a, rho_b), ais.atoms._fidelity(rho_b, rho_a)
            )
            # Test if F(rhoA, rhoB) is equal or less than 1
            assert np.all(ais.atoms._fidelity(rho_a, rho_b) <= 1)
            # Test if F(rhoA, rhoA) is equal to 1
            np.testing.assert_array_almost_equal(ais.atoms._fidelity(rho_a, rho_a), 1)
