import pytest  # noqa
import numpy as np
import aisim as ais


def test_free_propagation():
    free_prop1 = ais.FreePropagator(time_delta=1)
    free_prop2 = ais.FreePropagator(time_delta=-1)

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
        'std_vz': ais.convert.vel_from_temp(0.2e-6),
    }
    atoms = ais.create_random_ensemble_from_gaussian_distribution(
        pos_params, vel_params, 100)

    prop_atoms1 = free_prop1.propagate(atoms)
    assert (prop_atoms1.state_kets == atoms.state_kets).all()
    with pytest.raises(AssertionError):
        # check that the positions changed
        np.testing.assert_array_almost_equal(
            prop_atoms1.position, atoms.position)
    # propagate back
    prop_atoms2 = free_prop2.propagate(prop_atoms1)
    np.testing.assert_array_almost_equal(prop_atoms2.position, atoms.position)


def test_two_level_transition_propagator():
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
        'std_vz': ais.convert.vel_from_temp(0.2e-6),
    }

    atoms = ais.create_random_ensemble_from_gaussian_distribution(
        pos_params, vel_params, 100)

    intensity_profile = ais.IntensityProfile(1, 1)

    prop = ais.TwoLevelTransitionPropagator(
        time_delta=1, intensity_profile=intensity_profile)

    matrices = prop._prop_matrix(atoms)

    for matrix in matrices:
        np.testing.assert_almost_equal(
            np.matmul(matrix, np.conjugate(matrix.T)), np.eye(2))
