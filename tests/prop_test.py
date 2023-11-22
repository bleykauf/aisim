import numpy as np
import pytest  # noqa

import aisim as ais


def create_random_thermal_atoms(n_atoms, state_kets=[1, 0]):
    pos_params = {
        "mean_x": 0.0,
        "std_x": 3.0e-3,  # cloud radius in m
        "mean_y": 0.0,
        "std_y": 3.0e-3,  # cloud radius in m
        "mean_z": 0.0,
        "std_z": 3.0e-3,  # ignore z dimension, its not relevant here
    }
    vel_params = {
        "mean_vx": 0.0,
        # cloud velocity spread in m/s at tempearture of 3 uK
        "std_vx": ais.convert.vel_from_temp(3.0e-6),
        "mean_vy": 0.0,
        # cloud velocity spread in m/s at tempearture of 3 uK
        "std_vy": ais.convert.vel_from_temp(3.0e-6),
        "mean_vz": 0.0,
        "std_vz": ais.convert.vel_from_temp(280e-9),
    }
    return ais.create_random_ensemble_from_gaussian_distribution(
        pos_params, vel_params, n_atoms, state_kets=state_kets, seed=0
    )


def test_propagator():
    atoms = create_random_thermal_atoms(100)

    prop = ais.Propagator(1)
    with pytest.raises(NotImplementedError):
        prop._prop_matrix(atoms)


def test_free_propagator():
    free_prop1 = ais.FreePropagator(time_delta=1)
    free_prop2 = ais.FreePropagator(time_delta=-1)

    atoms = create_random_thermal_atoms(100)

    prop_atoms1 = free_prop1.propagate(atoms)
    assert (prop_atoms1.state_kets == atoms.state_kets).all()
    with pytest.raises(AssertionError):
        # check that the positions changed
        np.testing.assert_array_almost_equal(prop_atoms1.position, atoms.position)
    # propagate back
    prop_atoms2 = free_prop2.propagate(prop_atoms1)
    np.testing.assert_array_almost_equal(prop_atoms2.position, atoms.position)


def test_two_level_transition_propagator_unitarity():
    atoms = create_random_thermal_atoms(100)

    intensity_profile = ais.IntensityProfile(1, 1)
    wf = ais.gen_wavefront(10)
    wave_vectors = ais.Wavevectors()

    prop1 = ais.TwoLevelTransitionPropagator(
        time_delta=1, intensity_profile=intensity_profile
    )

    prop2 = ais.TwoLevelTransitionPropagator(
        time_delta=1,
        intensity_profile=intensity_profile,
        wf=wf,
        wave_vectors=wave_vectors,
        phase_scan=np.pi,
    )
    matrices = prop1._prop_matrix(atoms)

    for prop in [prop1, prop2]:
        matrices = prop._prop_matrix(atoms)
        for matrix in matrices:
            np.testing.assert_almost_equal(
                np.matmul(matrix, np.conjugate(matrix.T)), np.eye(2)
            )


def test_spatial_superposition_transition_propagator_unitarity():
    # define helper function
    def prop_unitarity_tester(n_pulses, pi_half_time):
        # create initial state ket [1,0,0,...,0] with length 2*n_pulses
        init_state = [1]
        for i in range(1, 2 * n_pulses):
            init_state.append(0)
            atoms = create_random_thermal_atoms(100, state_kets=init_state)

        intensity_profile = ais.IntensityProfile(1, 1)
        wf = ais.gen_wavefront(10)
        wave_vectors = ais.Wavevectors()

        for n_pulse in range(n_pulses):
            prop = ais.SpatialSuperpositionTransitionPropagator(
                1, intensity_profile, n_pulses, n_pulse + 1, wave_vectors, wf=wf
            )
            matrices = prop._prop_matrix(atoms)
            for matrix in matrices:
                multiplied = np.matmul(matrix, np.conjugate(matrix.T))
                np.testing.assert_almost_equal(multiplied, np.eye(2 * n_pulses))

    # test for different number of pulses and propagation times
    for pi_half_time in [1e-15, 13.5e-6, 1]:  # time of a pi/2 pulse
        for n_pulses in [1, 3, 10]:
            prop_unitarity_tester(n_pulses, pi_half_time)


def test_spatial_superposition_transition_propagator_time_reversal():
    # define helper function
    def prop_time_reversal_tester(n_pulses, pi_half_time):
        # create initial state ket [1,0,0,...,0] with length 2*n_pulses
        init_state = [1]
        for i in range(1, 2 * n_pulses):
            init_state.append(0)

        atoms = create_random_thermal_atoms(100, state_kets=init_state)
        atoms0 = atoms
        # create Raman beam
        r_beam = 29.5e-3 / 2  # 1/e^2 beam radius in m
        wave_vectors = ais.Wavevectors(k1=8.052945514e6, k2=-8.052802263e6)
        center_rabi_freq = 2 * np.pi / 4 / pi_half_time
        intensity_profile = ais.IntensityProfile(r_beam, center_rabi_freq)

        for n_pulse in range(0, n_pulses):
            propagator = ais.SpatialSuperpositionTransitionPropagator(
                pi_half_time, intensity_profile, n_pulses, n_pulse + 1, wave_vectors
            )
            atoms = propagator.propagate(atoms)
            # check if trace is one for atomic ensembles density matrix
            np.testing.assert_array_almost_equal(np.trace(atoms.density_matrix), 1)
            # check if rho^2 = rho for pure states
            np.testing.assert_array_almost_equal(
                np.matmul(atoms.density_matrices, atoms.density_matrices),
                atoms.density_matrices,
            )

        # check for time-reversibility (unitarity)
        for n_pulse in range(0, n_pulses):
            # change counting direction to descending, e.g. 3, 2, ...
            n_pulse = n_pulses - n_pulse
            propagator = ais.SpatialSuperpositionTransitionPropagator(
                -pi_half_time, intensity_profile, n_pulses, n_pulse, wave_vectors
            )
            atoms = propagator.propagate(atoms)
            # check if trace is one for atomic ensembles density matrix
            np.testing.assert_array_almost_equal(np.trace(atoms.density_matrix), 1)
            # check if rho^2 = rho for pure states
            np.testing.assert_array_almost_equal(
                np.matmul(atoms.density_matrices, atoms.density_matrices),
                atoms.density_matrices,
            )

        # check if final state is initial state
        np.testing.assert_array_almost_equal(
            atoms0.density_matrices, atoms.density_matrices
        )

    # test for different number of pulses and propagation times
    for pi_half_time in [1e-15, 13.5e-6, 1]:  # time of a pi/2 pulse
        for n_pulses in [1, 3, 10]:
            prop_time_reversal_tester(n_pulses, pi_half_time)
