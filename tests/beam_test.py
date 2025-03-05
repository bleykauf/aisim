import matplotlib.pyplot as plt
import numpy as np

import aisim as ais


def test_wavefront():
    # test the methods of Wavefront that are not covered by wf_test.py
    wf = ais.gen_wavefront(2, r_beam=1.0, seed=0)
    wf.plot()
    wf.plot_coeff()
    _, ax = plt.subplots()
    wf.plot(ax=ax)
    _, ax = plt.subplots()
    wf.plot_coeff(ax=ax)

    pos = np.array([[0, 0, 0], [1.1, 0, 0], [0, 2.1, 0]])
    res2 = wf.get_value(pos)
    assert not np.isnan(res2[0])
    assert np.isnan(res2[1])
    assert np.isnan(res2[2])


def test_wavevectors():
    wv1 = ais.Wavevectors(1, -1)
    wv2 = ais.Wavevectors(-1, 1)

    pos_params = {
        "mean_x": 1.0,
        "std_x": 1.0,
        "mean_y": 1.0,
        "std_y": 1.0,
        "mean_z": 1.0,
        "std_z": 1.0,
    }
    vel_params = {
        "mean_vx": 1.0,
        "std_vx": ais.convert.vel_from_temp(3.0e-6),
        "mean_vy": 1.0,
        "std_vy": ais.convert.vel_from_temp(3.0e-6),
        "mean_vz": 1.0,
        "std_vz": ais.convert.vel_from_temp(0.2e-6),
    }

    atoms = ais.create_random_ensemble_from_gaussian_distribution(
        pos_params, vel_params, n_samples=10, seed=0
    )
    diff = wv1.doppler_shift(atoms) + wv2.doppler_shift(atoms)
    assert diff.all() == 0


def test_intensity_profile():
    r_profile = 1
    pos1 = np.array([[0, 0, 0]])
    pos2 = np.array([[0, 1, 0]])
    intensity_profile = ais.IntensityProfile(r_profile, 1)

    rabi1 = intensity_profile.get_rabi_freq(pos1)
    rabi2 = intensity_profile.get_rabi_freq(pos2)
    ratio = rabi2 / rabi1
    # test that Rabi frequency has dropped to 1/e**2 of center value
    np.testing.assert_almost_equal(ratio[0], 1 / np.e**2)

    # test that Rabi frequency is zero outside of the beam
    intensity_profile2 = ais.IntensityProfile(r_profile, 1, r_beam=0.5)
    rabi3 = intensity_profile2.get_rabi_freq(pos1)
    assert rabi3[0] == 1
    rabi4 = intensity_profile2.get_rabi_freq(pos2)
    assert rabi4[0] == 0
