import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa

import aisim as ais


def test_wavefront():
    # test the methods of Wavefront that are not covered by wf_test.py
    wf = ais.gen_wavefront(1)
    wf.plot()
    wf.plot_coeff()
    fig, ax = plt.subplots()
    wf.plot(ax=ax)
    fig, ax = plt.subplots()
    wf.plot_coeff(ax=ax)

    with pytest.raises(ValueError):
        rho = np.array([1.0, 1.1])
        theta = np.array([0, 0])
        ais.Wavefront.zern_iso(rho, theta, coeff=wf.coeff, r_beam=1)


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
    r_beam = 1
    pos1 = np.array([[0, 0, 0]])
    pos2 = np.array([[0, 1, 0]])
    intensity_profile = ais.IntensityProfile(r_beam, 1)

    rabi1 = intensity_profile.get_rabi_freq(pos1)
    rabi2 = intensity_profile.get_rabi_freq(pos2)
    ratio = rabi2 / rabi1
    # test that Rabi frequency has dropped to 1/e**2 of center value
    np.testing.assert_almost_equal(ratio[0], 1 / np.e**2)
