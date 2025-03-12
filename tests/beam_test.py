from functools import partial

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

    atoms = ais.create_random_ensemble(
        10,
        mean_x=1.0,
        mean_y=1.0,
        mean_z=1.0,
        mean_vx=1.0,
        mean_vy=1.0,
        mean_vz=1.0,
        x_dist=partial(ais.dist.position_dist_gaussian, std=1.0),
        y_dist=partial(ais.dist.position_dist_gaussian, std=1.0),
        z_dist=partial(ais.dist.position_dist_gaussian, std=3.0e-3),
        vx_dist=partial(ais.dist.velocity_dist_from_temp, temperature=3.0e-6),
        vy_dist=partial(ais.dist.velocity_dist_from_temp, temperature=3.0e-6),
        vz_dist=partial(
            ais.dist.velocity_dist_for_box_pulse_velsel, pulse_duration=100e-6
        ),
        seed=0,
    )

    diff = wv1.doppler_shift(atoms) + wv2.doppler_shift(atoms)
    assert diff.all() == 0


def test_intensity_profile():
    r_profile = 1
    pos1 = np.array([[0, 0, 0]])
    pos2 = np.array([[0, 1, 0]])
    intensity_profile = ais.IntensityProfile(r_profile=r_profile, center_rabi_freq=1)

    rabi1 = intensity_profile.get_rabi_freq(pos1)
    rabi2 = intensity_profile.get_rabi_freq(pos2)
    ratio = rabi2 / rabi1
    # test that Rabi frequency has dropped to 1/e**2 of center value
    np.testing.assert_almost_equal(ratio[0], 1 / np.e**2)

    # test that Rabi frequency is zero outside of the beam
    intensity_profile2 = ais.IntensityProfile(
        r_profile=r_profile, center_rabi_freq=1, r_beam=0.5
    )
    rabi3 = intensity_profile2.get_rabi_freq(pos1)
    assert rabi3[0] == 1
    rabi4 = intensity_profile2.get_rabi_freq(pos2)
    assert rabi4[0] == 0
