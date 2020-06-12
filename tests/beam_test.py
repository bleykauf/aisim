import pytest  # noqa
import aisim as ais


def test_wavefront():
    # test the methods of Wavefront that are not covered by wf_test.py
    wf = ais.gen_wavefront(1e-3)
    wf.plot()
    wf.plot_coeff()


def test_wavevectors():
    wv1 = ais.Wavevectors(1, -1)
    wv2 = ais.Wavevectors(-1, 1)

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

    atoms = ais.create_random_ensemble_from_gaussian_distribution(pos_params,
                                                                  vel_params,
                                                                  n_samples=10,
                                                                  seed=0)
    diff = wv1.doppler_shift(atoms) - wv2.doppler_shift(atoms)
    assert diff.all() == 0
