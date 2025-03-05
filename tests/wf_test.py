import json

import numpy as np

import aisim as ais


def test_wf():
    data = json.load(open("docs/examples/data/experimental_data.json"))
    r_dets = np.array(data["r_det"])
    grav_data = np.array(data["g"])

    coeff_window = {
        j: val for j, val in enumerate(np.loadtxt("docs/examples/data/wavefront.txt"))
    }
    wf = ais.Wavefront(
        10.91e-3, coeff_window, zern_norm=None, zern_order=ais.ZernikeOrder.WYANT
    )
    for n in [0, 1, 2]:
        wf.coeff[n] = 0  # remove piston, tip and tilt

    pos_params = {
        "mean_x": 0.0,
        "std_x": 3.0e-3,  # cloud radius in m
        "mean_y": 0.0,
        "std_y": 3.0e-3,  # cloud radius in m
        "mean_z": 0.0,
        "std_z": 0.0,  # ignore z dimension, its not relevant here
    }
    vel_params = {
        "mean_vx": 0.0,
        "std_vx": ais.convert.vel_from_temp(
            3.5e-6
        ),  # cloud velocity spread in m/s at tempearture of 3 uK
        "mean_vy": 0.0,
        "std_vy": ais.convert.vel_from_temp(
            3.5e-6
        ),  # cloud velocity spread in m/s at tempearture of 3 uK
        "mean_vz": 0.0,
        "std_vz": ais.convert.vel_from_temp(
            160e-9
        ),  # after velocity selection, velocity in z direction is 160 nK
    }

    atoms = ais.create_random_ensemble_from_gaussian_distribution(
        pos_params, vel_params, int(1e5), state_kets=[0, 1], seed=1
    )

    t_det = 778e-3  # time of the detection in s

    T = 260e-3  # interferometer time in s
    t1 = 130e-3  # time of first pulse in s
    t2 = t1 + T
    t3 = t2 + T

    awfs = []
    for r_det in r_dets:
        # creating detector with new detection radius
        det = ais.PolarDetector(t_det, r_det=r_det)

        det_atoms = det.detected_atoms(atoms)

        # calculate the imprinted phase for each "test atom" at each pulse. This is the
        # computationally heavy part
        phi1 = 2 * np.pi * wf.get_value(det_atoms.calc_position(t1))
        phi2 = 2 * np.pi * wf.get_value(det_atoms.calc_position(t2))
        phi3 = 2 * np.pi * wf.get_value(det_atoms.calc_position(t3))

        # calculate a complex amplitude factor for the Mach-Zehnder sequence and
        # weight their contribution to the signal
        awfs.append(np.nanmean(np.exp(1j * (phi1 - 2 * phi2 + phi3))))

    grav = 2 * ais.phase_error_to_grav(np.angle(awfs), T=260e-3, keff=1.610574779769e7)

    # test that simulation and data fit reasonably
    assert np.sqrt(np.sum((grav - grav_data) ** 2)) < 1.2e-8
