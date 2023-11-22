import numpy as np
import pytest  # noqa

import aisim as ais

data = np.loadtxt("docs/examples/data/wf_grav_data.csv", skiprows=1, delimiter=",")
r_dets = data[:, 0]
grav_data = data[:, 1]

coeff_window = np.loadtxt("docs/examples/data/wf_window.txt")

wf = ais.Wavefront(11e-3, coeff_window)
wf.coeff[0:2] = 0  # remove piston, tip and tilt

pos_params = {
    "std_rho": 3.0e-3,  # cloud radius in m
    "std_z": 0,  # ignore z dimension, its not relevant here
    "n_rho": 20,  # within each standard deviation of the distribution
    "n_theta": 36,  # using a resolution of 10°
    "n_z": 1,  # use one value for the distribution along z
    "m_std_rho": 3,  # use 3 standard deviations of the distribution
    "m_std_z": 0,  # ignore z dimension, its not relevant here
}

vel_params = {
    # velocity spread in m/s from a temperature of 3 uK
    "std_rho": ais.vel_from_temp(3e-6),
    "std_z": 0,  # ignore z dimension, its not relevant here
    # within each standard deviation of the distribution we use 20 points
    "n_rho": 20,
    "n_theta": 36,  # using a resolution of 10°
    "n_z": 1,  # use one value for the distribution along z
    "m_std_rho": 3,  # use 3 standard deviations of the distribution
    "m_std_z": 0,  # ignore z dimension, its not relevant here
}

atoms = ais.create_ensemble_from_grids(pos_params, vel_params)

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

    # calculate the imprinted phase for each "test atom" at each pulse.
    # This is the computationally heavy part
    phi1 = 2 * np.pi * wf.get_value(det_atoms.calc_position(t1))
    phi2 = 2 * np.pi * wf.get_value(det_atoms.calc_position(t2))
    phi3 = 2 * np.pi * wf.get_value(det_atoms.calc_position(t3))

    # calculate a complex amplitude factor for the Mach-Zehnder sequence and
    # weight their contribution to the signal
    awf = np.exp(1j * (phi1 - 2 * phi2 + phi3))
    weighted_awf = np.sum(det_atoms.weights * awf) / np.sum(det_atoms.weights)

    awfs.append(weighted_awf)

grav = ais.phase_to_grav(np.angle(awfs), T=260e-3, keff=1.610574779769e6)

# test that simulation and data fit reasonably
assert np.sqrt(np.sum((grav - grav_data) ** 2)) < 1e-8
