import numpy as np
import pytest  # noqa

import aisim as ais


def test_generic_detection():
    det = ais.Detector(0, a_keyworded_argument=1)
    assert det.a_keyworded_argument == 1  # test that attribute is saved

    with pytest.raises(NotImplementedError):
        det._detected_idx(0)


def test_spherical_detection():
    phase_space_vector = np.array([1, 1, 1, 0, 0, 0]).reshape(1, 6)
    atom = ais.AtomicEnsemble(phase_space_vector)
    det1 = ais.SphericalDetector(t_det=0, r_det=1)
    det2 = ais.SphericalDetector(t_det=0, r_det=np.sqrt(3))
    det3 = ais.SphericalDetector(t_det=0, r_det=2)
    det_atom1 = det1.detected_atoms(atom)
    det_atom2 = det2.detected_atoms(atom)
    det_atom3 = det3.detected_atoms(atom)

    assert len(det_atom1) == 0
    assert len(det_atom2) == 1
    assert len(det_atom3) == 1


def test_polar_detection():
    phase_space_vector = np.array([1, 1, 100, 0, 0, 0]).reshape(1, 6)
    atom = ais.AtomicEnsemble(phase_space_vector)
    det1 = ais.PolarDetector(t_det=0, r_det=1)
    det2 = ais.PolarDetector(t_det=0, r_det=np.sqrt(3))
    det3 = ais.PolarDetector(t_det=0, r_det=2)
    det_atom1 = det1.detected_atoms(atom)
    det_atom2 = det2.detected_atoms(atom)
    det_atom3 = det3.detected_atoms(atom)

    assert len(det_atom1) == 0
    assert len(det_atom2) == 1
    assert len(det_atom3) == 1
