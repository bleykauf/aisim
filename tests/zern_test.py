import numpy as np
import pytest

from aisim.zern import ZernikeNorm, ZernikeOrder, j_to_n_m, zernike_term

# Explicit test cases from https://en.wikipedia.org/wiki/Zernike_polynomials
noll_mapping_examples = [
    (0, 0),
    (1, 1),
    (1, -1),
    (2, 0),
    (2, -2),
    (2, 2),
    (3, -1),
    (3, 1),
    (3, -3),
    (3, 3),
    (4, 0),
    (4, 2),
    (4, -2),
    (4, 4),
    (4, -4),
    (5, 1),
    (5, -1),
    (5, 3),
    (5, -3),
    (5, 5),
]

ansi_mapping_examples = [
    (0, 0),
    (1, -1),
    (1, 1),
    (2, -2),
    (2, 0),
    (2, 2),
    (3, -3),
    (3, -1),
    (3, 1),
    (3, 3),
    (4, -4),
    (4, -2),
    (4, 0),
    (4, 2),
    (4, 4),
    (5, -5),
    (5, -3),
    (5, -1),
    (5, 1),
    (5, 3),
]

wyant_mapping_examples = [
    (0, 0),
    (1, 1),
    (1, -1),
    (2, 0),
    (2, 2),
    (2, -2),
    (3, 1),
    (3, -1),
    (4, 0),
    (3, 3),
    (3, -3),
    (4, 2),
    (4, -2),
    (5, 1),
    (5, -1),
    (6, 0),
    (4, 4),
    (4, -4),
    (5, 3),
    (5, -3),
]


def test_j_to_n_m():
    for j, (n, m) in enumerate(noll_mapping_examples, start=1):
        assert j_to_n_m(j, ZernikeOrder.NOLL) == (n, m)

    for j, (n, m) in enumerate(ansi_mapping_examples):
        assert j_to_n_m(j, ZernikeOrder.ANSI) == (n, m)

    for j, (n, m) in enumerate(wyant_mapping_examples):
        assert j_to_n_m(j, ZernikeOrder.WYANT) == (n, m)

    for j, (n, m) in enumerate(wyant_mapping_examples, start=1):
        assert j_to_n_m(j, ZernikeOrder.FRINGE) == (n, m)

    with pytest.raises(ValueError):
        j_to_n_m(-1, "unknown")

    with pytest.raises(ValueError):
        j_to_n_m(10000, ZernikeOrder.WYANT)

    with pytest.raises(ValueError):
        j_to_n_m(0, ZernikeOrder.FRINGE)

    with pytest.raises(ValueError):
        j_to_n_m(-1, ZernikeOrder.ANSI)


def test_normalization():
    # Test for Example p. 110 of https://www.thorlabs.com/drawings/8b3619460cf44a9-01039B9C-E2C2-1948-D47C301CE6F71CB0/WFS40-5C-Manual.pdf
    defocus = -1e-6 * zernike_term(np.array([0, 1]), 0, n=2, m=0, norm=ZernikeNorm.OSA)
    assert np.round(1e6 * (defocus[0] - defocus[1]), 2) == 3.46
