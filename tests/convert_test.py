import numpy as np
import pytest

import aisim as ais


def test_mass():
    for mass in ais.convert.mass.values():
        assert mass > 0


def test_temp():
    for species in ais.convert.mass:
        assert ais.convert.temp(0, species=species) == 0
        assert ais.convert.temp(1.0, species=species) > 0
        with pytest.raises(ValueError):
            ais.convert.temp(-1.0, species=species)


def test_vel_from_temp():
    for species in ais.convert.mass:
        assert ais.convert.vel_from_temp(0, species=species) == 0
        assert ais.convert.vel_from_temp(1.0, species=species) > 0
        with pytest.raises(ValueError):
            ais.convert.vel_from_temp(-1.0, species=species)


def test_pol2cart_and_cart2pol():
    cart = np.array([[0, 0, 0], [1, 2, 3], [-1, -2, -3], [-10, 10, -10]])
    pol = ais.convert.cart2pol(cart)
    np.testing.assert_array_almost_equal(cart, ais.convert.pol2cart(pol))


def test_arrival_time():
    t12 = ais.convert.arrival_time(-1, 0, 0, 0, 1)
    assert t12[0] == -np.sqrt(2)
    assert t12[1] == np.sqrt(2)
