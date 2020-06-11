import pytest
import aisim as ais


def test_mass():
    for mass in ais.mass.values():
        assert mass > 0


def test_temp():
    for species in ais.mass:
        assert ais.temp(0, species=species) == 0
        assert ais.temp(1.0, species=species) > 0
        with pytest.raises(ValueError):
            ais.temp(-1.0, species=species)


def test_vel_from_temp():
    for species in ais.mass:
        assert ais.vel_from_temp(0, species=species) == 0
        assert ais.vel_from_temp(1.0, species=species) > 0
        with pytest.raises(ValueError):
            ais.vel_from_temp(-1.0, species=species)
