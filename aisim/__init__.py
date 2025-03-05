"""Python package for simulating light-pulse atom interferometers."""

from importlib.metadata import version

from .atoms import AtomicEnsemble, create_random_ensemble_from_gaussian_distribution
from .beam import IntensityProfile, Wavefront, Wavevectors, gen_wavefront
from .convert import cart2pol, phase_error_to_grav, pol2cart, temp, vel_from_temp
from .det import Detector, PolarDetector, SphericalDetector
from .prop import (
    FreePropagator,
    Propagator,
    SpatialSuperpositionTransitionPropagator,
    TwoLevelTransitionPropagator,
)
from .zern import ZernikeNorm, ZernikeOrder, ZernikePolynomial, j_to_n_m

__version__ = version("aisim")

__all__ = [
    "AtomicEnsemble",
    "create_random_ensemble_from_gaussian_distribution",
    "IntensityProfile",
    "Wavefront",
    "Wavevectors",
    "gen_wavefront",
    "cart2pol",
    "phase_error_to_grav",
    "pol2cart",
    "temp",
    "vel_from_temp",
    "Detector",
    "PolarDetector",
    "SphericalDetector",
    "FreePropagator",
    "Propagator",
    "SpatialSuperpositionTransitionPropagator",
    "TwoLevelTransitionPropagator",
    "ZernikeNorm",
    "ZernikeOrder",
    "ZernikePolynomial",
    "j_to_n_m",
]
