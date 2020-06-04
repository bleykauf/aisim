"""aisim is a python package for simulating light-pulse atom interferometers"""

from .sims import *
from .prop import *
from .det import *
from .convert import *
from .beam import *
from .atoms import *
from . import atoms, beam, convert, det, sims, prop
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
