from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import atoms, beam, convert, det, sims
from .atoms import *
from .beam import *
from .convert import  *
from .det import *
from .sims import *
