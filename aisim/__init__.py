"""Python package for simulating light-pulse atom interferometers."""

from importlib.metadata import PackageNotFoundError, version

from .atoms import *  # noqa
from .beam import *  # noqa
from .convert import *  # noqa
from .det import *  # noqa
from .prop import *  # noqa

try:
    __version__ = version("aisim")
except PackageNotFoundError:
    # package is not installed
    pass
