[![PyPI version](https://badge.fury.io/py/aisim.svg)](https://badge.fury.io/py/aisim)
[![Build Status](https://travis-ci.com/bleykauf/aisim.svg?branch=master)](https://travis-ci.com/bleykauf/aisim)
[![Documentation Status](https://readthedocs.org/projects/aisim/badge/?version=latest)](https://aisim.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/bleykauf/aisim/badge.svg?branch=master)](https://coveralls.io/github/bleykauf/aisim?branch=master)

AISim ‒ Simulations for light-pulse atom interferometry
=======================================================

AISim is a Python package for simulating light-pulse atom
interferometers.

It uses dedicated objects to model the laser beams, the atomic ensemble
and the detection system and store experimental parameters in a neat
way. After you define these objects you can use built-in propagators to
simulate internal and external degrees of freedom of cold atoms.

Installation
------------

The latest tagged release can installed via pip:

    pip install aisim

Alternatively, if you plan to make changes to the code, use

    git clone https://github.com/bleykauf/aisim.git
    cd aisim
    python setup.py develop

Usage
-----

For basic usage and code reference, see the
[documentation](https://aisim.readthedocs.io).

Examples
--------

Some examples are provided in the form of [Jupyter
notebooks](https://jupyter.org/):

-   [Effect of wavefront aberrations in atom
    interferometry](https://github.com/bleykauf/aisim/blob/master/docs/examples/wavefront-aberrations.ipynb)
-   [Rabi oscillations with a Gaussian beam and thermal
    atoms](https://github.com/bleykauf/aisim/blob/master/docs/examples/rabi-oscillations.ipynb)
-   [Multiport atom interferometer](https://github.com/bleykauf/aisim/blob/master/docs/examples/multiport-ai.ipynb)

Contributing
------------

Contributions are very welcome. If you want to help, check out [our contributions guide](https://github.com/bleykauf/aisim/blob/master/docs/CONTRIBUTING.rst).

Authors
-------

-   Bastian Leykauf (<https://github.com/bleykauf>)
-   Sascha Vowe (<https://github.com/savowe>)

License
-------

AISim is licensed under [GPL
3.0](https://www.gnu.org/licenses/gpl-3.0.txt).
