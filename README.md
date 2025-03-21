[![PyPI](https://img.shields.io/pypi/v/aisim?color=blue)](https://pypi.org/project/aisim/)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fbleykauf%2Faisim%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml)
![Test Status](https://github.com/bleykauf/aisim/actions/workflows/workflow.yml/badge.svg)
![Test Coverage](https://raw.githubusercontent.com/bleykauf/aisim/master/docs/coverage.svg)
[![Documentation Status](https://readthedocs.org/projects/aisim/badge/?version=latest)](https://aisim.readthedocs.io/en/latest/?badge=latest)

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

The latest tagged release can installed via pip with

    pip install aisim


Alternatively, if you plan to make changes to the code, use

    git clone https://github.com/bleykauf/aisim.git
    cd aisim
    pip install -e .

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

AISim ‒ Simulations for light-pulse atom interferometry

Copyright © 2020-2025 B. Leykauf

Copyright © 2020 S. Vowe

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
