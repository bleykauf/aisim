Contributing
============

We welcome everybody to help by reporting bugs or making suggestions via our 
`issue tracker <https://github.com/bleykauf/aisim/issues>`__ or contribute code via  
`pull requests <https://github.com/bleykauf/aisim/pulls>`__ .

Code style
----------
We use the `black <https://black.readthedocs.io/en/stable/index.html>`__code style.

Testing
-------
Our tests are based on :code:`pytest`. To run the tests locally with your current environment, run 

::

    python -m pytest

from the root directory.

We use tox run via `Travis CI <https://travis-ci.com/github/bleykauf/aisim>`__  for testing for 
various python versions. New commits will automatically be tested and the results reported within
github. If all tests are succesful, coverage will be checked via 
`coveralls <https://coveralls.io/github/bleykauf/aisim>`__ for the latest python version.


Documentation
-------------

For documentation, AISim uses `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`__ style 
docstrings which are rendered to html with `Sphinx <https://www.sphinx-doc.org/en/master/>`__ and 
the `napoleon extension <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`__.
Examples provided as `Jupyter <https://jupyter.org/>`__ notebooks are rendered with the 
`nbsphinx  <https://nbsphinx.readthedocs.io/en/0.7.0/>`__ extension. The creation of the online 
documentation is done via `readthedocs  <https://readthedocs.org/>`__ and hosted 
`here  <https://aisim.readthedocs.io/>`__.


Building docs locally
^^^^^^^^^^^^^^^^^^^^^

Install :code:`sphinx`, :code:`sphinx_rtd_theme` and :code:`nbsphinx` for example using pip. Then 
run :code:`make html` from the `docs` folder. The html documentation can then be found in 
:code:`docs/_build/html`.

The :ref:`apiref` documentation is automatically generated with

:: 

    sphinx-apidoc --force --no-toc --module-first -o docs aisim