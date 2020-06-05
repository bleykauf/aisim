Developement
============

Documentation
-------------

For documentation, AISim uses `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`__ style 
docstrings which are rendered to html with `Sphinx <https://www.sphinx-doc.org/en/master/>`__ and 
the `napoleon extension <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`__.
Examples provided as `Jupyter <https://jupyter.org/>`__ notebooks are rendered with the 
`nbsphinx  <https://nbsphinx.readthedocs.io/en/0.7.0/>`__ extension. 

.. note::
    Symlinks are used to get relative paths to images and notebooks right when including the 
    `README.rst` in the sphinx documentation. Specifically, the :code:`docs/examples` is a symlink
    to :code:`examples`. Make sure your git installation handles symlinks correctly, especially 
    under Windows.

Building docs locally
^^^^^^^^^^^^^^^^^^^^^

Install :code:`sphinx`, :code:`sphinx_rtd_theme` and :code:`nbsphinx` for example using pip. Then 
run :code:`make html` from the `docs` folder. The html documentation can then be found in 
:code:`docs/_build/html`.

The :ref:`apiref` documentation is automatically generated with

:: 

    sphinx-apidoc --force --no-toc --module-first -o docs aisim
