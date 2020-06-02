# `aisim` -- Simulations for light-pulse atom interferometry

## Installation

```
git clone https://github.com/bleykauf/aisim.git
cd aisim
python setup.py install
```

Alternatively, if you plan to make changes to the code, use

```
python setup.py develop
```

## Usage


```python
import aisim as ais
```

Checking the currently installed version:


```python
print(ais.__version__)
```

    v0.4.0
    

## Examples

Some examples are provided in the form of [Jupyter notebooks](https://jupyter.org/):

* [Effect of wavefront aberrations in atom interferometry](examples/wavefront-aberrations.ipynb)
* [Rabi oscillations with a Gaussian beam and thermal atoms](examples/rabi-oscillations.ipynb)

