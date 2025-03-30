# pyHMS
![GitHub Test Badge][1] [![codecov][2]](https://codecov.io/gh/agh-a2s/pyhms) [![Documentation Status][3]](https://pyhms.readthedocs.io/en/latest/?badge=latest) [![pypi.org][4]][5] [![versions][6]][7] ![license][8]

[1]: https://github.com/agh-a2s/pyhms/actions/workflows/pytest.yml/badge.svg "GitHub CI Badge"
[2]: https://codecov.io/gh/agh-a2s/pyhms/graph/badge.svg?token=srsivvv2ff
[3]: https://readthedocs.org/projects/pyhms/badge/?version=latest
[4]: https://img.shields.io/pypi/v/pyhms.svg "Pypi Latest Version Badge"
[5]: https://pypi.python.org/pypi/pyhms "Pypi site"
[6]: https://img.shields.io/pypi/pyversions/pyhms.svg
[7]: https://github.com/agh-a2s/pyhms
[8]: https://img.shields.io/github/license/agh-a2s/pyhms

`pyHMS` is a Python implementation of Hierarchic Memetic Strategy (HMS).

The Hierarchic Memetic Strategy is a stochastic global optimizer designed to tackle highly multimodal problems. It is a composite global optimization strategy consisting of a multi-population evolutionary strategy and some auxiliary methods. The HMS makes use of a dynamically-evolving data structure that provides an organization among the component populations. It is a tree with a fixed maximal height and variable internal node degree. Each component population is governed by a particular optimization engine. This package provides a simple python implementation.

### Installation
Installation can be done using `pypi`:
```
pip install pyhms
```
It's also possible to install the current main branch:
```
pip install git+https://github.com/agh-a2s/pyhms.git@main
```

### Quick Start

```python
from pyhms import minimize
import numpy as np

fun = lambda x: sum(x**2)
bounds = np.array([(-20, 20), (-20, 20)])
solution = minimize(
    fun=fun,
    bounds=bounds,
    maxfun=10000,
    log_level="debug",
    seed=42
)
```

`pyhms` provides an interface similar to `scipy.optimize.minimize`. This is the simplest way to run HMS with default parameters.

```python
import numpy as np
from pyhms import (
    EALevelConfig,
    hms,
    get_NBC_sprout,
    DontStop,
    MetaepochLimit,
    SEA,
    Problem,
)

square_bounds = np.array([(-20, 20), (-20, 20)])
square_problem = Problem(lambda x: sum(x**2), maximize=False, bounds=square_bounds)

config = [
    EALevelConfig(
        ea_class=SEA,
        generations=2,
        problem=square_problem,
        pop_size=20,
        mutation_std=1.0,
        lsc=DontStop(),
    ),
    EALevelConfig(
        ea_class=SEA,
        generations=4,
        problem=square_problem,
        pop_size=10,
        mutation_std=0.25,
        sample_std_dev=1.0,
        lsc=DontStop(),
    ),
]
global_stop_condition = MetaepochLimit(limit=10)
sprout_condition = get_NBC_sprout(level_limit=4)
hms_tree = hms(config, global_stop_condition, sprout_condition)
print(hms_tree.summary())
```

### Relevant literature

- J. Sawicki, M. Łoś, M. Smołka, R. Schaefer. Understanding measure-driven algorithms solving irreversibly ill-conditioned problems. Natural Computing 21:289-315, 2022. doi: [10.1007/s11047-020-09836-w](https://doi.org/10.1007/s11047-020-09836-w)
- J. Sawicki, M. Łoś, M. Smołka, J. Alvarez-Aramberri. Using Covariance Matrix Adaptation Evolutionary Strategy to boost the search accuracy in hierarchic memetic computations. Journal of computational science, 34, 48-54, 2019. doi: [10.1016/j.jocs.2019.04.005](https://doi.org/10.1016/j.jocs.2019.04.005)
