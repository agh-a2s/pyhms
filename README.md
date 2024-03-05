# pyhms

`pyhms` is a Python implementation of Hierarchic Memetic Strategy (HMS).

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
import numpy as np
from leap_ec.problem import FunctionProblem
from pyhms.config import EALevelConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.hms import hms
from pyhms.sprout import get_NBC_sprout
from pyhms.stop_conditions import DontStop, MetaepochLimit

square_problem = FunctionProblem(lambda x: sum(x**2), maximize=False)
square_bounds = np.array([(-20, 20), (-20, 20)])

config = [
    EALevelConfig(
        ea_class=SEA,
        generations=2,
        problem=square_problem,
        bounds=square_bounds,
        pop_size=20,
        mutation_std=1.0,
        lsc=DontStop(),
    ),
    EALevelConfig(
        ea_class=SEA,
        generations=4,
        problem=square_problem,
        bounds=square_bounds,
        pop_size=10,
        mutation_std=0.25,
        sample_std_dev=1.0,
        lsc=DontStop(),
    ),
]
global_stop_condition = MetaepochLimit(limit=10)
sprout_condition = get_NBC_sprout(level_limit=4)
hms_tree = hms(config, global_stop_condition, sprout_condition)
print(f"Best fitness: {hms_tree.best_individual.fitness}")
```

### Relevant literature

- J. Sawicki, M. Łoś, M. Smołka, R. Schaefer. Understanding measure-driven algorithms solving irreversibly ill-conditioned problems. Natural Computing 21:289-315, 2022. doi: [10.1007/s11047-020-09836-w](https://doi.org/10.1007/s11047-020-09836-w)
- J. Sawicki, M. Łoś, M. Smołka, J. Alvarez-Aramberri. Using Covariance Matrix Adaptation Evolutionary Strategy to boost the search accuracy in hierarchic memetic computations. Journal of computational science, 34, 48-54, 2019. doi: [10.1016/j.jocs.2019.04.005](https://doi.org/10.1016/j.jocs.2019.04.005)
