Usage
=====

.. _installation:

Installation
------------

To use pyhms, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyhms

.. _usage:

Quick start
-----------

The following example demonstrates how to use the pyhms library to perform optimization on a simple square function using the minimize method.

.. code-block:: python

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

The output of the function is a OptimizeResult object:

.. code-block:: python

    @dataclass
    class OptimizeResult:
        x: np.ndarray
        nfev: int
        fun: float
        nit: int


Usage
-----

Let's begin by defining a problem that we want to solve. We will use the following example:

.. code-block:: python

    import numpy as np
    from pyhms import FunctionProblem

    square_bounds = np.array([(-20, 20), (-20, 20)])
    square_problem = FunctionProblem(lambda x: sum(x**2), maximize=False, bounds=square_bounds)

Our problem is to minimize the sum of the squares of the elements of a vector. The vector has two elements, and each element is bounded between -20 and 20.

.. math::

    \min_{(x_1, x_2) \in [-20, 20]^2} (x_1^2 + x_2^2)


The solution to this problem is the vector [0, 0], which has a value of 0.
To use HMS we need to define global stop condition, in this case we want to run the algorithm for 10 iterations (called metaepochs).

.. code-block:: python

    from pyhms import MetaepochLimit
    global_stop_condition = MetaepochLimit(limit=10)

Now we need to decide what should be the height of our tree (maximum number of levels) and what optimization algorithms to run on each level. We will use the following configuration:

.. code-block:: python

    from pyhms import EALevelConfig, DontStop, SEA

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

Next step is to define sprout condition for our tree. We will use Nearest Better Clustering (NBC) sprout condition.

.. code-block:: python

    from pyhms import get_NBC_sprout
    sprout_condition = get_NBC_sprout(level_limit=4)

Finally we can run the algorithm:

.. code-block:: python

    from pyhms import hms
    hms_tree = hms(config, global_stop_condition, sprout_condition)
    print(f"Best fitness: {hms_tree.best_individual.fitness}")
