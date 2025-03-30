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
        x: np.ndarray    # Best solution found
        nfev: int        # Number of function evaluations
        fun: float       # Function value at the best solution
        nit: int         # Number of iterations (metaepochs)


Detailed Usage
--------------

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

Now we need to configure the structure of our HMS tree by defining the optimization algorithms for each level. Each level configuration specifies the following:

1. The optimization algorithm to use (`EALevelConfig` which can run multiple different GAs)
2. The number of iterations per metaepoch (`generations`)
3. The problem to solve (can be different for each level e.g. less accurate for higher levels)
4. Population size and other algorithm-specific parameters
5. Local stop condition (`lsc`)

.. code-block:: python

    from pyhms import EALevelConfig, DontStop, SEA

    config = [
        EALevelConfig(
            ea_class=SEA,               # Use Simple Evolutionary Algorithm (GA)
            generations=2,              # Number of generations per metaepoch
            problem=square_problem,     # The problem to solve (problems can be different for each level)
            pop_size=20,                # Population size
            mutation_std=1.0,           # Standard deviation for mutation
            lsc=DontStop(),             # Local stop condition (never stop)
        ),
        EALevelConfig(
            ea_class=SEA,
            generations=4,              # More generations for deeper exploration
            problem=square_problem,
            pop_size=10,                # Smaller population size at lower levels
            mutation_std=0.25,          # Smaller mutations for local refinement
            sample_std_dev=1.0,         # Standard deviation for sampling around parent
            lsc=DontStop(),
        ),
    ]

The HMS algorithm creates a tree-like structure where demes (populations) at higher levels perform broad exploration, while demes at lower levels refine promising solutions. The configuration above defines two levels in our tree.

Next, we need to define a sprouting condition that determines when and where to create new demes at lower levels. We'll use Nearest Better Clustering (NBC) sprouting:

.. code-block:: python

    from pyhms import get_NBC_sprout
    sprout_condition = get_NBC_sprout(level_limit=4)

The NBC sprouting condition identifies promising points in the search space by clustering solutions based on their fitness and proximity. See :doc:`sprout` for more details on sprouting mechanisms.

Finally, we can run the algorithm:

.. code-block:: python

    from pyhms import hms
    hms_tree = hms(config, global_stop_condition, sprout_condition)
    print(f"Best fitness: {hms_tree.best_individual.fitness}")
