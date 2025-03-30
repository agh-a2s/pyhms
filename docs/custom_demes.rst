Adding Custom Demes to pyHMS
============================

This guide explains how to create your own custom deme implementations for pyHMS.

Overview
--------

pyHMS allows you to extend the system with your own custom deme implementations. To create a custom deme, you need to:

1. Define a new config class that inherits from ``BaseLevelConfig``
2. Create a new deme class that inherits from ``AbstractDeme``
3. Register your custom deme by passing a ``config_class_to_deme_class`` mapping to the ``hms`` function

Step 1: Define Your Config Class
--------------------------------

Start by creating a config class that inherits from ``BaseLevelConfig``. This class should:

- Accept a ``problem`` and a stop condition (``lsc``) as required parameters
- Include any additional parameters your deme implementation needs
- Call the parent class's ``__init__`` method

.. code-block:: python

    from pyhms.config import BaseLevelConfig
    from pyhms.core.problem import Problem
    from pyhms.stop_conditions import LocalStopCondition, UniversalStopCondition

    class RandomSearchConfig(BaseLevelConfig):
        def __init__(
            self,
            problem: Problem,
            lsc: LocalStopCondition | UniversalStopCondition,
            pop_size: int,
        ) -> None:
            super().__init__(problem, lsc)
            self.pop_size = pop_size

Step 2: Create Your Deme Class
------------------------------

Next, create a deme class that inherits from ``AbstractDeme``. This class must implement the required interface:

.. code-block:: python

    import numpy as np
    from pyhms.core.individual import Individual
    from pyhms.demes.abstract_deme import AbstractDeme, DemeInitArgs

    class RandomSearchDeme(AbstractDeme):
        def __init__(
            self,
            deme_init_args: DemeInitArgs,
        ) -> None:
            super().__init__(deme_init_args)
            config: RandomSearchConfig = deme_init_args.config  # type: ignore[assignment]
            self._pop_size = config.pop_size
            self.lower_bounds = config.bounds[:, 0]
            self.upper_bounds = config.bounds[:, 1]
            self.rng = np.random.RandomState(deme_init_args.random_seed)
            self.run()

        def run(self) -> None:
            genomes = np.random.uniform(
                self.lower_bounds,
                self.upper_bounds,
                size=(self._pop_size, len(self.lower_bounds))
            )
            population = [Individual(genome, problem=self._problem) for genome in genomes]
            Individual.evaluate_population(population)
            self._history.append([population])

        def run_metaepoch(self, tree) -> None:
            # This method is called in each meta-epoch
            self.run()

            # Check if stopping conditions are met
            if (gsc_value := tree._gsc(tree)) or self._lsc(self):
                self._active = False
                message = "Random Search Deme finished due to GSC" if gsc_value else "Random Search Deme finished due to LSC"
                self.log(message)
                return

Understanding DemeInitArgs
--------------------------

When implementing a custom deme, you'll receive a ``DemeInitArgs`` object in the constructor. This dataclass contains all the necessary initialization parameters for your deme:

.. code-block:: python

    @dataclass
    class DemeInitArgs:
        id: str
        level: int
        config: BaseLevelConfig
        logger: FilteringBoundLogger
        started_at: int = 0
        sprout_seed: Individual | None = None
        random_seed: int | None = None
        parent_deme: AbstractDeme | None = None

Understanding these fields:

- ``id``: A unique string identifier for your deme
- ``level``: The hierarchical level in the HMS tree (starts at 0 for root)
- ``config``: Your custom configuration class instance that inherits from ``BaseLevelConfig``
- ``logger``: A structured logger for outputting debug information
- ``started_at``: The metaepoch number when this deme was created
- ``sprout_seed``: For non-root demes, this is the first individual that sprouted this deme
- ``random_seed``: A seed for random number generation to ensure reproducibility
- ``parent_deme``: Reference to the parent deme that sprouted this deme (None for root demes)

In your custom deme implementation, you'll typically:

1. Pass the ``DemeInitArgs`` object to the parent constructor
2. Cast the ``config`` field to your specific config class type
3. Access the configuration parameters you need
4. Use the provided random seed for any randomized operations

Step 3: Register and Use Your Custom Deme
-----------------------------------------

Finally, register your custom deme by creating a mapping from your config class to your deme class and passing it to the ``hms`` function:

.. code-block:: python

    from pyhms import hms
    from pyhms.stop_conditions import DontStop, MetaepochLimit

    # Create your deme configuration
    random_search_config = RandomSearchConfig(
        problem=your_problem,
        lsc=DontStop(),
        pop_size=100
    )

    # Define the mapping from config class to deme class
    config_class_to_deme_class = {
        RandomSearchConfig: RandomSearchDeme
    }

    # Use your custom deme in pyHMS
    result = hms(
        level_config=[random_search_config],
        gsc=MetaepochLimit(10),
        sprout_cond=your_sprout_condition,
        config_class_to_deme_class=config_class_to_deme_class
    )

Important AbstractDeme Properties and Methods
---------------------------------------------

When implementing your custom deme, you can use the following properties and methods from the ``AbstractDeme`` base class:

- ``self._problem``: The optimization problem
- ``self._bounds``: The bounds of the search space
- ``self._active``: A flag indicating if the deme is active
- ``self._history``: History of populations (list of lists of individuals)
- ``self.log(message)``: Log a message with additional meta information
- ``self.centroid``: Compute the centroid of the current population
- ``self.best_individual``: Get the best individual found by the deme
- ``self.current_population``: Get the current population

The most important method you must implement is ``run_metaepoch(self, tree)``, which is called in each meta-epoch of the HMS algorithm.
