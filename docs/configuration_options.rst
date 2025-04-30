Configuration Options
=====================

This document provides a comprehensive overview of the available configuration options for different algorithms in the pyHMS library.

CMA-ES Configuration (CMALevelConfig)
-------------------------------------

The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a powerful optimization algorithm for continuous domains.

Required Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Type
     - Description
   * - ``problem``
     - ``Problem``
     - The optimization problem to solve
   * - ``lsc``
     - ``LocalStopCondition | UniversalStopCondition``
     - Local stopping condition for this level
   * - ``generations``
     - ``int``
     - Number of generations to run in each metaepoch

Optional Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``sigma0``
     - ``float | None``
     - ``None``
     - Initial step size parameter. If ``None`` and ``set_stds=True``, defaults to 1.0. Otherwise (if ``None`` and ``set_stds=False``) calculated automatically from the parent deme's population.
   * - ``set_stds``
     - ``bool``
     - ``False``
     - If ``True``, uses standard deviations estimated from parent deme population for each dimension separately instead of a single sigma value. This adapts the search to the local landscape shape.
   * - ``**kwargs``
     -
     -
     - Additional parameters passed directly to the ``CMAEvolutionStrategy`` constructor from the ``cma`` package (https://pypi.org/project/cma/). See pycma documentation for all available parameters.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- The CMA-ES implementation is based on the ``cma`` (https://pypi.org/project/cma/)  package's ``CMAEvolutionStrategy`` class
- The following parameters are set automatically:

  - ``bounds``: Set from the problem's bounds
  - ``CMA_stds``: When ``set_stds=True``, calculated from parent deme population
  - Random seed support through the ``random_seed`` option in ``TreeConfig``

- The implementation is located in ``pyhms/demes/cma_deme.py``

Evolutionary Algorithm Configuration (EALevelConfig)
----------------------------------------------------

The Evolutionary Algorithm (EA) implementations in pyHMS provide flexible population-based optimization approaches.

Required Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Type
     - Description
   * - ``pop_size``
     - ``int``
     - Population size for the evolutionary algorithm
   * - ``problem``
     - ``Problem``
     - The optimization problem to solve
   * - ``lsc``
     - ``LocalStopCondition | UniversalStopCondition``
     - Local stopping condition for this level
   * - ``generations``
     - ``int``
     - Number of generations to run in each metaepoch

Optional Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``ea_class``
     - ``Type[BaseSEA]``
     - ``SEA``
     - The class of evolutionary algorithm to use. Must inherit from ``BaseSEA``.
   * - ``sample_std_dev``
     - ``float``
     - ``1.0``
     - Standard deviation used when sampling new individuals around a sprout seed. Controls diversity of the initial population when sprouting.
   * - ``mutation_std``
     - ``float``
     -
     - Standard deviation for Gaussian mutation. Controls exploration vs exploitation balance.
   * - ``mutation_std_step``
     - ``float``
     -
     - Optional parameter to adapt ``mutation_std`` over time. If provided, ``mutation_std`` will increase by this amount after each generation if child deme was not sprouted.
   * - ``k_elites``
     - ``int``
     -
     - Number of elite individuals to preserve in each generation.
   * - ``p_mutation``
     - ``float``
     -
     - Probability of mutation for each individual.
   * - ``p_crossover``
     - ``float``
     -
     - Probability of crossover (used in ``SEAWithCrossover``).

Available EA Classes
~~~~~~~~~~~~~~~~~~~~

pyHMS offers several variants of evolutionary algorithms:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``SEA``
     - Standard Evolutionary Algorithm with tournament selection and Gaussian mutation
   * - ``SEAWithCrossover``
     - EA with tournament selection, arithmetic crossover, and Gaussian mutation
   * - ``SEAWithAdaptiveMutation``
     - EA with adaptive mutation rate that changes over time

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- The implementation is located in ``pyhms/demes/ea_deme.py``, with specific algorithm variants in ``pyhms/demes/single_pop_eas/``
- When used as a root deme, the initial population is sampled uniformly within the problem bounds
- When used as a child deme, the initial population is sampled using a normal distribution around the sprout seed
- The random seed can be set via the ``random_seed`` option in ``TreeConfig``

Differential Evolution Configuration (DELevelConfig)
----------------------------------------------------

Differential Evolution (DE) is a population-based optimization algorithm that's particularly effective for continuous optimization problems and robust against noise.

Required Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Type
     - Description
   * - ``pop_size``
     - ``int``
     - Population size for the DE algorithm
   * - ``problem``
     - ``Problem``
     - The optimization problem to solve
   * - ``lsc``
     - ``LocalStopCondition | UniversalStopCondition``
     - Local stopping condition for this level
   * - ``generations``
     - ``int``
     - Number of generations to run in each metaepoch

Optional Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``sample_std_dev``
     - ``float``
     - ``1.0``
     - Standard deviation used when sampling new individuals around a sprout seed. Controls diversity of the initial population when sprouting.
   * - ``dither``
     - ``bool``
     - ``False``
     - If True, uses adaptive scaling factor (dithering) which can improve convergence and robustness.
   * - ``scaling``
     - ``float``
     - ``0.8``
     - Differential weight (F) in the range [0, 2]. Controls the amplification of differential vectors during mutation.
   * - ``crossover``
     - ``float``
     - ``0.9``
     - Crossover probability (CR) in the range [0, 1]. Controls the fraction of parameter values copied from the mutant.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- The implementation is located in ``pyhms/demes/de_deme.py`` with the core DE algorithm in ``pyhms/demes/single_pop_eas/de.py``
- When used as a root deme, the initial population is sampled uniformly within the problem bounds
- When used as a child deme, the initial population is sampled using a normal distribution around the sprout seed
- The random seed can be set via the ``random_seed`` option in ``TreeConfig``

Success-History Based Adaptive DE Configuration (SHADELevelConfig)
------------------------------------------------------------------

Success-History based Adaptive Differential Evolution (SHADE) is an advanced variant of DE that adaptively tunes its parameters based on successful search history.

Required Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Type
     - Description
   * - ``pop_size``
     - ``int``
     - Population size for the SHADE algorithm
   * - ``problem``
     - ``Problem``
     - The optimization problem to solve
   * - ``lsc``
     - ``LocalStopCondition | UniversalStopCondition``
     - Local stopping condition for this level
   * - ``generations``
     - ``int``
     - Number of generations to run in each metaepoch
   * - ``memory_size``
     - ``int``
     - Size of the historical memory used to store successful parameter values

Optional Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``sample_std_dev``
     - ``float``
     - ``1.0``
     - Standard deviation used when sampling new individuals around a sprout seed. Controls diversity of the initial population when sprouting.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- The implementation is located in ``pyhms/demes/shade_deme.py`` with the core SHADE algorithm in ``pyhms/demes/single_pop_eas/de.py``
- SHADE maintains a historical memory of successful control parameters (CR and F values)
- When used as a root deme, the initial population is sampled uniformly within the problem bounds
- When used as a child deme, the initial population is sampled using a normal distribution around the sprout seed
- The random seed can be set via the ``random_seed`` option in ``TreeConfig``
