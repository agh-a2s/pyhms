Hierarchic Memetic Strategy (HMS)
=================================

The HMS is a strategy that manages multiple populations (called `demes`) of solutions to a problem, 
each group evolving independently according to its own set of rules. 
Imagine it as a tree where each deme evolves separately but is part of a bigger structure with a single starting point (root). 
This tree is organized in levels, and each level can work a bit differently because it has its own settings or parameters.
A single step of the strategy, called metaepoch, consists of a series of executions in each deme.
HMS can utilize various optimization algorithms as deme engines. Currently we support:

* Differential Evolution (DE),
* Covariance Matrix Adaptation Evolution Strategy (CMA-ES),
* Simple Evolutionary Algorithm (SEA),
* Local Optimization Method (e.g. L-BFGS-B).

The strategy is designed to be flexible and easy to extend. It is possible to add new deme engines or modify the existing ones.
HMS is designed to perform chaotic search on high levels (e.g. at the root level) to look for promising regions and high-precision search on the lower levels.
This way it inherently balances between exploration and exploitation.

.. image:: _static/images/pseudocode.png
   :alt: pseudocode
   :align: center
