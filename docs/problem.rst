Problem
=======

The `Problem` class hierarchy in `pyHMS` provides the foundation for defining optimization problems and wrapping them with additional functionality.

Base Classes
------------

.. autoclass:: pyhms.core.problem.Problem
   :members:

.. autoclass:: pyhms.core.problem.FunctionProblem
   :members:

.. autoclass:: pyhms.core.problem.ProblemWrapper
   :members:

Problem Wrappers
----------------

`pyHMS` provides various `Problem` wrappers. These wrappers enhance problem instances with additional functionality without modifying their core behavior. Common use cases include:

1. Monitoring optimization performance (counting evaluations, measuring time)
2. Enforcing constraints (maximum evaluations, precision thresholds)
3. Collecting statistics for analysis

Available Wrappers
^^^^^^^^^^^^^^^^^^

.. autoclass:: pyhms.core.problem.EvalCountingProblem
   :members:

   A wrapper that counts the number of function evaluations performed.

.. autoclass:: pyhms.core.problem.EvalCutoffProblem
   :members:

   A wrapper that stops evaluations after a specified limit is reached, returning infinity (or negative infinity for maximization problems).

.. autoclass:: pyhms.core.problem.PrecisionCutoffProblem
   :members:

   A wrapper that tracks when the solution reaches a specified precision threshold relative to the known global optimum.

.. autoclass:: pyhms.core.problem.StatsGatheringProblem
   :members:

   A wrapper that collects statistics about evaluation times, useful for performance analysis.

Helper Functions
----------------

.. autofunction:: pyhms.core.problem.get_function_problem
