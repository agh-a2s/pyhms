Problem
=======

`pyhms` provides different `Problem` wrappers. These wrappers are used to wrap the problem and provide additional functionality such as counting the number of evaluations (`EvalCountingProblem`, `EvalCutoffProblem`), or stopping the evaluation when a certain precision is reached (`PrecisionCutoffProblem`).

.. autoclass:: pyhms.core.problem.EvalCountingProblem

.. autoclass:: pyhms.core.problem.EvalCutoffProblem

.. autoclass:: pyhms.core.problem.PrecisionCutoffProblem

.. autoclass:: pyhms.core.problem.StatsGatheringProblem
