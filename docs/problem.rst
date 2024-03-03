Problem
=====

`pyhms` provides different `leap_ec.Problem` wrappers. These wrappers are used to wrap the problem and provide additional functionality such as counting the number of evaluations (`EvalCountingProblem`, `EvalCutoffProblem`), or stopping the evaluation when a certain precision is reached (`PrecisionCutoffProblem`).

.. autoclass:: pyhms.problem.EvalCountingProblem

.. autoclass:: pyhms.problem.EvalCutoffProblem

.. autoclass:: pyhms.problem.PrecisionCutoffProblem

.. autoclass:: pyhms.problem.StatsGatheringProblem