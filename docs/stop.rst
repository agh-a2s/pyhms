.. _stop_section:

Stop Conditions
===============

There are three types of stop conditions:

* Local Stop Conditions (LSC) - used to stop a single deme,
* Global Stop Conditions (GSC) - used to stop the whole HMS,
* Universal Stop Conditions (USC) - these conditions can be used as both LSC and GSC.


Local Stop Conditions
---------------------

An interface for LSC is defined in the :class:`LocalStopCondition` class. It has a single method, :meth:`LocalStopCondition.__call__`, which takes a single argument, a deme (AbstractDeme).
The method should return a boolean value indicating whether the deme should stop.

.. code-block:: python

    class LocalStopCondition(ABC):
        @abstractmethod
        def __call__(self, deme: "AbstractDeme") -> bool:
            raise NotImplementedError()

A simple example of LSC is :class:`AllChildrenStopped`:

.. code-block:: python

    class AllChildrenStopped(LocalStopCondition):
    """
    LSC is true if all children of the deme are stopped.
    """

    def __call__(self, deme: "AbstractDeme") -> bool:
        if not deme.children:
            return False

        return all(not child.is_active for child in deme.children)

It's easy to create a custom LSC by extending :class:`LocalStopCondition` and implementing the :meth:`LocalStopCondition.__call__` method.

Supported Local Stop Conditions
--------------------------------

.. autoclass:: pyhms.stop_conditions.lsc.FitnessSteadiness

.. autoclass:: pyhms.stop_conditions.lsc.AllChildrenStopped

Global Stop Conditions
----------------------

An interface for GSC is defined in the :class:`GlobalStopCondition` class. It has a single method, :meth:`GlobalStopCondition.__call__`, which takes a single argument, a DemeTree.

.. code-block:: python

    class GlobalStopCondition(ABC):
        @abstractmethod
        def __call__(self, tree: "DemeTree") -> bool:
            raise NotImplementedError()

A simple example of GSC is :class:`RootStopped`:

.. code-block:: python

    class RootStopped(GlobalStopCondition):
    """
    GSC is true if the root is not active.
    """

    def __call__(self, tree: "DemeTree") -> bool:
        return not tree.root.is_active

Supported Global Stop Conditions
--------------------------------

.. autoclass:: pyhms.stop_conditions.gsc.RootStopped

.. autoclass:: pyhms.stop_conditions.gsc.AllStopped

.. autoclass:: pyhms.stop_conditions.gsc.FitnessEvalLimitReached

.. autoclass:: pyhms.stop_conditions.gsc.SingularProblemEvalLimitReached

.. autoclass:: pyhms.stop_conditions.gsc.SingularProblemPrecisionReached

.. autoclass:: pyhms.stop_conditions.gsc.NoActiveNonrootDemes

Universal Stop Conditions
-------------------------

USC can be used as  LSC and GSC. An interface for USC is defined in the :class:`UniversalStopCondition` class. It has a single method, :meth:`UniversalStopCondition.__call__`, which takes a single argument, a deme (AbstractDeme) or tree (DemeTree).
The method should return a boolean value indicating whether the deme or tree should stop.

.. code-block:: python

    class UniversalStopCondition(ABC):
        @abstractmethod
        def __call__(self, obj: Union["DemeTree", "AbstractDeme"]) -> bool:
            raise NotImplementedError()


Supported Universal Stop Conditions
-----------------------------------

.. autoclass:: pyhms.stop_conditions.usc.MetaepochLimit

.. autoclass:: pyhms.stop_conditions.usc.DontStop

.. autoclass:: pyhms.stop_conditions.usc.DontRun
