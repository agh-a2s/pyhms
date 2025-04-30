from typing import Type, TypedDict

import numpy as np

from .core.problem import Problem
from .demes.single_pop_eas.sea import SEA, BaseSEA
from .logging_ import LoggingLevel
from .stop_conditions import GlobalStopCondition, LocalStopCondition, UniversalStopCondition


class BaseLevelConfig:
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
    ):
        self.problem = problem
        self.lsc = lsc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def bounds(self) -> np.ndarray:
        return self.problem.bounds


class EALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        ea_class: Type[BaseSEA] = SEA,
        sample_std_dev: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Configure an Evolutionary Algorithm (EA) level.

        Parameters:
        -----------
        pop_size: int
            Population size for the evolutionary algorithm
        problem: Problem
            The optimization problem to solve
        lsc: LocalStopCondition | UniversalStopCondition
            Local stopping condition for this level
        generations: int
            Number of generations to run in each metaepoch
        ea_class: Type[BaseSEA], default=SEA
            The class of evolutionary algorithm to use. Must inherit from BaseSEA.
            Default is standard Evolutionary Algorithm (SEA).
        sample_std_dev: float, default=1.0
            Standard deviation used when sampling new individuals around a sprout seed.
            Controls diversity of the initial population when sprouting.
        **kwargs:
            Additional parameters passed to the EA implementation:

            - mutation_std: float
              Standard deviation for Gaussian mutation. Controls exploration vs exploitation.
            - mutation_std_step: float
              Optional parameter to adapt mutation_std over time.
              If provided, mutation_std will increase by this amount after each generation.
            - k_elites: int
              Number of elite individuals to preserve in each generation.
            - p_mutation: float
              Probability of mutation for each individual (SEA).
            - p_crossover: float
              Probability of crossover (SEAWithCrossover).
        """
        super().__init__(problem, lsc)
        self.ea_class = ea_class
        self.pop_size = pop_size
        self.generations = generations
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)


class DELevelConfig(BaseLevelConfig):
    def __init__(
        self,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        sample_std_dev: float = 1.0,
        dither: bool = False,
        scaling: float = 0.8,
        crossover: float = 0.9,
    ):
        """
        Configure a Differential Evolution (DE) level.

        Parameters:
        -----------
        pop_size: int
            Population size for the DE algorithm
        problem: Problem
            The optimization problem to solve
        lsc: LocalStopCondition | UniversalStopCondition
            Local stopping condition for this level
        generations: int
            Number of generations to run in each metaepoch
        sample_std_dev: float, default=1.0
            Standard deviation used when sampling new individuals around a sprout seed.
            Controls diversity of the initial population when sprouting.
        dither: bool, default=False
            If True, uses adaptive scaling factor (dithering) which can improve
            convergence and robustness
        scaling: float, default=0.8
            Differential weight (F) in the range [0, 2]. Controls the amplification
            of differential vectors during mutation.
        crossover: float, default=0.9
            Crossover probability (CR) in the range [0, 1]. Controls the fraction
            of parameter values copied from the mutant.
        """
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.generations = generations
        self.dither = dither
        self.scaling = scaling
        self.crossover = crossover
        self.sample_std_dev = sample_std_dev


class SHADELevelConfig(BaseLevelConfig):
    def __init__(
        self,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        memory_size: int,
        sample_std_dev: float = 1.0,
    ):
        """
        Configure a Success-History based Adaptive Differential Evolution (SHADE) level.

        Parameters:
        -----------
        pop_size: int
            Population size for the SHADE algorithm
        problem: Problem
            The optimization problem to solve
        lsc: LocalStopCondition | UniversalStopCondition
            Local stopping condition for this level
        generations: int
            Number of generations to run in each metaepoch
        memory_size: int
            Size of the historical memory used to store successful parameter values.
            Typically set between 5-20. Larger values may slow adaptation,
            while smaller values may cause oscillations.
        sample_std_dev: float, default=1.0
            Standard deviation used when sampling new individuals around a sprout seed.
            Controls diversity of the initial population when sprouting.
        """
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.generations = generations
        self.memory_size = memory_size
        self.sample_std_dev = sample_std_dev


class CMALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        sigma0: float | None = None,
        set_stds: bool = False,
        **kwargs,
    ) -> None:
        """
        Configure a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) level.

        Parameters:
        -----------
        problem: Problem
            The problem to optimize
        lsc: LocalStopCondition | UniversalStopCondition
            The local stop condition for this level
        generations: int
            Number of generations to run in each metaepoch
        sigma0: float | None, default=None
            Initial step size. If None:
            - When set_stds=True: defaults to 1.0
            - Otherwise: calculated automatically based on parent deme's population
        set_stds: bool, default=False
            If True, uses standard deviations estimated from parent deme population
            for each dimension instead of a single sigma value. This helps adapt
            the search to the local landscape shape.
        **kwargs:
            Additional parameters to pass to CMAEvolutionStrategy constructor.
            See pycma documentation for all available parameters.

        Notes:
        ------
        The CMADeme implementation passes some parameters to the underlying CMA-ES:
        - bounds: Automatically set from problem bounds
        - CMA_stds: When set_stds=True, calculated from parent deme population
        - random_seed: If provided in TreeConfig options

        The CMA-ES algorithm will also terminate when any of its built-in
        stopping criteria are met, regardless of the LSC provided.
        """
        super().__init__(problem, lsc)
        self.sigma0 = sigma0
        self.generations = generations
        self.set_stds = set_stds
        self.__dict__.update(kwargs)


class LocalOptimizationConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
        self.method = method
        self.__dict__.update(kwargs)


class LHSLevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_size: int,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.__dict__.update(kwargs)


class SobolLevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_size: int,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.__dict__.update(kwargs)


class Options(TypedDict, total=False):
    log_level: LoggingLevel | None  # Default value: "warning"
    hibernation: bool | None  # Default value: False
    random_seed: int | None  # Default value: None


DEFAULT_OPTIONS: Options = {"log_level": LoggingLevel.WARNING, "hibernation": False}


class TreeConfig:
    def __init__(
        self,
        levels: list[BaseLevelConfig],
        gsc: GlobalStopCondition | UniversalStopCondition,
        sprout_mechanism,
        options: Options = DEFAULT_OPTIONS,
        config_class_to_deme_class: dict[type[BaseLevelConfig], "type[AbstractDeme]"] = {},  # type: ignore # noqa: F821
    ) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_mechanism = sprout_mechanism
        self.options = options
        self.config_class_to_deme_class = config_class_to_deme_class
