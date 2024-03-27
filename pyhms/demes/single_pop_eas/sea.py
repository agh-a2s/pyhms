from abc import ABC, abstractmethod
from typing import Any, Iterator

import leap_ec.ops as lops
import numpy as np
import toolz
from leap_ec.individual import Individual
from leap_ec.problem import Problem
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation
from leap_ec.util import wrap_curry
from toolz import pipe

DEFAULT_K_ELITES = 1
DEFAULT_GENERATIONS = 1
DEFAULT_MUTATION_STD = 1.0
DEFAULT_P_XOVER = 0.3


def find_closest(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_expanded = X[:, np.newaxis, :]
    Y_expanded = Y[np.newaxis, :, :]
    distances_squared = np.sum((X_expanded - Y_expanded) ** 2, axis=2)
    closest_indices = np.argmin(distances_squared, axis=1)
    return closest_indices


@wrap_curry
@lops.listlist_op
def crowding_survival(offsprings: list[Individual], parents: list[Individual], k: int = 1):
    """Replaces an individual in the population with the offspring based on crowding."""
    original_num_offspring = len(offsprings)
    elites = list(toolz.itertoolz.topk(k, parents))
    parent_genomes = np.array([ind.genome for ind in parents])
    offspring_genomes = np.array([ind.genome for ind in offsprings])
    closest_population_genome_indices = find_closest(offspring_genomes, parent_genomes)
    new_population: list[Individual] = []
    for offspring_idx, population_idx in enumerate(closest_population_genome_indices):
        parent = parents[population_idx]
        offspring = offsprings[offspring_idx]
        if offspring.fitness > parent.fitness:
            new_population.append(offspring)
        else:
            new_population.append(parent)
    new_population.extend(elites)
    return list(toolz.itertoolz.topk(original_num_offspring, new_population))


@wrap_curry
@lops.iteriter_op
def mutate_uniform(next_individual: Iterator, p_mutate: float = 0.1, bounds=(-np.inf, np.inf)) -> Iterator:
    """
    Mutate an individual by applying a uniform mutation to a single random gene.

    :param next_individual: Iterator of individuals to be mutated
    :param magnitude_range: A tuple specifying the range (min, max) for the uniform mutation
    :param expected_num_mutations: Not used in this function, kept for interface compatibility
    :param bounds: Tuple specifying the lower and upper bounds for the mutation
    :return: Iterator of mutated individuals
    """

    while True:
        individual = next(next_individual)
        if np.random.rand() <= p_mutate:
            gene_index = np.random.randint(0, len(individual.genome))
            mutated_gene = np.random.uniform(bounds[gene_index][0], bounds[gene_index][1])
            individual.genome[gene_index] = mutated_gene
            individual.fitness = None
        yield individual


@wrap_curry
@lops.iteriter_op
def mutate_aggresive_uniform(next_individual: Iterator, p_mutate: float = 0.1, bounds=(-np.inf, np.inf)) -> Iterator:
    """
    Mutate an individual by applying a uniform mutation to a single random gene.

    :param next_individual: Iterator of individuals to be mutated
    :param magnitude_range: A tuple specifying the range (min, max) for the uniform mutation
    :param expected_num_mutations: Not used in this function, kept for interface compatibility
    :param bounds: Tuple specifying the lower and upper bounds for the mutation
    :return: Iterator of mutated individuals
    """

    while True:
        individual = next(next_individual)
        for gene_index in range(len(individual.genome)):
            if np.random.rand() <= p_mutate:
                mutated_gene = np.random.uniform(bounds[gene_index][0], bounds[gene_index][1])
                individual.genome[gene_index] = mutated_gene
                individual.fitness = None
        yield individual


class ArithmeticCrossover(lops.Crossover):
    def __init__(self, p_xover: float = 1.0, persist_children=False, fix=None):
        """
        Initialize the arithmetic crossover without a fixed alpha.
        Alpha will be sampled from a uniform distribution for each crossover event.
        :param p_xover: The probability of crossover.
        :param persist_children: Whether to persist children in the population.
        """
        super().__init__(p_xover=p_xover, persist_children=persist_children)
        self.fix = fix

    def recombine(self, parent_a, parent_b):
        """
        Perform arithmetic recombination between two parents to produce two new individuals.
        For each recombination, alpha is sampled from a uniform distribution [0, 1].
        """
        assert isinstance(parent_a.genome, np.ndarray) and isinstance(parent_b.genome, np.ndarray)

        if np.random.rand() <= self.p_xover:
            # Ensure both genomes are of the same length
            min_length = min(parent_a.genome.shape[0], parent_b.genome.shape[0])

            # Sample alpha from a uniform distribution for each crossover
            alpha = np.random.uniform(0, 1)

            # Create offspring by linear combination of parents' genomes
            offspring_a_genome = alpha * parent_a.genome[:min_length] + (1 - alpha) * parent_b.genome[:min_length]
            offspring_b_genome = (1 - alpha) * parent_a.genome[:min_length] + alpha * parent_b.genome[:min_length]

            # Update genomes of offspring
            parent_a.genome[:min_length] = offspring_a_genome
            parent_b.genome[:min_length] = offspring_b_genome
        if self.fix is not None:
            parent_a.genome = self.fix(parent_a.genome)
            parent_b.genome = self.fix(parent_b.genome)
        return parent_a, parent_b


class AbstractEA(ABC):
    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = bounds
        self.pop_size = pop_size

    @abstractmethod
    def run(self, parents: list[Individual] | None = None):
        raise NotImplementedError()

    @classmethod
    def create(cls, problem: Problem, bounds: np.ndarray, pop_size: int, **kwargs):
        return cls(problem=problem, bounds=bounds, pop_size=pop_size, **kwargs)


class SimpleEA(AbstractEA):
    """
    A simple single population EA (SEA skeleton).
    """

    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        pipeline: list[Any],
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
        elitist_survival: Any = lops.elitist_survival,
    ) -> None:
        super().__init__(problem, bounds, pop_size)
        self.generations = generations
        self.pipeline = pipeline
        self.k_elites = k_elites
        if representation is not None:
            self.representation = representation
        else:
            self.representation = Representation(initialize=create_real_vector(bounds=bounds))
        self.elitist_survival = elitist_survival

    def run(self, parents: list[Individual] | None = None) -> list[Individual]:
        if parents is None:
            parents = self.representation.create_population(pop_size=self.pop_size, problem=self.problem)
            parents = Individual.evaluate_population(parents)
        else:
            assert self.pop_size == len(parents)

        return pipe(parents, *self.pipeline, self.elitist_survival(parents=parents, k=self.k_elites))


class SEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
        mutation_std: float | None = DEFAULT_MUTATION_STD,
    ) -> None:
        super().__init__(
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                mutate_gaussian(std=mutation_std, bounds=bounds, expected_num_mutations="isotropic"),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, problem: Problem, bounds: np.ndarray, pop_size: int, **kwargs):
        return cls(
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            generations=kwargs.get("generations", DEFAULT_GENERATIONS),
            k_elites=kwargs.get("k_elites", DEFAULT_K_ELITES),
            mutation_std=kwargs.get("mutation_std", DEFAULT_MUTATION_STD),
        )


class SEAWithCrossover(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
        mutation_std: float | None = DEFAULT_MUTATION_STD,
        p_xover: float | None = DEFAULT_P_XOVER,
    ) -> None:
        super().__init__(
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(p_xover),
                mutate_gaussian(std=mutation_std, bounds=bounds, expected_num_mutations="isotropic"),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, problem: Problem, bounds: np.ndarray, pop_size: int, **kwargs):
        return cls(
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            generations=kwargs.get("generations", DEFAULT_GENERATIONS),
            k_elites=kwargs.get("k_elites", DEFAULT_K_ELITES),
            mutation_std=kwargs.get("mutation_std", DEFAULT_MUTATION_STD),
            p_xover=kwargs.get("p_xover", DEFAULT_P_XOVER),
        )


class SEAWithCrowding(SimpleEA):
    """
    An implementation of SEA using LEAP with crowding survival.
    """

    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
        mutation_std: float | None = DEFAULT_MUTATION_STD,
        p_xover: float | None = DEFAULT_P_XOVER,
    ) -> None:
        super().__init__(
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(p_xover),
                mutate_gaussian(std=mutation_std, bounds=bounds, expected_num_mutations="isotropic"),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
            elitist_survival=crowding_survival,
        )

    @classmethod
    def create(cls, problem: Problem, bounds: np.ndarray, pop_size: int, **kwargs):
        return cls(
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            generations=kwargs.get("generations", DEFAULT_GENERATIONS),
            k_elites=kwargs.get("k_elites", DEFAULT_K_ELITES),
            mutation_std=kwargs.get("mutation_std", DEFAULT_MUTATION_STD),
            p_xover=kwargs.get("p_xover", DEFAULT_P_XOVER),
        )


class GAStyleEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        generations,
        problem,
        bounds,
        pop_size,
        k_elites=1,
        representation=None,
        p_mutation=1,
        p_crossover=1,
        use_aggresive_mutation: bool = False,
    ) -> None:
        mutation = (
            mutate_aggresive_uniform(
                bounds=bounds,
                p_mutate=p_mutation,
            )
            if use_aggresive_mutation
            else mutate_uniform(
                bounds=bounds,
                p_mutate=p_mutation,
            )
        )
        super().__init__(
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(
                    p_xover=p_crossover,
                ),
                mutation,
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, generations, problem, bounds, pop_size, **kwargs):
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 0.9
        p_crossover = kwargs.get("p_crossover") or 0.9
        use_aggresive_mutation = kwargs.get("use_aggresive_mutation") or False
        return cls(
            generations=generations,
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            k_elites=k_elites,
            p_mutation=p_mutation,
            p_crossover=p_crossover,
            use_aggresive_mutation=use_aggresive_mutation,
        )
