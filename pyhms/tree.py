import dill as pkl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from structlog.typing import FilteringBoundLogger

from .config import TreeConfig
from .core.individual import Individual
from .core.problem import StatsGatheringProblem
from .demes.abstract_deme import AbstractDeme
from .demes.initialize import init_from_config, init_root
from .logging_ import DEFAULT_LOGGING_LEVEL, get_logger
from .sprout.sprout_candidates import DemeCandidates
from .sprout.sprout_mechanisms import SproutMechanism
from .utils.deme_performance import get_average_variance_per_generation
from .utils.print_tree import format_deme, format_deme_children_tree
from .utils.visualisation.animate import tree_animation
from .utils.visualisation.dimensionality_reduction import DimensionalityReducer, NaiveDimensionalityReducer
from .utils.visualisation.grid import Grid2DProblemEvaluation


class DemeTree:
    def __init__(self, config: TreeConfig) -> None:
        self.metaepoch_count: int = 0
        self.config: TreeConfig = config
        self._gsc = config.gsc
        self._sprout_mechanism: SproutMechanism = config.sprout_mechanism
        self._logger: FilteringBoundLogger = get_logger(config.options.get("log_level", DEFAULT_LOGGING_LEVEL))

        nlevels = len(config.levels)
        if nlevels < 1:
            raise ValueError("Level number must be positive")

        if "random_seed" in config.options and config.options["random_seed"] is not None:
            self._random_seed = config.options["random_seed"]
            import random

            import numpy as np

            random.seed(self._random_seed)
            np.random.seed(self._random_seed)
        else:
            self._random_seed = None

        self._levels: list[list[AbstractDeme]] = [[] for _ in range(nlevels)]
        root_deme = init_root(config.levels[0], self._logger)
        self._levels[0].append(root_deme)

    @property
    def levels(self) -> list[list[AbstractDeme]]:
        return self._levels

    @property
    def height(self) -> int:
        return len(self.levels)

    @property
    def root(self) -> AbstractDeme:
        return self.levels[0][0]

    @property
    def all_demes(self) -> list[tuple[int, AbstractDeme]]:
        return [(level_no, deme) for level_no in range(self.height) for deme in self.levels[level_no]]

    @property
    def leaves(self) -> list[AbstractDeme]:
        return self.levels[-1]

    @property
    def active_demes(self) -> list[tuple[int, AbstractDeme]]:
        return [(level_no, deme) for level_no in range(self.height) for deme in self.levels[level_no] if deme.is_active]

    @property
    def active_non_leaves(self) -> list[tuple[int, AbstractDeme]]:
        return [
            (level_no, deme) for level_no in range(self.height - 1) for deme in self.levels[level_no] if deme.is_active
        ]

    @property
    def n_evaluations(self) -> int:
        return sum(deme.n_evaluations for _, deme in self.all_demes)

    @property
    def best_leaf_individual(self) -> Individual:
        return max(deme.best_individual for deme in self.leaves)

    @property
    def best_individual(self) -> Individual:
        return max(deme.best_individual for level in self._levels for deme in level if deme.best_individual)

    @property
    def all_individuals(self) -> list[Individual]:
        individuals_from_all_demes = []
        for _, deme in self.all_demes:
            individuals_from_all_demes.extend(deme.all_individuals)
        return individuals_from_all_demes

    def run(self) -> None:
        self._logger.debug(
            "Starting HMS",
            height=self.height,
            options=self.config.options,
            levels=self.config.levels,
            gsc=str(self.config.gsc),
        )
        while not self._gsc(self):
            self.metaepoch_count += 1
            self._logger = self._logger.bind(metaepoch=self.metaepoch_count)
            self.run_metaepoch()
            if not self._gsc(self):
                self.run_sprout()
            if len(self.leaves) > 0:
                self._logger.info(
                    "Metaepoch finished",
                    best_fitness=self.best_leaf_individual.fitness,
                    best_individual=self.best_leaf_individual.genome,
                )
            else:
                self._logger.info("Metaepoch finished. No leaf demes yet.")

    def run_metaepoch(self) -> None:
        for _, deme in reversed(self.active_demes):
            if "hibernation" in self.config.options and self.config.options["hibernation"] and deme._hibernating:
                continue

            deme.run_metaepoch(self)

    def run_sprout(self) -> None:
        deme_seeds = self._sprout_mechanism.get_seeds(self)
        self._do_sprout(deme_seeds)

        if "hibernation" in self.config.options and self.config.options["hibernation"]:
            for _, deme in reversed(self.active_non_leaves):
                if deme in deme_seeds:
                    if deme._hibernating:
                        self._logger.debug("Deme stopped hibernating", deme=deme.id)
                    deme._hibernating = False
                else:
                    if not deme._hibernating:
                        self._logger.debug("Deme started hibernating", deme=deme.id)
                    deme._hibernating = True

    def _do_sprout(self, deme_seeds: dict[AbstractDeme, DemeCandidates]) -> None:
        for deme, deme_candidates in deme_seeds.items():
            target_level = deme.level + 1

            for ind in deme_candidates.individuals:
                new_id = self._next_child_id(deme)
                config = self.config.levels[target_level]

                child = init_from_config(
                    config,
                    new_id,
                    target_level,
                    self.metaepoch_count,
                    sprout_seed=ind,
                    logger=self._logger,
                    random_seed=self._random_seed,
                    parent_deme=deme,
                )
                deme.add_child(child)
                self._levels[target_level].append(child)
                self._logger.debug(
                    "Sprouted new child",
                    seed=child._sprout_seed.genome,
                    id=new_id,
                    tree_level=target_level,
                )

    def _next_child_id(self, deme: AbstractDeme) -> str:
        if deme.level >= self.height - 1:
            raise ValueError("Only non-leaf levels are admissible")

        id_suffix = len(self._levels[deme.level + 1])
        if deme.id == "root":
            return str(id_suffix)
        else:
            return f"{deme.id}/{id_suffix}"

    def summary(self, level_summary: bool | None = True, deme_summary: bool | None = True) -> str:
        """
        Generates a summary report for the HMS.

        Parameters:
        - level_summary (bool | None, optional): If True (default), includes a summary of each level
        in the report. If False, this level detail is omitted.
        - deme_summary (bool | None, optional): If True (default), includes a summary of each deme
        in the report (see `tree` for more details). If False, deme details are omitted.

        Returns:
        - str: A multi-line string containing the formatted summary.
        """
        lines = []
        lines.append(f"Metaepoch count: {self.metaepoch_count}")
        lines.append(f"Best fitness: {self.best_leaf_individual.fitness:.4e}")
        lines.append(f"Best individual: {self.best_leaf_individual.genome}")
        lines.append(f"Number of evaluations: {self.n_evaluations}")
        lines.append(f"Number of demes: {len(self.all_demes)}")
        if level_summary:
            for level, level_demes in enumerate(self.levels):
                lines.append(f"\nLevel {level+1}.")
                level_best_individual = max(deme.best_individual for deme in level_demes)
                lines.append(f"Best fitness: {level_best_individual.fitness:.4e}")
                lines.append(f"Best individual: {level_best_individual.genome}")
                lines.append(f"Number of evaluations: {sum(deme.n_evaluations for deme in level_demes)}")
                lines.append(f"Number of demes: {len(level_demes)}")
                level_problem = self.config.levels[level].problem
                if isinstance(level_problem, StatsGatheringProblem):
                    m, sd = level_problem.duration_stats
                    lines.append(f"Problem duration avg. {m:.4e} std. dev. {sd:.4e}")
        if deme_summary:
            lines.append("\n" + self.tree())
        return "\n".join(lines)

    def pickle_dump(self, filepath: str = "hms_snapshot.pkl") -> None:
        self._logger.info("Dumping tree snapshot", filepath=filepath)
        with open(filepath, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def pickle_load(filepath: str) -> "DemeTree":
        with open(filepath, "rb") as f:
            tree = pkl.load(f)
        tree._logger.info("Tree loaded from snapshot", filepath=filepath)
        return tree

    def tree(self) -> str:
        """
        Generates a string representation of the tree.

        Returns:
        - str: A multi-line string containing the formatted tree.

        Notes:
        - Each deme is represented by a line containing its type (e.g. CMADeme), id, best solution
        and best fitness, sprout seed (for non root demes), number of evaluations.
        - The fitness value is formatted using {:.2e} (that's why we use ~= instead of =).
        - Solutions are formatted using {:#.2f}.
        - *** is appended to the best deme (one with the best fitness value) representation.
        """
        return (
            format_deme(self.root, self.best_individual.fitness)
            + "\n"
            + format_deme_children_tree(self.root, best_fitness=self.best_individual.fitness)
        )

    def animate(
        self,
        filepath: str | None = None,
        dimensionality_reducer: DimensionalityReducer = NaiveDimensionalityReducer(),
    ) -> animation.FuncAnimation:
        """
        Returns an animation (animation.FuncAnimation) of the tree evolution.
        It can save an animation of the tree evolution as an gif/mp4 file.
        To save the animation, provide the filepath argument.
        In case of multidimensional genomes dimensionality reducer is employed.
        By default it uses NaiveDimensionalityReducer which takes only the first two dimensions.
        """
        animation = tree_animation(self, dimensionality_reducer)
        if filepath is not None:
            animation.save(filepath)
        return animation

    def plot_problem_surface(self, filepath: str | None = None) -> None:
        """
        Plots 2D grid for the root level problem.
        In case of multidimensional problem, only the first two dimensions are considered.
        To save the plot, provide the filepath argument.
        """
        grid = Grid2DProblemEvaluation(self.config.levels[0].problem, self.config.levels[0].bounds)
        grid.evaluate()
        grid.plot(filepath)

    def plot_best_fitness(self, filepath: str | None = None) -> None:
        """
        Plots the best fitness value for each metaepoch.
        To save the plot, provide the filepath argument.
        """
        pd.concat(
            [pd.DataFrame([deme.best_fitness_by_metaepoch], index=[deme.name]) for _, deme in self.all_demes]
        ).T.plot(marker="o", linestyle="-")
        plt.title("Best fitness by metaepoch")
        plt.xlabel("Metaepoch")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        if filepath:
            plt.savefig(filepath)
        plt.show()

    def plot_deme_variance(
        self, filepath: str | None = None, deme_id: str | None = "root", selected_dimensions: list[int] | None = None
    ) -> None:
        """
        Plots the average variance of genes/dimensions across generations for a given deme
        to analyze its convergence behavior and exploration of the search space.
        This plot is useful for visualizing how the diversity within a deme changes over time.
        The main use case is the analysis of the root deme.
        To save the plot, provide the filepath argument.
        """
        deme = next(deme for _, deme in self.all_demes if deme.id == deme_id)
        variance_per_gene = get_average_variance_per_generation(deme, selected_dimensions)
        variance_per_gene.plot()
        plt.plot(
            variance_per_gene.index,
            variance_per_gene["Average Variance of Genome"],
            marker="o",
            linestyle="-",
            color="b",
        )
        plt.title(f"Variance Across Generations for {deme.id.capitalize()} Deme")
        plt.xlabel("Generation Number")
        plt.ylabel("Average Variance of Genome")
        plt.grid(True)
        if filepath:
            plt.savefig(filepath)
        plt.show()
