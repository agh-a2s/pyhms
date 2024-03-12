import dill as pkl
from leap_ec.individual import Individual
from structlog.typing import FilteringBoundLogger

from .config import TreeConfig
from .demes.abstract_deme import AbstractDeme
from .demes.initialize import init_from_config, init_root
from .logging_ import DEFAULT_LOGGING_LEVEL, get_logger
from .problem import StatsGatheringProblem
from .sprout.sprout_mechanisms import SproutMechanism
from .utils.print_tree import format_deme, format_deme_children_tree
from .utils.visualisation.animate import save_tree_animation
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

        if "random_seed" in config.options:
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
        return max(deme.best_individual for level in self._levels for deme in level)

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

    def _do_sprout(self, deme_seeds: dict[AbstractDeme, tuple[dict[str, float], list[Individual]]]) -> None:
        for deme, info in deme_seeds.items():
            target_level = deme.level + 1

            for ind in info[1]:
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

    def save_animation(self, filepath: str = "hms_tree.mp4") -> None:
        """
        Saves an animation of the tree evolution as an mp4 file.
        """
        save_tree_animation(self, filepath)

    def plot_problem(self) -> None:
        """
        Plots 2D grid for the root level problem.
        """
        grid = Grid2DProblemEvaluation(self.config.levels[0].problem, self.config.levels[0].bounds)
        grid.evaluate()
        grid.plot()
