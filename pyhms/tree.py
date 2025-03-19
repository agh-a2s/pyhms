from typing import Literal

import dill as pkl
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from structlog.typing import FilteringBoundLogger

from .config import TreeConfig
from .core.individual import Individual
from .core.problem import StatsGatheringProblem, get_function_problem
from .demes.abstract_deme import AbstractDeme
from .demes.cma_deme import CMADeme
from .demes.initialize import init_from_config, init_root
from .logging_ import DEFAULT_LOGGING_LEVEL, get_logger
from .sprout.sprout_candidates import DemeCandidates
from .sprout.sprout_mechanisms import SproutMechanism
from .utils.clusterization import NearestBetterClustering, NearestBetterClusteringWithRule2
from .utils.deme_performance import NAME_TO_METRIC
from .utils.print_tree import format_deme, format_deme_children_tree
from .utils.r5s import R5SSelection
from .utils.redundancy_factor import count_redundant_evaluations_for_cma_demes
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

    @property
    def r5s_solutions(self) -> list[Individual]:
        best_individuals_from_leaves = [deme.best_individual for deme in self.leaves if deme.best_individual]
        selection = R5SSelection()
        return selection(best_individuals_from_leaves)

    def run(self) -> None:
        while not self._gsc(self):
            self.run_step()

    def run_step(self) -> None:
        if self.metaepoch_count == 0:
            self._logger.debug(
                "Starting HMS",
                height=self.height,
                options=self.config.options,
                levels=self.config.levels,
                gsc=str(self.config.gsc),
            )
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
        lines.append(f"Best fitness: {self.best_individual.fitness:.4e}")
        lines.append(f"Best individual: {self.best_individual.genome}")
        lines.append(f"Number of evaluations: {self.n_evaluations}")
        lines.append(f"Number of demes: {len(self.all_demes)}")
        if level_summary:
            for level, level_demes in enumerate(self.levels):
                lines.append(f"\nLevel {level+1}.")
                level_best_individual = max(
                    [deme.best_individual for deme in level_demes if deme.best_individual],
                    default=None,
                )
                if level_best_individual is not None:
                    lines.append(f"Best fitness: {level_best_individual.fitness:.4e}")
                    lines.append(f"Best individual: {level_best_individual.genome}")
                    lines.append(f"Number of evaluations: {sum(deme.n_evaluations for deme in level_demes)}")
                    lines.append(f"Number of demes: {len(level_demes)}")
                    level_problem = self.config.levels[level].problem
                    if isinstance(level_problem, StatsGatheringProblem):
                        m, sd = level_problem.duration_stats
                        lines.append(f"Problem duration avg. {m:.4e} std. dev. {sd:.4e}")
                else:
                    lines.append("No demes available.")
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

    def get_redundancy_factor(self, optimal_solution: Individual | None = None) -> float:
        assert self.height == 2, "This method is only applicable to trees with two levels."
        assert all(
            isinstance(deme, CMADeme) for deme in self.leaves
        ), "This method is only applicable if all leaves use CMA-ES."
        n_redundant_evaluations = count_redundant_evaluations_for_cma_demes(
            self.leaves, optimal_solution, k=10  # type: ignore[arg-type]
        )
        return n_redundant_evaluations / self.n_evaluations

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
        df = pd.concat(
            [pd.DataFrame([deme.best_fitness_by_metaepoch], index=[deme.name]) for _, deme in self.all_demes]
        ).T
        df_long = df.reset_index().melt(id_vars="index", var_name="Deme", value_name="Best Fitness")
        df_long.rename(columns={"index": "Metaepoch"}, inplace=True)
        fig = px.line(
            df_long,
            x="Metaepoch",
            y="Best Fitness",
            color="Deme",
            markers=True,
            title="Best Fitness by Metaepoch",
            labels={"Metaepoch": "Metaepoch", "Best Fitness": "Best Fitness"},
        )
        if filepath:
            fig.write_image(filepath)
        fig.show()

    def plot_deme_metric(
        self,
        filepath: str | None = None,
        deme_id: str | None = "root",
        selected_dimensions: list[int] | None = None,
        metric: Literal["AvgVar", "SD", "SDNN", "SPD"] = "AvgVar",
    ) -> None:
        """
        Plots the average value of divergence metric of genes/dimensions across generations
        for a given deme to analyze its convergence behavior and exploration of the search space.
        This plot is useful for visualizing how the diversity within a deme changes over time.
        The main use case is the analysis of the root deme.
        To save the plot, provide the filepath argument.
        Available metrics: AvgVar, SD, SDNN, SPD.
        """
        deme = next(deme for _, deme in self.all_demes if deme.id == deme_id)
        if metric not in NAME_TO_METRIC:
            raise ValueError(f"Indicator {metric} is not available. Choose from {NAME_TO_METRIC.keys()}")
        indicator_df = NAME_TO_METRIC[metric](deme, selected_dimensions)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=indicator_df.index,
                y=indicator_df[indicator_df.columns[0]],
                mode="lines+markers",
                name=metric,
            )
        )

        fig.update_layout(
            template="plotly_white",
            font=dict(size=14),
            title=f"{metric} for {deme_id.capitalize()} Deme",
            xaxis=dict(
                title="Generation Number",
            ),
            yaxis=dict(
                title=f"{metric} Value",
            ),
        )
        if filepath:
            fig.write_image(filepath, scale=2)

        fig.show()

    def plot_fitness_value_by_distance(self, filepath: str | None = None) -> None:
        data = []  # type: ignore[var-annotated]
        best_genome = self.best_individual.genome
        for level, deme in self.all_demes:
            genomes = np.array([x.genome for x in deme.all_individuals])
            distances_to_best = np.linalg.norm(genomes - best_genome, axis=1)
            fitness_differences = np.array([x.fitness for x in deme.all_individuals]) - self.best_individual.fitness
            data.extend(
                zip(
                    distances_to_best,
                    fitness_differences,
                    [str(level)] * len(distances_to_best),
                )
            )
        df = pd.DataFrame(
            data,
            columns=["Distance to Best Solution", "Fitness Value Difference", "Level"],
        )

        # Calculate correlation coefficient
        corr_coef, _ = pearsonr(df["Distance to Best Solution"], df["Fitness Value Difference"])
        corr_coef = round(corr_coef, 2)

        fig = px.scatter(
            df,
            x="Distance to Best Solution",
            y="Fitness Value Difference",
            color="Level",
            labels={"x": "Distance to Best Genome", "y": "Fitness Value Difference"},
            title=f"Scatter Plot of Individual Fitness vs Distance to Best by Level (Correlation: {corr_coef})",
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(size=14),
        )

        if filepath:
            fig.write_image(filepath)
        fig.show()

    def plot_sprout_seed_distances(self, filepath: str | None = None, level: int | None = 1) -> None:
        """
        Creates a heatmap of the distances between sprout seeds of demes at a given level (1 by default).
        """
        deme_id_with_sprout_seed = [
            (deme.id, deme._sprout_seed) for deme_level, deme in self.all_demes if deme_level == level
        ]
        sprout_seeds = [sprout_seed for _, sprout_seed in deme_id_with_sprout_seed]
        deme_ids = [deme_id for deme_id, _ in deme_id_with_sprout_seed]
        distances = pairwise_distances([ind.genome for ind in sprout_seeds])

        fig = go.Figure(
            data=go.Heatmap(
                z=distances,
                x=deme_ids,
                y=deme_ids,
                colorscale="Viridis",
                text=[[f"{distances[i][j]:.2f}" for j in range(len(distances))] for i in range(len(distances))],
                hoverinfo="text",
            )
        )

        annotations = []
        for i in range(len(distances)):
            for j in range(len(distances)):
                annotations.append(
                    go.layout.Annotation(
                        x=deme_ids[j],
                        y=deme_ids[i],
                        text=f"{distances[i][j]:.2f}",
                        showarrow=False,
                        font=dict(color="white"),
                    )
                )

        fig.update_layout(
            title="Distances between Sprout Seeds",
            xaxis_title="Deme ID",
            yaxis_title="Deme ID",
            xaxis_nticks=len(deme_ids),
            yaxis_nticks=len(deme_ids),
            template="plotly_white",
            annotations=annotations,
        )

        if filepath:
            fig.write_image(filepath)

        fig.show()

    def plot_sprout_candidates(self, filepath: str | None = None, deme_id: str | None = "root") -> None:
        """
        Plots the number of candidates generated and used for a given deme (root by default) across metaepochs.
        """
        generated_candidates_history = self._sprout_mechanism._generated_deme_ids_to_candidates_history
        generated_candidates_for_deme = [candidates.get(deme_id) for candidates in generated_candidates_history]
        used_candidates_history = self._sprout_mechanism._used_deme_ids_to_candidates_history
        used_candidates_for_deme = [candidates.get(deme_id) for candidates in used_candidates_history]
        data = pd.DataFrame(
            {
                "used": [len(candidates.individuals) for candidates in used_candidates_for_deme],
                "generated": [len(candidates.individuals) for candidates in generated_candidates_for_deme],
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["used"],
                mode="lines+markers",
                name="Used Candidates",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["generated"],
                mode="lines+markers",
                name="Generated Candidates",
            )
        )

        fig.update_layout(
            title=f"Number of candidates for deme {deme_id}",
            xaxis_title="Metaepoch",
            yaxis_title="Number of candidates",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            template="plotly_white",
            font=dict(size=14),
        )

        if filepath:
            fig.write_image(filepath)

        fig.show()

    def plot_nbc(
        self,
        dimensionality_reducer: DimensionalityReducer = NaiveDimensionalityReducer(),
        distance_factor: float | None = 2.0,
        truncation_factor: float | None = 1.0,
        use_correction: bool = False,
        use_rule2: bool = False,
    ) -> None:
        """
        Runs the Nearest Better Clustering algorithm and plots the clustered individuals.
        In case of a multidimensional genome, dimensionality reducer is employed.
        """
        nbc_class = NearestBetterClusteringWithRule2 if use_rule2 else NearestBetterClustering
        nbc = nbc_class(self.all_individuals, distance_factor, truncation_factor, use_correction)
        nbc._prepare_spanning_tree()
        nbc.plot_clusters(dimensionality_reducer)

    def plot_population(
        self,
        filepath: str | None = None,
        show_grid: bool = False,
        grid_granularity: float | None = None,
        optimal_fitness_value: float | None = None,
        optimal_genome: np.ndarray | None = None,
        show_all_individuals: bool = False,
    ) -> None:
        function_problem = get_function_problem(self.root._problem)
        bounds = function_problem.bounds
        if show_grid:
            grid_granularity = grid_granularity or (bounds[0][1] - bounds[0][0]) / 200
            grid = Grid2DProblemEvaluation(function_problem, bounds, 0.05)
            grid.evaluate()
            fig = px.imshow(
                grid.z.T,
                labels={"x": "x", "y": "y", "color": "f(x, y)"},
                x=grid.x,
                y=grid.y,
                origin="lower",
                aspect="auto",
                color_continuous_scale="Viridis",
            )
        else:
            fig = go.Figure()

        for _, deme in self.all_demes:
            deme_history = deme.all_individuals if show_all_individuals else deme.history[-1]
            genomes = np.array([x.genome for x in deme_history])
            fitness_values = np.array([x.fitness for x in deme_history])
            labels = [f"f(x, y): {val:.2f}" for val in fitness_values]
            scatter = go.Scatter(
                x=genomes[:, 0],
                y=genomes[:, 1],
                text=labels,
                mode="markers",
                marker=dict(size=10),
                name=deme.id,
            )
            fig.add_trace(scatter)
        if optimal_genome is not None and optimal_fitness_value is not None:
            scatter = go.Scatter(
                x=[optimal_genome[0]],
                y=[optimal_genome[1]],
                text=[f"f(x, y): {optimal_fitness_value:.2f}"],
                mode="markers",
                marker=dict(
                    size=15,
                    symbol="diamond",
                    color="yellow",
                    line=dict(
                        width=2,
                    ),
                ),
                name="Optimum",
            )
            fig.add_trace(scatter)

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            width=1000,
            height=1000,
            template="plotly_white",
            coloraxis_colorbar=dict(
                title=dict(
                    text="f(x, y)",
                    side="right",
                    font=dict(size=14),
                ),
                x=1.15,
                y=0.5,
                len=0.8,
            ),
            legend=dict(
                x=1.05,
                y=0.5,
            ),
        )
        if filepath:
            fig.write_image(filepath)

        fig.show()
