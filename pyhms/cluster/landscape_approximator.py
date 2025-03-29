import numpy as np

from ..core.problem import get_function_problem
from ..demes.cma_deme import CMADeme
from ..demes.single_pop_eas.sea import MWEA
from ..tree import DemeTree
from .cluster import Cluster
from .consolidator import PairwiseNeighborConsolidator
from .kriging import KrigingLandscapeApproximator
from .local_basin_agent import LocalBasinAgentExecutor
from .merge_conditions import MergeCondition


class LandscapeApproximator:
    """
    Sawicki, Jakub, et al. "Approximating landscape insensitivity regions in solving ill-conditioned inverse problems."
    """

    def __init__(
        self,
        hms_tree: DemeTree,
        merge_condition: MergeCondition,
        local_basin_epochs: int,
        mwea: MWEA | None = None,
    ) -> None:
        assert all(isinstance(deme, CMADeme) for deme in hms_tree.leaves), "All leaves must be CMADeme instances"
        self.clusters = [Cluster.from_cma_deme(deme) for deme in hms_tree.leaves]  # type: ignore[arg-type]
        self.cluster_reducer = PairwiseNeighborConsolidator(merge_condition=merge_condition, max_distance=None)
        if mwea is None:
            problem = get_function_problem(hms_tree.root._problem)
            self.mwea = MWEA.create(problem=problem)
        else:
            self.mwea = mwea
        self.local_basin_agent_executor = LocalBasinAgentExecutor(ea=self.mwea, n_epochs=local_basin_epochs)
        self.kriging = KrigingLandscapeApproximator()

    def fit(self) -> None:
        reduced_clusters = self.cluster_reducer.reduce_clusters(self.clusters)
        local_basin_population = self.local_basin_agent_executor(reduced_clusters)
        self.kriging.fit(local_basin_population)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.kriging.predict(x)

    def plot(self, filepath: str | None = None) -> None:
        self.kriging.plot(filepath=filepath)

    def plot_plateau_contour(
        self,
        threshold: float | None = None,
        filepath: str | None = None,
        number_of_points_per_dim: int = 100,
        show_true_plateau: bool = False,
    ) -> None:
        self.kriging.plot_plateau_contour(
            threshold=threshold,
            filepath=filepath,
            show_true_plateau=show_true_plateau,
            number_of_points_per_dim=number_of_points_per_dim,
        )
