from ..tree import DemeTree
from .cluster import Cluster
from .merge_conditions import MergeCondition
from .consolidator import PairwiseNeighborConsolidator
from .local_basin_agent import LocalBasinAgentExecutor
from ..demes.single_pop_eas.sea import MWEA
from ..core.problem import get_function_problem
from .kriging import KrigingLandscapeApproximator
import numpy as np


class LandscapeApproximator:
    def __init__(
        self,
        hms_tree: DemeTree,
        merge_condition: MergeCondition,
        local_basin_epochs: int,
        mwea: MWEA | None = None,
    ):
        self.clusters = [Cluster.from_deme(deme) for deme in hms_tree.leaves]
        self.cluster_reducer = PairwiseNeighborConsolidator(
            merge_condition=merge_condition, max_distance=None
        )
        if mwea is None:
            problem = get_function_problem(hms_tree.root._problem)
            self.mwea = MWEA.create(problem=problem)
        else:
            self.mwea = mwea
        self.local_basin_agent_executor = LocalBasinAgentExecutor(
            ea=self.mwea, n_epochs=local_basin_epochs
        )
        self.kriging = KrigingLandscapeApproximator()

    def fit(self) -> None:
        reduced_clusters = self.cluster_reducer.reduce_clusters(self.clusters)
        local_basin_population = self.local_basin_agent_executor(reduced_clusters)
        self.kriging.fit(local_basin_population)

    def predict(self, x: np.ndarray) -> float:
        return self.kriging.predict(x)

    def plot(self) -> None:
        self.kriging.plot()
