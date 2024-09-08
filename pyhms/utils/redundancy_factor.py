from ..cluster.cluster import Cluster
from ..cluster.merge_conditions import HillValleyMergeCondition
from ..core.individual import Individual
from ..demes.cma_deme import CMADeme


def count_redundant_evaluations_for_cma_demes(
    demes: list[CMADeme],
    optimal_solution: Individual | None,
    k: int,
) -> float:
    sorted_demes = sorted(demes, key=lambda deme: deme.started_at)
    sorted_demes = [deme for deme in sorted_demes if deme.best_individual is not None]
    problem = sorted_demes[0]._problem
    clusters = [Cluster.from_cma_deme(deme) for deme in sorted_demes]
    optimal_cluster = Cluster.from_individuals([optimal_solution], estimate_params=False) if optimal_solution else None
    merge_condition = HillValleyMergeCondition(problem=problem, k=k)
    redundant_evaluations_count = 0
    for cluster_idx in range(len(clusters)):
        if optimal_cluster and merge_condition.can_merge(optimal_cluster, clusters[cluster_idx]):
            continue
        for previous_cluster_idx in range(cluster_idx):
            if merge_condition.can_merge(clusters[previous_cluster_idx], clusters[cluster_idx]):
                redundant_evaluations_count += sorted_demes[cluster_idx].n_evaluations
                break
    return redundant_evaluations_count
