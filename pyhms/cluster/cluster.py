import numpy as np

from ..core.individual import Individual
from ..core.population import Population
from ..demes.cma_deme import CMADeme
from ..utils.distances import mahalanobis_distance

DEFAULT_CLUSTER_SIZE = 30


class Cluster:
    def __init__(
        self,
        population: Population,
        mean: np.ndarray | None,
        covariance_matrix: np.ndarray | None,
    ):
        self.population = population
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.covariance_matrix_inverse = np.linalg.inv(covariance_matrix) if covariance_matrix is not None else None

    @classmethod
    def from_cma_deme(cls, deme: CMADeme) -> "Cluster":
        return cls(
            Population.from_individuals(deme.all_individuals[-DEFAULT_CLUSTER_SIZE:]),
            deme.mean,
            deme.covariance_matrix,
        )

    @classmethod
    def from_individuals(cls, individuals: list[Individual], estimate_params: bool = True) -> "Cluster":
        population = Population.from_individuals(individuals)
        mean = np.mean(population.genomes, axis=0) if estimate_params else None
        covariance_matrix = np.cov(population.genomes, rowvar=False) if estimate_params else None
        return cls(
            population,
            mean,
            covariance_matrix,
        )

    def is_in_extension(self, point: np.ndarray, threshold: float = 1) -> bool:
        return mahalanobis_distance(point, self.mean, self.covariance_matrix_inverse) < threshold

    @classmethod
    def merge(cls, clusters: list["Cluster"]) -> "Cluster":
        if not clusters:
            raise ValueError("Cannot merge an empty list of clusters")

        if len(clusters) == 1:
            return clusters[0]

        merged_population = clusters[0].population
        for cluster in clusters[1:]:
            merged_population = merged_population.merge(cluster.population)

        total_size = sum(cluster.population.size for cluster in clusters)
        weights = [cluster.population.size / total_size for cluster in clusters]

        merged_mean = np.average([cluster.mean for cluster in clusters], axis=0, weights=weights)
        n = len(clusters[0].mean)
        merged_cov = np.zeros((n, n))

        for i, cluster in enumerate(clusters):
            merged_cov += weights[i] * cluster.covariance_matrix
            diff = cluster.mean - merged_mean
            merged_cov += weights[i] * np.outer(diff, diff)

        return cls(merged_population, merged_mean, merged_cov)
