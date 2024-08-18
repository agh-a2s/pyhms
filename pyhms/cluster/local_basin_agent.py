from ..core.population import Population
from ..demes.single_pop_eas.sea import BaseSEA
from .cluster import Cluster


class LocalBasinAgentExecutor:
    def __init__(self, ea: BaseSEA, n_epochs: int = 10):
        self.ea = ea
        self.n_epochs = n_epochs

    def __call__(self, clusters: list[Cluster]) -> Population:
        all_individuals = []
        for cluster in clusters:
            individuals = cluster.population.to_individuals()
            for _ in range(self.n_epochs):
                individuals = self.ea.run(individuals)
                all_individuals.extend(individuals)
        return Population.from_individuals(all_individuals)
