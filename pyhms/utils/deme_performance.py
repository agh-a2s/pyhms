from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..demes.abstract_deme import AbstractDeme


class Indicator(ABC):
    def __init__(self, indicator_title: str) -> None:
        self.indicator_title = indicator_title

    def __call__(self, deme: AbstractDeme, selected_dimensions: list[int] | None) -> pd.DataFrame:
        rows = []
        for metaepoch_populations in deme._history:
            for population in metaepoch_populations:
                population_genomes = np.array([ind.genome for ind in population])
                if selected_dimensions is not None:
                    population_genomes = population_genomes[:, selected_dimensions]
                indicator = self.compute(population_genomes)
                row = {
                    self.indicator_title: indicator,
                }
                rows.append(row)
        return pd.DataFrame(rows)

    @abstractmethod
    def compute(self, population_genomes: np.ndarray) -> float:
        raise NotImplementedError()


class AverageVariancePerGeneration(Indicator):
    def __init__(self) -> None:
        super().__init__("Average Variance of Genome")

    def compute(self, population_genomes: np.ndarray) -> float:
        generation_variances = np.var(population_genomes, axis=0)
        return np.mean(generation_variances)


class SD(Indicator):
    """
    Sum of Distances (SD).
    For more information, see Preuss, M., and Wessing. S.
    "Measuring multimodal optimization solution sets with a view to multiobjective techniques."
    """

    def __init__(self) -> None:
        super().__init__("Sum of Distances")

    def compute(self, population_genomes: np.ndarray) -> float:
        distances = pairwise_distances(population_genomes)
        return np.sqrt(np.sum(distances))


class SDNN(Indicator):
    """
    Sum of Distances to Nearest Neighbor (SDNN).
    For more information, see Preuss, M., and Wessing. S.
    "Measuring multimodal optimization solution sets with a view to multiobjective techniques."
    """

    def __init__(self) -> None:
        super().__init__("Sum of Distances to Nearest Neighbor")

    def compute(self, population_genomes: np.ndarray) -> float:
        distances = pairwise_distances(population_genomes)
        np.fill_diagonal(distances, np.inf)
        return np.sum(np.min(distances, axis=1))


class SPD(Indicator):
    """
    Solow-Polasky Diversity (SPD)
    For more information, see Preuss, M., and Wessing. S.
    "Measuring multimodal optimization solution sets with a view to multiobjective techniques."
    """

    def __init__(self) -> None:
        super().__init__("Solow-Polasky Diversity")
        self.theta = 1.0

    def compute(self, population_genomes: np.ndarray) -> float:
        distances = pairwise_distances(population_genomes)
        C = np.exp(-self.theta * distances)
        inverse_C = np.linalg.inv(C)
        return float(np.ones(len(distances)) @ inverse_C @ np.ones(len(distances)))


INDICATOR_NAME_TO_INDICATOR = {
    "AvgVar": AverageVariancePerGeneration(),
    "SD": SD(),
    "SDNN": SDNN(),
    "SPD": SPD(),
}
