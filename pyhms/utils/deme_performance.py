import numpy as np
import pandas as pd

from ..demes.abstract_deme import AbstractDeme


def get_average_variance_per_generation(deme: AbstractDeme, selected_dimensions: list[int] | None) -> pd.DataFrame:
    """Computes the average variance of genomes for each generation within a deme."""
    rows = []
    for metaepoch_populations in deme._history:
        for population in metaepoch_populations:
            population_genomes = np.array([ind.genome for ind in population])
            if selected_dimensions is not None:
                population_genomes = population_genomes[:, selected_dimensions]
            generation_variances = np.var(
                population_genomes,
                axis=0,
            )
            row = {
                "Average Variance of Genome": np.mean(generation_variances),
            }
            rows.append(row)
    return pd.DataFrame(rows)
