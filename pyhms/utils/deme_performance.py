import numpy as np
import pandas as pd

from ..demes.abstract_deme import AbstractDeme


def get_average_variance_per_generation(deme: AbstractDeme) -> pd.DataFrame:
    """Computes the average variance of genomes for each generation within a deme."""
    rows = []
    for metaepoch_populations in deme._history:
        for population in metaepoch_populations:
            generation_variances = np.var([ind.genome for ind in population], axis=0)
            row = {
                "Average Variance of Genome": np.mean(generation_variances),
            }
            rows.append(row)
    return pd.DataFrame(rows)
