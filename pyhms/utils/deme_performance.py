import numpy as np
import pandas as pd

from ..demes.abstract_deme import AbstractDeme


def get_variance_per_gene(deme: AbstractDeme) -> pd.DataFrame:
    rows = []
    for metaepoch_idx, metaepoch_populations in enumerate(deme._history):
        for generation_idx, population in enumerate(metaepoch_populations):
            generation_variances = np.var([ind.genome for ind in population], axis=0)
            row = {f"Gene {gene_idx}": variance for gene_idx, variance in enumerate(generation_variances)}
            row["Metaepoch"] = f"{metaepoch_idx}/{generation_idx}"
            rows.append(row)
    return pd.DataFrame(rows)
