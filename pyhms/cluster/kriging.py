import numpy as np
import plotly.graph_objects as go
from pykrige.ok import OrdinaryKriging

from ..core.population import Population


class KrigingLandscapeApproximator:
    def __init__(self) -> None:
        self.population: Population | None = None
        self.model: OrdinaryKriging | None = None

    def fit(self, population: Population) -> None:
        self.population = population
        if population.genomes.shape[1] != 2:
            raise ValueError("Kriging only supports 2-dimensional genomes")
        self.model = OrdinaryKriging(
            population.genomes[:, 0],
            population.genomes[:, 1],
            population.fitnesses,
            variogram_model="linear",
            verbose=False,
            enable_plotting=False,
        )

    def predict(self, genomes: np.ndarray) -> np.ndarray:
        return self.model.execute("points", genomes)

    def plot(self, number_of_points_per_dim: int = 100) -> None:
        bounds = self.population.problem.bounds
        x = np.linspace(bounds[0][0], bounds[0][1], number_of_points_per_dim)
        y = np.linspace(bounds[1][0], bounds[1][1], number_of_points_per_dim)
        z, _ = self.model.execute("grid", x, y)
        fig = go.Figure(
            data=go.Contour(
                z=z,
                x=x,
                y=y,
                contours=dict(
                    start=z.min(),
                    end=z.max(),
                    size=(z.max() - z.min()) / 10,
                    coloring="lines",
                ),
                line=dict(color="blue"),
            )
        )

        fig.update_layout(
            title="Kriging Contour Plot",
            xaxis_title="X1",
            yaxis_title="X2",
        )

        fig.show()
