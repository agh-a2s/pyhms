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

    def plot(
        self,
        number_of_points_per_dim: int = 100,
        filepath: str | None = None,
    ) -> None:
        if self.population is None or self.model is None:
            raise ValueError("Model must be fitted before plotting")

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
                    size=(z.max() - z.min()) / 15,
                    coloring="fill",
                    showlabels=True,
                ),
                colorscale="Viridis",
                line=dict(width=2),
            )
        )

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            width=1000,
            height=1000,
            template="plotly_white",
            font=dict(size=16),
            margin=dict(l=80, r=80, t=100, b=80),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        fig.show()

        if filepath is not None:
            fig.write_image(filepath, scale=2)

    def plot_plateau_contour(
        self,
        threshold: float | None = None,
        number_of_points_per_dim: int = 100,
        filepath: str | None = None,
    ) -> None:
        if self.population is None or self.model is None:
            raise ValueError("Model must be fitted before plotting")

        bounds = self.population.problem.bounds
        x = np.linspace(bounds[0][0], bounds[0][1], number_of_points_per_dim)
        y = np.linspace(bounds[1][0], bounds[1][1], number_of_points_per_dim)
        z, _ = self.model.execute("grid", x, y)

        if threshold is None:
            threshold = np.median(self.population.fitnesses)

        fig = go.Figure(
            data=go.Contour(
                z=z,
                x=x,
                y=y,
                contours=dict(
                    start=threshold,
                    end=threshold,
                    size=0.1,
                    showlabels=True,
                    labelfont=dict(size=14, color="black"),
                ),
                line=dict(width=3, color="blue"),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
            )
        )

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            width=1000,
            height=1000,
            template="plotly_white",
            font=dict(size=16),
            showlegend=True,
            xaxis_range=[bounds[0][0], bounds[0][1]],
            yaxis_range=[bounds[1][0], bounds[1][1]],
            margin=dict(l=80, r=80, t=80, b=80),
            xaxis=dict(automargin=True, ticklabelposition="outside"),
            yaxis=dict(automargin=True, ticklabelposition="outside"),
        )

        fig.show()

        if filepath is not None:
            fig.write_image(filepath, scale=2)
