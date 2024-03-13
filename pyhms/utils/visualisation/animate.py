from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...tree import DemeTree

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .dimensionality_reduction import DimensionalityReducer, NaiveDimensionalityReducer


def save_tree_animation(
    tree: "DemeTree",
    filepath: str = "hms_tree.gif",
    dimensionality_reducer: DimensionalityReducer = NaiveDimensionalityReducer(),
) -> None:
    fig, ax = plt.subplots()
    bounds = tree.config.levels[0].bounds
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.autoscale(enable=False, tight=True)
    ax.scatter([], [])

    def animate(frame):
        ax.clear()
        ax.autoscale(enable=False, tight=True)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])

        for _, deme in tree.all_demes:
            deme_marker = None
            if frame >= deme.started_at:
                deme_marker = "o"
                deme_epoch = frame - deme.started_at
            if frame > deme.started_at + deme.metaepoch_count:
                deme_marker = "x"
                deme_epoch = -1

            if deme_marker is not None:
                deme_pop = deme.history[deme_epoch]
                X = np.array([ind.genome for ind in deme_pop])
                X_reduced = dimensionality_reducer.fit_transform(X)
                ax.scatter(
                    X_reduced[:, 0],
                    X_reduced[:, 1],
                    label=f"{deme.__class__.__name__} {deme.id}",
                    marker=deme_marker,
                )

        ax.legend(loc="upper left", title=f"Step {frame}")
        return ax

    def init():
        pass

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=tree.metaepoch_count + 1,
        init_func=init,
        interval=1000,
        repeat=False,
    )

    ani.save(filepath)
