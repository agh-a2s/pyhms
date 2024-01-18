import os.path as op
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pyhms.persist import DemeTreeData

data_file = sys.argv[1]
tree = DemeTreeData.load_binary(data_file)

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
            x = [ind.point[0] for ind in deme_pop]
            y = [ind.point[1] for ind in deme_pop]
            ax.scatter(x, y, label=deme.id, marker=deme_marker)

    ax.legend(loc="upper left", title=f"Step {frame}")
    return ax


# Init only required for blitting to give a clean slate.
def init():
    pass


ani = animation.FuncAnimation(
    fig, animate, frames=tree.metaepoch_count + 1, init_func=init, interval=1000, repeat=False
)  # , blit=True)

ani_file, _ = op.splitext(data_file)
ani.save(ani_file + ".mp4")
plt.show()
