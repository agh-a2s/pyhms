import sys

from hms.config import TreeConfig
from hms import AbstractDemeTree
from hms.persist import DemeTreeData
from hms.visualisation.grid import Grid2DEvaluation


class Summary:
    def __init__(self, tree: AbstractDemeTree):
        self._tree = tree

    @property
    def metaepoch_count(self) -> int:
        return self._tree.metaepoch_count

    @property
    def number_of_levels(self) -> int:
        return self._tree.height

    @property
    def tree_config(self) -> TreeConfig:
        return self._tree.config

    @property
    def local_optima(self) -> list:
        return self._tree.optima

    def __str__(self) -> str:
        s = f"Metaepoch count: {self.metaepoch_count}\n"
        s += "\nTree configuration:\n\n"
        s += f"Number of levels: {self.number_of_levels}\n"
        s += f"Global stop condition: {self.tree_config.gsc}\n"
        s += f"Sprout condition: {self.tree_config.sprout_cond}\n"
        indent = " " * 2
        for i, level in enumerate(self.tree_config.levels):
            s += f"Level {i}\n"
            s += indent + f"Problem: {level.problem}\n"
            s += indent + f"Bounds: {level.bounds}\n"
            s += indent + f"EA class: {level.ea_class}\n"
            s += indent + f"Metaepoch length: {level.generations}\n"
            s += indent + f"Population size: {level.pop_size}\n"
            s += indent + f"Local stop condition: {level.lsc}\n"
            s += indent + f"Sample std. dev. {level.sample_std_dev}\n"
            std_keys = {'problem', 'bounds', 'ea_class', 'generations', 'pop_size', 'lsc', 'sample_std_dev'}
            other_pars = {k: v for k, v in level.__dict__.items() if k not in std_keys}
            s += indent + f"Other params: {other_pars}\n"
            s += indent + f"Problem evaluations: {level.problem.n_evaluations}\n"
            m, sd = level.problem.duration_stats
            s += indent + f"Problem duration avg. {m} std. dev. {sd}\n"

        s += "\nLocal optima found\n"
        for o in self.local_optima:
            s += str(o) + "\n"

        s += "\nDeme info:\n"
        for _, deme in self._tree.all_demes:
            start = deme.started_at
            end = deme.started_at + deme.metaepoch_count
            s += f"Deme {deme.id} [{start}-{end}] avg. fitness {deme.avg_fitness()} best {deme.best}\n"

        return s


def main():
    file_name = sys.argv[1]
    if file_name.endswith(".dat"):
        tree_data = DemeTreeData.load_binary(file_name)
        summary = Summary(tree_data)
        print(summary)
    elif file_name.endswith(".gdat"):
        grid = Grid2DEvaluation.load_binary(file_name)
        print(grid.summary())
    else:
        sys.exit("Unknown file extension")


if __name__ == '__main__':
    main()
