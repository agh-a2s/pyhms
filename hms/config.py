class LevelConfig:
    def __init__(self, ea_class, generations, pop_size, problem, bounds, lsc, 
        sample_std_dev=1.0, **kwargs) -> None:
        self.ea_class = ea_class
        self.generations = generations
        self.pop_size = pop_size
        self.problem = problem
        self.bounds = bounds
        self.lsc = lsc
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.__dict__)
