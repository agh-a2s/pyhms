class LevelConfig:
    def __init__(self, ea, sample_std_dev=0.1, 
        lsc=lambda deme: False, run_gradient_method=False) -> None:
        self.ea = ea
        self.sample_std_dev = sample_std_dev
        self.lsc = lsc
        self.run_gradient_method = run_gradient_method
