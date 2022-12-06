import logging
import subprocess as sp
from typing import List
from leap_ec.problem import ScalarProblem

from hms.util import str_to_list
from .loss import squared_loss

logger = logging.getLogger(__name__)

class ErikksonProblem(ScalarProblem):
    def __init__(
        self, 
        script_path: str, 
        solver_path: str, 
        accuracy_level: int,
        observed_data: List[float],
        loss_function=squared_loss
        ):

        super().__init__(maximize=False)
        self.script_path = script_path
        self.solver_path = solver_path
        self.accuracy_level = accuracy_level
        self.observed_data = observed_data
        self.loss_function = loss_function

    def evaluate(self, phenome, *args, **kwargs):
        if len(phenome) != 2:
            raise ValueError("Erikkson problem is 2-dimensional")

        cmd = self.make_command(phenome)
        qoi_str = self.run_command(cmd)
        qoi = str_to_list(qoi_str)
        logger.debug(f"QOI: {qoi}")

        return self.loss_function(qoi, self.observed_data)

    def make_command(self, phenome) -> str:
        cmd = [
            self.script_path, 
            str(self.accuracy_level)
        ] 
        cmd += [str(coord) for coord in phenome] 
        cmd.append(self.solver_path)
        return cmd

    def run_command(self, cmd) -> str:
        logger.debug(f"Command: {cmd}")
        p = sp.run(cmd, capture_output=True, check=True, text=True)
        p.check_returncode()
        logger.debug(f"External cmd STDOUT: {p.stdout}")
        logger.debug(f"External cmd STDERR: {p.stderr}")
        outs = p.stdout.split('\n')
        # First line ignored
        return outs[1]

    def __str__(self) -> str:
        return f"ErikksonProblem(acc={self.accuracy_level}, loss={self.loss_function.__name__})"