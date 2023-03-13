from pymoo.core.callback import Callback
import numpy as np   


class HistoryCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["history"] = []

    def notify(self, algorithm):
        self.data["history"].append(algorithm.pop)
