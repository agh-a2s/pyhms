from pymoo.core.callback import Callback   


class HistoryCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["history"] = []

    def notify(self, algorithm):
        self.data["history"].append(algorithm.pop.get("X", "F"))
