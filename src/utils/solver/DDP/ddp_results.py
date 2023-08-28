import numpy as np

from utils.solver import SolverResultsBase


class DDPResults(SolverResultsBase):
    def __init__(self, state: np.ndarray, u: np.ndarray, solved: bool):
        self.state = state
        self.u = u
        self.solved = solved

    def __str__(self) -> str:
        return "DDP"

    def save_to_file(self) -> dict:
        return {"state": self.state, "u": self.u}

    def save_to_metadata(self) -> dict:
        return {"solver": "DDP"}
