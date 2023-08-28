from utils.costs import Penalty
from utils import DataBase


class PenaltyTrajectory(DataBase):
    def __init__(
        self,
        u_pen: Penalty = None,
        x_pen: Penalty = None,
        rot_pen: Penalty = None,
        state_bound_pen: Penalty = None,
        state_pen: Penalty = None,
        q_pen: Penalty = None,
        v_pen: Penalty = None,
        prefix: str = "",
    ):

        self.u_pen = u_pen
        self.x_pen = x_pen
        self.rot_pen = rot_pen
        self.state_bound_pen = state_bound_pen
        self.state_pen = state_pen
        self.q_pen = q_pen
        self.v_pen = v_pen

        # e.g. "ddp_" or "mpc_" if you use two penalty classes
        self.prefix = prefix

    def __str__(self) -> str:
        return ""

    def save_to_metadata(self) -> dict:
        return {
            self.prefix + "u_pen": self.u_pen.pen if self.u_pen is not None else None,
            self.prefix + "x_pen": self.x_pen.pen if self.x_pen is not None else None,
            self.prefix + "rot_pen": self.rot_pen.pen
            if self.rot_pen is not None
            else None,
            self.prefix + "state_bound_pen": self.state_bound_pen.pen
            if self.state_bound_pen is not None
            else None,
            self.prefix + "state_pen": self.state_pen.pen
            if self.state_pen is not None
            else None,
            self.prefix + "q_pen": self.q_pen.pen if self.q_pen is not None else None,
            self.prefix + "v_pen": self.v_pen.pen if self.v_pen is not None else None,
        }

    def save_to_file(self) -> dict:
        return {}
