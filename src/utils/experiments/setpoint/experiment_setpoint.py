import numpy as np

from utils.experiments import ExperimentBase


class ExperimentSetpoint(ExperimentBase):
    def __init__(
        self,
        x_frame_id: int,
        start_point: np.ndarray,
        target_raw: dict,
        rot_frame_id: int,
        time_steps: int,
        traj_rot: np.ndarray = np.eye(3),
        q: np.ndarray = None,
        qdot: np.ndarray = None,
        rest_pos: np.ndarray = None,
        vel_pen: np.ndarray = None,
        control_dim: int = 6,
        show_in_console: bool = True,
    ):
        super(ExperimentSetpoint, self).__init__()
        self.x_frame_id = x_frame_id

        if show_in_console:
            print(
                "Experiment: "
                + "\nDirection: {}".format(target_raw["direction"])
                + "\nOrientation: {}".format(target_raw["orientation"])
                + "\nRadius: {}".format(target_raw["radius"])
                + "\nStart_point: {}".format(start_point)
                + "\nx_frame_id: {}".format(x_frame_id)
                + "\nrot_frame_id: {}".format(rot_frame_id)
            )
        self.traj_x = self._compute_traj_x(start_point, target_raw, time_steps)
        self._target_str = self._get_target_str(target_raw)
        self.time_steps = time_steps

        self._target_raw = target_raw

        self.rot_frame_id = rot_frame_id
        self.traj_rot = time_steps * [traj_rot]
        self.q = q
        self.qdot = qdot
        self.rest_pos = rest_pos
        self.vel_pen = vel_pen
        self.control_dim = control_dim

    def _compute_traj_x(
        self, start_point: np.ndarray, target_raw: dict, time_steps: int
    ) -> np.ndarray:
        return np.tile(
            start_point + self._compute_add_target(target_raw), (time_steps, 1)
        )

    def _compute_add_target(self, target_raw: dict):
        """Direction, Orientation, Radius"""
        radius = target_raw["radius"]
        factor = target_raw["orientation"]
        direction = target_raw["direction"]

        dimension = {"x": 0, "y": 1, "z": 2}
        eye_vector = np.zeros(3)
        eye_vector[dimension[direction]] = 1
        add_target = factor * radius * eye_vector
        return add_target

    def _get_target_str(self, target_raw: dict):
        return "{}_{}_{}".format(
            target_raw["direction"],
            "neg" if target_raw["orientation"] < 0 else "pos",
            target_raw["radius"],
        )

    # file saving methods inherited from the base class
    def __str__(self):
        return "{}_setpoint_xID_{}_rotID_{}".format(
            self._target_str, self.x_frame_id, self.rot_frame_id
        )

    def save_to_metadata(self) -> dict:
        return {
            "experiment": "setpoint",
            "x_frame_id": self.x_frame_id,
            "rot_frame_id": self.rot_frame_id,
            "orientation": self._target_raw["orientation"],
            "radius": self._target_raw["radius"],
            "dir": self._target_raw["direction"],
        }

    def save_to_file(self) -> dict:
        return {}


class EvalExperimentSetpoint:
    def __init__(self, time_horizon: int, start_point: np.ndarray):
        self.time_horizon = time_horizon
        self.start_point = start_point

    def get_target(self, radius: float, orientatation: int, dir: str):
        dimension = {"x": 0, "y": 1, "z": 2}
        eye_vector = np.zeros(3)
        eye_vector[dimension[dir]] = 1

        add_target = orientatation * radius * eye_vector
        return np.tile(self.start_point + add_target, (self.time_horizon, 1))
