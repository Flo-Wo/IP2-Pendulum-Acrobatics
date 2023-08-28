import numpy as np

from utils.experiments import ExperimentBase


class ExperimentTrajectory(ExperimentBase):
    def __init__(
        self,
        x_frame_id: int = 24,
        traj_x: np.ndarray = None,
        rot_frame_id: int = 24,
        traj_rot: np.ndarray = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        q: np.ndarray = None,
        qdot: np.ndarray = None,
        rest_pos: np.ndarray = None,
        vel_pen: np.ndarray = None,
        control_dim: int = 6,
    ):
        super(ExperimentTrajectory, self).__init__()
        self.x_frame_id = x_frame_id
        self.traj_x = traj_x
        self.rot_frame_id = rot_frame_id
        self.traj_rot = traj_rot
        self.q = q
        self.qdot = qdot
        self.rest_pos = rest_pos
        self.vel_pen = vel_pen
        self.control_dim = control_dim

    # file saving methods inherited from the base class
    def __str__(self):
        return "trajectory_xID_{}_rotID_{}".format(self.x_frame_id, self.rot_frame_id)

    def save_to_metadata(self) -> dict:
        return {
            "experiment": "trajectory",
            "x_frame_id": self.x_frame_id,
            "rot_frame_id": self.rot_frame_id,
        }

    def save_to_file(self) -> dict:
        return {"goal": self.traj_x}
