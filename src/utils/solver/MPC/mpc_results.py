from typing import List

import numpy as np

from utils.solver.solver_results_base import SolverResultsBase


class MPCResults(SolverResultsBase):
    """
    Class to cache the results of an mpc optimization problem:
    - xs: crocoddyl list with (q,q_dot) pairs, saved only the first state
    of each iteration
    - us: crocoddyl list with (u) torques, only the first torque which is
    applied is saved
    - mj_xs: list with (q,q_dot) pairs of the mujoco simulation after
    applying the torque us
    - mj_sensordata: list with sensordata for each iteration
    """

    def __init__(
        self,
        ddp_x: np.ndarray,
        u: np.ndarray,
        state: np.ndarray,
        mj_sensor: np.ndarray,
        solved: bool,
        factor_integration_time: int,
        mpc_horizon: int,
        solver_comp_times: List[float] = [],
    ):
        self.ddp_x = ddp_x[None, :]
        self.u = u[None, :]
        self.states = state[None, :]
        self.observed_states = state[None, :]
        self.x, self.x_pend_half, self.x_pend_beg, self.rot = self._extract_mj_data(
            mj_sensor
        )
        self.solved = [solved]

        self.factor_integration_time = factor_integration_time
        self.mpc_horizon = mpc_horizon
        self.solver_comp_times = solver_comp_times

    def add_data(self, ddp_x, u, state, mj_sensor, solved, observer_state=None):
        self.ddp_x = np.vstack((self.ddp_x, ddp_x[None, :]))
        self.u = np.vstack((self.u, u[None, :]))

        self.states = np.vstack((self.states, state[None, :]))

        if observer_state is not None:
            self.observed_states = np.vstack(
                (self.observed_states, observer_state[None, :])
            )

        # extract the data
        x, x_half, x_pend_beg, rot = self._extract_mj_data(mj_sensor)

        self.x = np.vstack((self.x, x))
        self.x_pend_half = np.vstack((self.x_pend_half, x_half))
        self.x_pend_beg = np.vstack((self.x_pend_beg, x_pend_beg))
        self.rot = np.vstack((self.rot, rot))

        self.solved.append(solved)

    def add_computation_time(self, comp_time: float) -> None:
        self.solver_comp_times.append(comp_time)

    def _extract_mj_data(self, mj_sensor: np.ndarray):
        """Return: x, rot, x_pend_beg"""
        mj_copy = mj_sensor.copy()
        return (
            mj_copy[0:3][None, :],
            mj_copy[3:6][None, :],
            mj_copy[6:9][None, :],
            mj_copy[9:13][None, :],
        )

    def __str__(self) -> str:
        return "MPC_factor_int_time_{}_horizon_{}".format(
            self.factor_integration_time, self.mpc_horizon
        )

    def save_to_file(self) -> dict:
        return {
            "x": self.x,
            "u": self.u,
            "x_rot": self.rot,
            "x_pend_beg": self.x_pend_beg,
        }

    def save_to_metadata(self) -> dict:
        return {
            "solver": "MPC",
            "factor_integration_time": self.factor_integration_time,
            "mpc_horizon": self.mpc_horizon,
        }
