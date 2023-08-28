from typing import List, Tuple

import numpy as np

from utils.config.config_ip import ConfigWAM
from utils.experiments.setpoint.experiment_setpoint import ExperimentSetpoint
from utils.pin_helper.angles import angles_traj
from utils.pin_helper.world_frame import task_space_traj
from utils.solver.MPC.mpc_results import MPCResults


def table_header(
    mpc_horizon: float = 10,
    factor_int_time: int = 4,
    solver_max_iter: int = 20,
    n_fully_integrated_steps: int = 1,
    n_offset_500_Hz_steps: int = 0,
    beta: float = 0.5,
) -> str:
    return "\n\nMPC Horizon: {}, factor_int_time = {}, solver_max_iter = {}".format(
        mpc_horizon, factor_int_time, solver_max_iter
    ) + "\nn_fully_integrated_steps = {}, n_offset_500_Hz_steps = {}, beta = {}".format(
        n_fully_integrated_steps, n_offset_500_Hz_steps, beta
    )


def show_error(
    mpc_res: MPCResults,
    model: ConfigWAM,
    experiment: ExperimentSetpoint,
    mpc_time: float,
    comp_time: List[float],
    end_err: int = None,
    mj_time_step: float = 0.002,
) -> Tuple[float, float, float, float]:
    # necessary computation
    q_list = np.array([x[:6] for x in mpc_res.states[1:]])
    x_task = task_space_traj(model.pin_model, model.pin_data, q_list, 24)
    angles_task = angles_traj(
        model.pin_model, model.pin_data, q_list, 24, 22, radian=False
    )

    print("=========================")
    print(
        "TOOK {}s, SHOW ERROR ALONG THE LAST, N_POINTS {}".format(
            mpc_time, end_err if end_err is not None else "all"
        )
    )
    print("POSITIONAL")
    pos_err, pos_avg_err = _compute_errors(
        goal=experiment.traj_x, exp_traj=x_task, end_err=end_err
    )
    print("SUM-Error: {} \nAvg-Error: {}".format(pos_err, pos_avg_err))

    print("ROTATIONAL")
    angle_err, angle_avg_err = _compute_angle_error(angles=angles_task, end_err=end_err)
    print("SUM-Error: {} \nAvg-Error: {}".format(angle_err, angle_avg_err))
    print("TIME")
    mean_time, median_time, max_time = _compute_time_kpis(comp_time=comp_time)
    print("Mean: {} \nMedian: {}\nMax: {}".format(mean_time, median_time, max_time))
    print("STEPS")
    print(
        "Mean: {} \nMedian: {}\nMax: {}".format(
            mean_time / mj_time_step,
            median_time / mj_time_step,
            max_time / mj_time_step,
        )
    )
    print("=========================")

    return pos_err, pos_avg_err, angle_err, angle_avg_err


def _compute_time_kpis(comp_time: List[float]) -> Tuple[float, float, float]:
    """Mean, Median, Max Time in seconds"""
    return np.mean(comp_time), np.median(comp_time), np.max(comp_time)


def _compute_errors(
    goal: np.ndarray, exp_traj: np.ndarray, end_err: int = None
) -> Tuple[float, float]:
    """Compute the l2-error and the average l2-error along the entire trajectory.

    Parameters
    ----------
    goal : np.ndarray
        Goal trajectory.
    exp_traj : np.ndarray
        Real trajectory.
    end_error : int, optional
        Compuete e.g. the error for the last 300 nodes.

    Returns
    -------
    Tuple[float, float]
        l2-error and average l2-error.
    """
    if end_err is not None:
        goal = goal[-end_err:, :]
        exp_traj = exp_traj[-end_err:, :]

    err = np.sum(np.linalg.norm(goal - exp_traj, axis=1))
    avg_err = np.average(np.linalg.norm(goal - exp_traj, axis=1))
    return err, avg_err


def _compute_angle_error(
    angles: np.ndarray, end_err: int = None
) -> Tuple[float, float]:
    if end_err is not None:
        angles = angles[-end_err:]
    err = np.sum(np.abs(angles))
    avg_err = np.average(np.abs(angles))
    return err, avg_err
