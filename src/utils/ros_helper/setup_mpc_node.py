from copy import copy
from typing import List, Union

import crocoddyl
import matplotlib.pyplot as plt
import numpy as np

from utils.config import ConfigWAM
from utils.costs import ControlBounds
from utils.costs.penalty import PenaltyFactoryMPC
from utils.experiments import ExperimentBase
from utils.experiments.trajectory import (
    ExperimentTrajectory,
    PenaltyTrajectory,
    integrated_cost_models,
)
from utils.pin_helper import angles_traj, task_space_traj
from utils.solver.DDP import ddp, ddp_results
from utils.visualize.plots import plot_single_trajectory


def task_space_costs_from_preplanning(
    model: ConfigWAM,
    state: crocoddyl.StateMultibody,
    action_model: crocoddyl.ActuationModelFull,
    actuation_model: crocoddyl.DifferentialActionModelFreeFwdDynamics,
    ctrl_bounds: ControlBounds,
    ddp_res: ddp_results,
    dt_const: Union[float, List[float]],
    q_pen_weights: np.ndarray,
    v_pen_weights: np.ndarray,
    t_total: int = 3000,
    t_crop: int = None,
    beta: float = 0.5,
    control_dim: int = 6,
    x_frame_id: int = 24,
    rot_frame_id: int = 24,
) -> tuple[
    List[crocoddyl.IntegratedActionModelEuler], crocoddyl.IntegratedActionModelEuler
]:
    """Compute the task space trajectory from the precomputed optimal trajectory."""

    pin_model = copy(model.pin_model)
    pin_data = copy(model.pin_data)

    q_list = np.array([x[:control_dim] for x in ddp_res.state[1:]])

    # NOTE: CLIPPING
    x_task_raw = task_space_traj(pin_model, pin_data, q_list, x_frame_id)
    if t_crop is not None:
        x_task_cropped = np.concatenate(
            (
                x_task_raw[:t_crop, :],
                np.tile(x_task_raw[t_crop, :], (t_total - t_crop, 1)),
            )
        )
    else:
        x_task_cropped = x_task_raw

    angles_radian = angles_traj(
        pin_model,
        pin_data,
        q_list,
        upper_frame_id=x_frame_id,
        lower_frame_id=x_frame_id - 2,
        radian=True,
    )
    if t_crop is not None:
        angles_radian_cropped = np.concatenate(
            (
                angles_radian[:t_crop],
                np.tile(angles_radian[t_crop], t_total - t_crop),
            )
        )
    else:
        angles_radian_cropped = angles_radian
    # PART 3: BUILD COST MODELS FOR THE STATE SPACE OC
    experiment_mpc = ExperimentTrajectory(
        x_frame_id=x_frame_id,
        rot_frame_id=rot_frame_id,
        traj_x=x_task_cropped,
        traj_rot=t_total * [np.eye(3)],
        rest_pos=model.q_config,
        control_dim=control_dim,
    )
    get_penalty_mpc = PenaltyFactoryMPC.penalty_mpc("setpoint")

    penalty_mpc = get_penalty_mpc(
        angles_radian_cropped,
        q_pen_weights=q_pen_weights,
        v_pen_weights=v_pen_weights,
        beta=beta,
    )

    cost_models_mpc = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty_mpc,
        experiment_mpc,
        dt_const,
        ctrl_bounds,
    )

    terminal_cost_node_mpc = crocoddyl.CostModelSum(state)
    terminal_cost_model_mpc = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation_model, terminal_cost_node_mpc
        ),
        0,
    )
    return cost_models_mpc, terminal_cost_model_mpc


def ddp_optimal_task_space(
    cost_models_ddp: List[crocoddyl.IntegratedActionModelEuler],
    terminal_cost_model_ddp: crocoddyl.IntegratedActionModelEuler,
    q: np.ndarray,
    qdot: np.ndarray,
    ddp_max_iter: int = 1000,
    show_logs: bool = False,
) -> ddp_results:
    """Compute the optimal solution via the fddp solver."""

    ddp_res, ddp_time = ddp(
        cost_models_ddp,
        terminal_cost_model_ddp,
        q=q,
        qdot=qdot,
        solver_max_iter=ddp_max_iter,
        show_logs=show_logs,
    )
    return ddp_res


def setup_ddp_cost_models(
    state: crocoddyl.StateMultibody,
    action_model: crocoddyl.ActuationModelFull,
    actuation_model: crocoddyl.DifferentialActionModelFreeFwdDynamics,
    penalty_ddp: PenaltyTrajectory,
    experiment: ExperimentBase,
    dt_const: Union[float, List[float]],
    ctrl_bounds: ControlBounds,
) -> tuple[
    List[crocoddyl.IntegratedActionModelEuler], crocoddyl.IntegratedActionModelEuler
]:
    """Setup the cost models for the solution precomputation via the fddp solver."""
    cost_models_ddp = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty_ddp,
        experiment,
        dt_const,
        ctrl_bounds,
    )

    # build terminal cost models
    terminal_cost_node_ddp = crocoddyl.CostModelSum(state)
    terminal_cost_model_ddp = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation_model, terminal_cost_node_ddp
        ),
        0,
    )
    return cost_models_ddp, terminal_cost_model_ddp
