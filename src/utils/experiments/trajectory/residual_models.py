import logging
from copy import copy
from typing import List, Union

import crocoddyl

from utils.costs import (
    ControlBounds,
    ResidualBase,
    ResidualFactory,
    integrate_residuals,
)
from utils.experiments.trajectory import ExperimentTrajectory, PenaltyTrajectory


class EmptyResidualError(Exception):
    def __init__(self, msg: str):
        super(EmptyResidualError, self).__init__(msg)


def integrated_cost_models(
    state: crocoddyl.StateMultibody,
    action_model: crocoddyl.DifferentialActionModelAbstract,
    actuation_model,
    penalty: PenaltyTrajectory,
    experiment: ExperimentTrajectory,
    dt_const: Union[float, List[float]],
    ctrl_bounds: ControlBounds = None,
):
    residual_models = _build_residual_models(state, penalty, experiment)
    if ctrl_bounds is None:
        ctrl_bounds = ControlBounds()

    return integrate_residuals(
        state,
        action_model,
        actuation_model,
        dt_const,
        residual_models,
        ctrl_bounds,
    )


def _is_logbarrier(penalty: PenaltyTrajectory):
    try:
        return isinstance(
            penalty.rot_pen.act_func(*penalty.rot_pen.act_func_params),
            crocoddyl.ActivationModelLogBarrier,
        )
    except:
        return False


def _build_residual_models(
    state: crocoddyl.StateMultibody,
    penalty: PenaltyTrajectory,
    experiment: ExperimentTrajectory,
    symmetric_log_barrier: bool = True,
) -> List[ResidualBase]:
    """Create all possible residuals for a state, penaltyuration,
    frame ids and given trajectories.


    Parameters
    ----------
    state : crocoddyl.StateMultibody
        Crocoddyl state.
    penalty : penaltyTrajectory
        Penalty with all the penalty terms.
    dim_control : int
        Dimensionality of the torques.
    x_frame_id : int
        Id of the frame we want to follow a trajectory in the state space.
    rot_frame_id : int
        Id of the pendulum's tip.
    traj_x : np.ndarray
        Trajectory for the state space, gets tracked with by the frame
        with the id x_frame_id.
    traj_rot : np.ndarray
        Rotation trajectory for the frame with the id rot_frame_id.
    q : np.ndarray
        Target in the state space.
    qdot : np.ndarray
        Target velocity in the state space, gets tracked together
        with q.
    rest_pos : np.ndarray
        Optional resting position the robot should stay close to. Be careful,
        v is filled with zeros and thus needs to be cancelled out with weights
        in the activation function.
    vel_pen : np.ndarray
        Optional penalties on the velocities. Be careful, q is filled with
        zeros and thus needs to be cancelled out with weights in the activation
        function.

    Returns
    -------
    List[ResidualBase]
        List of Residuals, ready to be integrated.
    """
    res_factory = ResidualFactory(state, dim_control=experiment.control_dim)
    res_list = []
    if penalty.u_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "u", "control", penalty.u_pen, dim_residual=experiment.control_dim
            )
        )

    if penalty.state_bound_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "state_bound", "state_bound", penalty.state_bound_pen
            )
        )

    if penalty.x_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "x",
                "task_space",
                penalty.x_pen,
                **{"traj_x": experiment.traj_x, "frame_id": experiment.x_frame_id},
            )
        )
    if penalty.rot_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "x_rot",
                "rotation",
                penalty.rot_pen,
                **{
                    "traj_rot": experiment.traj_rot,
                    "frame_id": experiment.rot_frame_id,
                },
            )
        )
        # check for the LogBarrier case --> want upper and lower bound constraints
        if _is_logbarrier(penalty) and symmetric_log_barrier:
            logging.info(
                "LogBarrier for Rotation: also using negative weights. Disable by using symmetric_log_barrier=False."
            )
            rot_pen_copy = copy(penalty.rot_pen)
            negative_weights = list(
                rot_pen_copy.act_func_params
            )  # tuples are immutable
            # Arguments are: weights, bounds, damping
            negative_weights[0] *= -1
            rot_pen_copy.act_func_params = tuple(negative_weights)
            res_list.append(
                res_factory.compute_residual(
                    "x_rot_neg_weights",
                    "rotation",
                    penalty.rot_pen,
                    **{
                        "traj_rot": experiment.traj_rot,
                        "frame_id": experiment.rot_frame_id,
                    },
                )
            )
    if penalty.state_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "state_v_and_q",
                "state_space",
                penalty.state_pen,
                **{
                    "q": experiment.q,
                    "v": experiment.qdot,
                },
            )
        )
    if penalty.q_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "q_pen",
                "q",
                penalty.q_pen,
                **{"q": experiment.rest_pos},
            )
        )
    if penalty.v_pen is not None:
        res_list.append(
            res_factory.compute_residual(
                "v_pen",
                "v",
                penalty.v_pen,
                **{"v": experiment.vel_pen},
            )
        )

    if len(res_list) == 0:
        raise EmptyResidualError("All penalty terms are None.")
    return res_list
