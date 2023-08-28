import crocoddyl
import numpy as np

from utils.costs.penalty_base import Penalty
from utils.experiments.trajectory import PenaltyTrajectory

q_pen_weights_default = np.concatenate((np.ones(6), np.zeros(6)))
v_pen_weights_default = np.concatenate((np.zeros(6), np.ones(6)))


class PenaltyFactoryDDP:
    @staticmethod
    def penalty_ddp(penalty_type: str = "all"):
        return get_penalty_ddp


class PenaltyFactoryMPC:
    @staticmethod
    def penalty_mpc(penalty_type: str = "setpoint"):
        if penalty_type == "stabilization":
            return _get_stabilization_penalty_mpc
        elif penalty_type == "setpoint":
            return _get_setpoint_penalty_mpc
        elif penalty_type == "circle":
            return _get_penalty_mpc_circle
        elif penalty_type == "spiral":
            return _get_penalty_mpc_spiral
        else:
            raise NotImplementedError("Wrong penalty type.")


def get_penalty_ddp(
    q_pen_weights: np.ndarray = q_pen_weights_default,
    v_pen_weights: np.ndarray = v_pen_weights_default,
) -> PenaltyTrajectory:
    return PenaltyTrajectory(
        u_pen=Penalty(1e-2),
        x_pen=Penalty(1e3),
        q_pen=Penalty(1e-2),
        v_pen=Penalty(
            0.5,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        # rot_pen=Penalty(
        #     1e5,
        #     act_func=crocoddyl.ActivationModelWeightedQuad,
        #     act_func_params=np.array([1.0, 1.0, 0.0]),
        # ),
        prefix="ddp_",
    )


def _get_stabilization_penalty_mpc(
    angles_radian: np.ndarray,
    q_pen_weights: np.ndarray = q_pen_weights_default,
    v_pen_weights: np.ndarray = v_pen_weights_default,
    q_qdot_list: np.ndarray = None,
    beta: float = 0.5,
) -> PenaltyTrajectory:
    """Penalty task space weights for the rotated wam for the online mpc given the
    precomputed trajectory."""
    return PenaltyTrajectory(
        u_pen=Penalty(1e-1),
        # u_pen=Penalty(1e0),
        x_pen=Penalty(6e5),  # 6e5, 7e4 for setpoint tracking
        v_pen=Penalty(
            1e0,  # 1e2,  # 3e-2 for setpoint tracking
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        q_pen=Penalty(
            1e2,  # 8e2,
            # act_func=crocoddyl.ActivationModelWeightedQuad,
            # act_func_params=np.concatenate((np.ones(4), np.zeros(8))),
        ),
        rot_pen=Penalty(
            1e6,  # 1e7
            act_func=crocoddyl.ActivationModelWeightedQuadraticBarrier,
            act_func_params=[
                (
                    crocoddyl.ActivationBounds(
                        np.array([-2 * angle, -2 * angle, 4 * np.pi]),
                        np.array([2 * angle, 2 * angle, 4 * np.pi]),
                        beta,
                    ),
                    np.array([1.0, 1.0, 0.0]),
                )
                for angle in angles_radian
            ],
        ),
        prefix="mpc_",
    )


def _get_setpoint_penalty_mpc(
    angles_radian: np.ndarray,
    q_pen_weights: np.ndarray = q_pen_weights_default,
    v_pen_weights: np.ndarray = v_pen_weights_default,
    q_qdot_list: np.ndarray = None,
    beta: float = 0.5,
) -> PenaltyTrajectory:
    """Penalty task space weights for the rotated wam for the online mpc given the
    precomputed trajectory."""
    return PenaltyTrajectory(
        u_pen=Penalty(1e-2),
        x_pen=Penalty(7e4),  # 7e4 for setpoint tracking
        v_pen=Penalty(
            3e-2,  # 3e-2 for setpoint tracking
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        q_pen=Penalty(
            5e-3,  # 5e-1,  # 5e-3
        ),
        rot_pen=Penalty(
            1e5,  # 1e5 for setpoint tracking
            act_func=crocoddyl.ActivationModelWeightedQuadraticBarrier,
            act_func_params=[
                (
                    crocoddyl.ActivationBounds(
                        np.array([-2 * angle, -2 * angle, 4 * np.pi]),
                        np.array([2 * angle, 2 * angle, 4 * np.pi]),
                        beta,
                    ),
                    np.array([1.0, 1.0, 0.0]),
                )
                for angle in angles_radian
            ],
        ),
        prefix="mpc_",
    )


def _get_penalty_mpc_spiral(
    angles_radian: np.ndarray,
    q_pen_weights: np.ndarray = q_pen_weights_default,
    v_pen_weights: np.ndarray = v_pen_weights_default,
    q_qdot_list: np.ndarray = None,
    beta: float = 0.5,
):
    return PenaltyTrajectory(
        u_pen=Penalty(1e-2),  # for the circle
        x_pen=Penalty(2e4),  # circle: 8e4
        v_pen=Penalty(
            3e-2,
            # circle: 3e-2
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        q_pen=Penalty(
            # 9e-1,
            1e-2,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=q_pen_weights,
        ),
        rot_pen=Penalty(
            7e5,  # circle: 4e5
            act_func=crocoddyl.ActivationModelWeightedQuadraticBarrier,
            act_func_params=[
                (
                    crocoddyl.ActivationBounds(
                        np.array([-2 * angle, -2 * angle, 4 * np.pi]),
                        np.array([2 * angle, 2 * angle, 4 * np.pi]),
                        beta,
                    ),
                    np.array([1.0, 1.0, 0.0]),
                )
                for angle in angles_radian
            ],
        ),
        prefix="mpc_",
    )


def _get_penalty_mpc_circle(
    angles_radian: np.ndarray,
    q_pen_weights: np.ndarray = q_pen_weights_default,
    v_pen_weights: np.ndarray = v_pen_weights_default,
    q_qdot_list: np.ndarray = None,
    beta: float = 0.5,
):
    return PenaltyTrajectory(
        u_pen=Penalty(1e-2),  # for the circle
        x_pen=Penalty(8e4),  # circle: 8e4
        v_pen=Penalty(
            3e-2,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        q_pen=Penalty(
            # 9e-1,
            1e-2,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=q_pen_weights,
        ),
        rot_pen=Penalty(
            4e5,  # circle: 4e5
            act_func=crocoddyl.ActivationModelWeightedQuadraticBarrier,
            act_func_params=[
                (
                    crocoddyl.ActivationBounds(
                        np.array([-2 * angle, -2 * angle, 4 * np.pi]),
                        np.array([2 * angle, 2 * angle, 4 * np.pi]),
                        beta,
                    ),
                    np.array([1.0, 1.0, 0.0]),
                )
                for angle in angles_radian
            ],
        ),
        prefix="mpc_",
    )
