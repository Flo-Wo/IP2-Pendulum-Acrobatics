import logging
import os
import pathlib
from copy import deepcopy

import crocoddyl
import numpy as np
import pinocchio

from utils.config.action_model_factory import DifferentialActionModelFactory
from utils.config.config_ip import ConfigWAM
from utils.costs.control_bounds import ControlBounds
from utils.costs.penalty.penalty_factory import PenaltyFactoryMPC
from utils.experiments.setpoint.experiment_setpoint import ExperimentSetpoint
from utils.experiments.trajectory.experiment_trajectory import ExperimentTrajectory
from utils.experiments.trajectory.residual_models import integrated_cost_models
from utils.solver.DDP.ddp_solver import ddp

if __name__ == "__main__":
    model = ConfigWAM(
        model_name="rotated",
        pendulum_len=0.374,
        use_friction_action_model=True,
    )
    pin_model = model.pin_model
    mj_model = model.mj_model

    # define experiment
    num_cost_models = 3000
    params_dict = dict(
        x_frame_id=deepcopy(model.get_pend_pole_tip_id()),
        start_point=deepcopy(model.get_pend_end_world()),
        rot_frame_id=deepcopy(model.get_pend_pole_tip_id()),
        time_steps=deepcopy(num_cost_models),
        rest_pos=deepcopy(model.q_config),
        target_raw=dict(direction="x", orientation=-1, radius=0.0),
    )
    experiment_task_space = ExperimentSetpoint(**params_dict)

    state = crocoddyl.StateMultibody(pin_model)
    actuation_model = crocoddyl.ActuationModelFull(state)

    action_model = DifferentialActionModelFactory.get_action_model(
        use_friction_action_model=True,
        armature=mj_model.dof_armature.copy(),
        damping=mj_model.dof_damping.copy(),
        coulomb_friction=np.zeros_like(mj_model.dof_frictionloss),
        gravity_comp_idxs=[],
    )

    dt_const = 0.002
    ctrl_bounds = ControlBounds(
        lower_bound=-pin_model.effortLimit, upper_bound=pin_model.effortLimit
    )

    # PART 0: compute the ddp cost_models
    penalty_ddp = PenaltyFactoryMPC.penalty_mpc("stabilization")(
        angles_radian=np.array([0.0] * num_cost_models),
        q_pen_weights=np.concatenate((np.ones(6), np.zeros(6))),
        v_pen_weights=np.concatenate((np.zeros(6), np.ones(6))),
    )

    # PART 1: BUILD COST MODELS FOR THE TASK SPACE OC
    # build intermediate/running cost models
    cost_models_ddp = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty_ddp,
        experiment_task_space,
        dt_const,
        ctrl_bounds,
    )
    # build terminal cost models
    terminal_costs_ddp = crocoddyl.CostModelSum(state)
    terminal_node_ddp = crocoddyl.IntegratedActionModelEuler(
        action_model(state, actuation_model, terminal_costs_ddp), 0.0
    )
    # terminal_node_ddp = cost_models_ddp[-1]

    # TODO: same number of iterations
    # TODO: correct input state and number of cost model
    folder = pathlib.Path(__file__).parent.parent.parent / "data/test_suite"
    state_list = np.load(os.path.join(folder, "test_observer_states.npy"))
    torque_list = np.load(os.path.join(folder, "test_observer_torques.npy"))

    mpc_horizon = 20
    for time_idx in range(0, 20, 10):
        state = state_list[time_idx, :]
        active_cost_models = cost_models_ddp[time_idx : time_idx + mpc_horizon]

        # PART 2: SOLVE THE TASK SPACE OC VIA CROCODDYL'S FDDP SOLVER
        logging.info("Starting DDP preplanning.")
        print("\n\nTime idx = ", time_idx)
        print("input state =")
        print(state)
        ddp_res, ddp_time = ddp(
            active_cost_models,
            terminal_node_ddp,
            q=state[:6],
            qdot=state[6:],
            solver_max_iter=20,
            show_logs=True,
        )
        print("Result: ")
        print("shape = ", ddp_res.u.shape)
        print("torques = ")
        print(ddp_res.u)
        logging.info("DDP preplanning is solved.")
