import logging
from copy import copy
from typing import Tuple

import crocoddyl
import mujoco
import numpy as np
import pinocchio as pin
from matplotlib import pyplot as plt

from utils.config import ConfigModelBase, DifferentialActionModelFactory
from utils.config.action_model_gravity_compensation import FrictionModel
from utils.costs import ControlBounds, Penalty
from utils.data_handling import AddData, DataSaver
from utils.enums import CompTimeNames
from utils.experiments.trajectory import ExperimentTrajectory, integrated_cost_models
from utils.pin_helper import angles_traj, task_space_traj
from utils.solver import ddp, mpc
from utils.solver.DDP.ddp_results import DDPResults
from utils.solver.MPC.mpc_results import MPCResults
from utils.visualize.plots.plot_traj import plot_single_trajectory, plot_trajectory
from utils.visualize.vis_mujoco.loop_trajectory import loop_mj_vis


def mpc_state_space(
    model: ConfigModelBase,
    experiment_task_space: ExperimentTrajectory,
    get_penalty_ddp: callable,
    get_penalty_mpc: callable,
    max_iter_ddp: int = 1000,
    max_iter_mpc: int = 100,
    mpc_horizon: int = 10,
    factor_int_time: int = 1,
    data_saver: DataSaver = None,
    beta: float = 0.5,
    t_crop: int = 1500,
    t_total: int = 3000,
    show_logs: bool = False,
    continue_end: str = "stationary",
    n_fully_integrated_steps: int = 1,
    n_offset_500_Hz_steps: int = 0,
    q_pen_weights: np.ndarray = np.concatenate((np.ones(6), np.zeros(6))),
    v_pen_weights: np.ndarray = np.concatenate((np.zeros(6), np.ones(6))),
    use_state_observer: bool = False,
    use_wam_low_pass_filter: bool = True,
    use_pend_low_pass_filter: bool = True,
    noise_on_q_observation: float = 0.0,
    noise_on_pendulum_observation: float = 0.0,
    pendulum_observation_updates_with_125_Hz: bool = False,
    use_forward_predictor: bool = False,
    use_friction_action_model: bool = False,
) -> Tuple[MPCResults, DDPResults, float]:
    # we need copy here, otherwise the results are incorrect due to side effects
    mj_model = copy(model.mj_model)
    mj_data = copy(model.mj_data)
    pin_model = copy(model.pin_model)
    pin_data = copy(model.pin_data)

    state = crocoddyl.StateMultibody(pin_model)
    actuation_model = crocoddyl.ActuationModelFull(state)

    action_model = DifferentialActionModelFactory.get_action_model(
        use_friction_action_model=use_friction_action_model,
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
    penalty_ddp = get_penalty_ddp(
        q_pen_weights=q_pen_weights,
        v_pen_weights=v_pen_weights,
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

    # PART 2: SOLVE THE TASK SPACE OC VIA CROCODDYL'S FDDP SOLVER
    logging.info("Starting DDP preplanning.")
    ddp_res, ddp_time = ddp(
        cost_models_ddp,
        terminal_node_ddp,
        model.q_config,
        qdot=np.zeros(state.nv),
        solver_max_iter=max_iter_ddp,
        show_logs=show_logs,
    )
    logging.info("DDP preplanning is solved.")
    # Test the dynamics errors in MuJoCo
    data = model.pin_model.createData()
    mujoco_accs = []
    pin_accs = []
    for i in range(ddp_res.state.shape[0] - 1):
        q = ddp_res.state[i, : mj_model.nq]
        dq = ddp_res.state[i, mj_model.nq :]

        mj_data.qpos[:] = q
        mj_data.qvel[:] = dq
        mj_data.ctrl[:] = np.zeros_like(mj_data.ctrl)
        # mj_data.ctrl[:] = ddp_res.u[i, :]
        mujoco.mj_forward(mj_model, mj_data)
        mujoco_accs.append(np.copy(mj_data.qacc))

        pin.computeAllTerms(model.pin_model, data, q, dq)
        # Pinocchio does not model damping in its algorithms so we need to do this ourselves
        # tau = tau - model.mj_model.damping * dq
        b = pin.rnea(model.pin_model, data, q, dq, np.zeros(6))
        pin_accs.append(np.linalg.solve(data.M, np.zeros(6) - b))

    mujoco_accs = np.array(mujoco_accs)
    pin_accs = np.array(pin_accs)
    print(
        f"Maximum deviation in acceleration along planned trajecotry: {np.max(np.linalg.norm(mujoco_accs - pin_accs, axis=-1))}"
    )

    # NOTE: CLIPPING
    q_list = np.array([x[:6] for x in ddp_res.state[1:]])
    x_task_raw = task_space_traj(
        pin_model, pin_data, q_list, experiment_task_space.x_frame_id
    )
    q_dot_list = np.array([x[6:] for x in ddp_res.state[1:]])

    # plot_trajectory(
    #     goal=experiment_task_space.traj_x,
    #     experiments={"ddp": x_task_raw},
    # )
    # plot_single_traj(traj=ddp_res.u)
    # plt.show()
    # loop_mj_vis(model, q=q_list, q_dot=q_dot_list, num_motors=4)
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
        experiment_task_space.x_frame_id,
        pin_model.getFrameId("links/pendulum/rotating_x_axis"),
        radian=True,
    )
    if t_crop is not None:
        angles_radian_cropped = np.concatenate(
            (angles_radian[:t_crop], np.tile(angles_radian[t_crop], t_total - t_crop))
        )
    else:
        angles_radian_cropped = angles_radian

    print("================")
    print("Uniform cost models")
    x_task_cropped = experiment_task_space.traj_x
    angles_radian_cropped = np.zeros(*np.shape(angles_radian_cropped))
    print("================")

    # PART 3: BUILD COST MODELS FOR THE STATE SPACE OC
    experiment_mpc = ExperimentTrajectory(
        x_frame_id=experiment_task_space.x_frame_id,
        rot_frame_id=experiment_task_space.rot_frame_id,
        traj_x=x_task_cropped,
        traj_rot=t_total * [np.eye(3)],
        rest_pos=model.q_config,
    )

    penalty_mpc = get_penalty_mpc(
        angles_radian=angles_radian_cropped,
        beta=beta,
        q_pen_weights=q_pen_weights,
        v_pen_weights=v_pen_weights,
        q_qdot_list=[x for x in ddp_res.state[1:]],
    )

    # TODO: currently the ddp works with the finest time and the mpc uses a more
    # coarse resolution
    dt_const_mpc = factor_int_time * dt_const
    cost_models_mpc = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty_mpc,
        experiment_mpc,
        dt_const_mpc,
        ctrl_bounds,
    )

    terminal_costs_mpc = crocoddyl.CostModelSum(state)
    terminal_node_mpc = crocoddyl.IntegratedActionModelEuler(
        action_model(state, actuation_model, terminal_costs_mpc), 0.0
    )
    # terminal_node_mpc = cost_models_mpc[-1]

    # PART 4: SOLVE THE PROBLEM WITH OUR ONLINE MPC

    # reset the robot and set it to its start position
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:] = ddp_res.state[0, : mj_model.nq]
    mj_data.qvel[:] = ddp_res.state[0, mj_model.nq :]
    # mujoco.mj_step(mj_model, mj_data)

    mpc_res, mpc_time = mpc(
        mujoco_model=mj_model,
        mujoco_data=mj_data,
        pin_model=pin_model,
        pin_data=pin_data,
        int_cost_models=cost_models_mpc,
        terminal_model=terminal_node_mpc,
        mpc_horizon=mpc_horizon,
        solver_max_iter=max_iter_mpc,
        show_logs=show_logs,
        factor_integration_time=factor_int_time,
        continue_end=continue_end,
        n_fully_integrated_steps=n_fully_integrated_steps,
        n_offset_500_Hz_steps=n_offset_500_Hz_steps,
        use_state_observer=use_state_observer,
        use_wam_low_pass_filter=use_wam_low_pass_filter,
        use_pend_low_pass_filter=use_pend_low_pass_filter,
        noise_on_q_observation=noise_on_q_observation,
        noise_on_pendulum_observation=noise_on_pendulum_observation,
        pendulum_observation_updates_with_125_Hz=pendulum_observation_updates_with_125_Hz,
        use_forward_predictor=use_forward_predictor,
        use_friction_action_model=use_friction_action_model,
    )
    # PART 5: SAVE THE DATA (INCLUDING THE INTERNAL SOLUTION OF THE CROCODDYL OC SOLVER)
    add_data = AddData(
        str_repr="beta_{}".format(float(2 * beta)),
        to_metadata={
            CompTimeNames.ddp_comp_time: ddp_time,
            CompTimeNames.mpc_comp_time: mpc_time,
            "beta": float(beta * 2),
            "crop_idx": t_crop,
            "n_fully_integrated_steps": n_fully_integrated_steps,
            "n_offset_500_Hz_steps": n_offset_500_Hz_steps,
        },
        to_files={"ddp_states": ddp_res.state},
        exclude=[CompTimeNames.ddp_comp_time, CompTimeNames.mpc_comp_time],
    )

    if data_saver is not None:
        print("Run trajectory: saving")
        data_saver.save_data(
            model, penalty_ddp, penalty_mpc, experiment_task_space, mpc_res, add_data
        )
    return mpc_res, ddp_res, mpc_time
