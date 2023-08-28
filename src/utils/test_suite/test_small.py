import os
import pathlib

import numpy as np
from matplotlib import pyplot as plt

from utils.config.config_ip import ConfigWAM
from utils.costs.penalty import PenaltyFactoryDDP, PenaltyFactoryMPC
from utils.decorators import log_time
from utils.experiments.setpoint.experiment_setpoint import ExperimentSetpoint
from utils.experiments.trajectory.traj_state_space import mpc_state_space
from utils.pin_helper.world_frame import task_space_traj
from utils.test_suite.test_helper import show_error, table_header
from utils.visualize import plot_task_space_latex
from utils.visualize.plots.plot_traj import (
    plot_computation_time,
    plot_single_trajectory,
    plot_trajectory,
)
from utils.visualize.vis_mujoco.loop_trajectory import loop_mj_vis
from utils.visualize.vis_mujoco.visualizer import visualize_tip_position


@log_time(time_in_mins=True)
def one_experiment(
    model: ConfigWAM,
    default_experiment: ExperimentSetpoint,
    direction: str,
    orientation: float = -1,
    radius: float = 0.15,
    mpc_horizon: float = 10,
    factor_int_time: int = 4,
    solver_max_iter: int = 20,
    n_fully_integrated_steps: int = 1,
    n_offset_500_Hz_steps: int = 0,
    beta: float = 0.5,
    make_visualization: bool = True,
    show_plots: bool = True,
    save_results: bool = False,
    save_plots: bool = True,
    use_state_observer: bool = False,
    use_wam_low_pass_filter: bool = True,
    use_pend_low_pass_filter: bool = True,
    noise_on_q_observation: float = 0.0,
    noise_on_pendulum_observation: float = 0.0,
    pendulum_observation_updates_with_125_Hz: bool = False,
    use_forward_predictor: bool = True,
    use_friction_action_model: bool = False,
):
    experiment: ExperimentSetpoint = default_experiment(
        target_raw={
            "direction": direction,
            "orientation": orientation,
            "radius": radius,
        }
    )
    mpc_res, ddp_res, mpc_time = mpc_state_space(
        model=model,
        experiment_task_space=experiment,
        get_penalty_ddp=PenaltyFactoryDDP.penalty_ddp("all"),
        get_penalty_mpc=PenaltyFactoryMPC.penalty_mpc(
            "setpoint" if radius > 0 else "stabilization"
        ),
        max_iter_ddp=1000,
        max_iter_mpc=solver_max_iter,
        mpc_horizon=mpc_horizon,
        factor_int_time=factor_int_time,
        beta=beta,
        t_crop=1500,
        t_total=experiment.time_steps,
        show_logs=False,
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
    # plot results, compute the average error
    print(
        table_header(
            mpc_horizon=mpc_horizon,
            factor_int_time=factor_int_time,
            solver_max_iter=solver_max_iter,
            n_fully_integrated_steps=n_fully_integrated_steps,
            n_offset_500_Hz_steps=n_offset_500_Hz_steps,
            beta=beta,
        )
    )
    _ = show_error(
        model=model,
        mpc_res=mpc_res,
        experiment=experiment,
        mpc_time=mpc_time,
        comp_time=mpc_res.solver_comp_times,
        end_err=300,
    )
    if save_results:
        path = pathlib.Path(__file__).parent.parent.parent.parent / "data/test_suite"
        filename = "test_observer"
        if use_friction_action_model:
            filename += "_friction"
        np.save(os.path.join(path, filename + "_states"), mpc_res.states)
        np.save(os.path.join(path, filename + "_torques"), mpc_res.u)
    if show_plots:
        q_list = np.array([x[:6] for x in ddp_res.state[1:]])
        x_task_raw = task_space_traj(
            model.pin_model, model.pin_data, q_list, experiment.x_frame_id
        )
        # q real vs. observation
        plot_trajectory(
            goal=mpc_res.states[1:, :6],
            experiments={"observer q": mpc_res.observed_states[1:, :6]},
        )
        # qdot real vs. observation
        plot_trajectory(
            goal=mpc_res.states[1:, 6:],
            experiments={"observer qdot": mpc_res.observed_states[1:, 6:]},
        )
        # Pendulum trajecotry
        print("plot task space trajectory")
        plot_trajectory(
            goal=experiment.traj_x,
            experiments={
                "mpc": mpc_res.x[1:, :],
                "ddp_preplanned": x_task_raw,
            },
        )
        plot_single_trajectory(traj=mpc_res.u, label="torques")
        plot_computation_time(comp_time=mpc_res.solver_comp_times, total_time=mpc_time)
        plt.show()

    ip1 = False
    no_noise_suffix = "_ip1" if ip1 else ""
    if save_plots:
        np.save(
            "../data/test_suite/test_small_target" + no_noise_suffix, experiment.traj_x
        )
        np.save("../data/test_suite/test_small_mpc" + no_noise_suffix, mpc_res.x[1:, :])
        np.save(
            "../data/test_suite/test_small_ddp_planned" + no_noise_suffix, x_task_raw
        )

    if make_visualization:
        print("Visualisation starts")
        # loop_mj_vis(model, mpc_res.u[1:, :], num_motors=4, make_cam=True)
        # visualize(
        #     model,
        #     "stabilization" + no_noise_suffix,
        #     path="../videos/",
        #     controls=mpc_res.u[1:, :],
        # )
        loop_mj_vis(
            model, q=mpc_res.states[1:, :6], q_dot=mpc_res.states[1:, 6:], num_motors=4
        )
