import crocoddyl
import matplotlib.pyplot as plt
import numpy as np

from utils.config import ConfigWAM
from utils.costs import Penalty
from utils.data_handling import DataSaver
from utils.experiments.setpoint import ExperimentSetpoint
from utils.experiments.trajectory import PenaltyTrajectory, mpc_task_space
from utils.visualize import loop_mj_vis, plot_single_trajectory, plot_trajectory

if __name__ == "__main__":

    standard = ConfigWAM("standard")
    standard_angled = ConfigWAM("standard_angled")
    rotated = ConfigWAM("rotated")
    human = ConfigWAM("human")

    models = [standard, standard_angled, rotated, human]

    penalties = {
        "standard": PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(5e4),
            rot_pen=Penalty(
                1e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
        "standard_angled": PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(1e5),
            rot_pen=Penalty(
                6e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
        "rotated": PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(3e4),
            rot_pen=Penalty(
                3e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
        "human": PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(5e4),
            rot_pen=Penalty(
                1e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
    }

    # define the data saver once
    # data_saver = DataSaver("../data/setpoint/mpc_raw.csv", "../data/setpoint/mpc_raw/")

    # normal_standard_0.3: x, -1, 0.15, 20, 2
    # normal_standard_angled_0.3: x, -1, 0.1, 20, 2
    model = standard_angled
    penalty = penalties[model.model_name]
    dir = "z"
    orientation = -1
    radius = 0.15
    mpc_horizon = 10
    factor_int_time = 4
    # define the individual penalties
    experiment = ExperimentSetpoint(
        24,
        model.get_pend_end_world(),
        {
            "direction": dir,
            "orientation": orientation,
            "radius": radius,
        },
        24,
        3000,
        rest_pos=model.q_config,
    )

    res, time_needed = mpc_task_space(
        model,
        experiment,
        penalty,
        solver="mpc",
        solver_max_iter=100,
        mpc_horizon=mpc_horizon,
        mpc_factor_integration_time=factor_int_time,
    )

    from_db = np.load(
        "../data/setpoint/mpc_raw/normal_standard_angled_0.3_x_neg_0.05_setpoint_xID_24_rotID_24_MPC_factor_int_time_1_horizon_20_x.npy"
    )
    plot_trajectory(experiment.traj_x, {"mpc": res.x[1:, :], "db": from_db[1:, :]})
    plot_single_trajectory(res.u, label="torques u")
    plt.show()
    # plot_single_traj(q_list, label="q(t)")
    # plt.show()
    # loop_mj_vis(model, q=q_list, q_dot=q_dot_list, num_motors=4)
    loop_mj_vis(model, res.u, num_motors=4)
