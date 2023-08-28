import pandas as pd

from utils.config.config_ip import ConfigWAM
from utils.costs.penalty.penalty_factory import PenaltyFactoryDDP, PenaltyFactoryMPC
from utils.decorators import log_time
from utils.experiments.setpoint.experiment_setpoint import ExperimentSetpoint
from utils.experiments.trajectory.traj_state_space import mpc_state_space
from utils.test_suite.test_helper import show_error, table_header

pd.set_option("display.float_format", "{:.2e}".format)


# Duration: ca. 180s = 3 mins
@log_time(time_in_mins=True)
def both_direction_all_orientations_single_horizon(
    model: ConfigWAM,
    default_experiment: ExperimentSetpoint,
    mpc_horizon: float = 10,
    factor_int_time: int = 4,
    solver_max_iter: int = 20,
    n_fully_integrated_steps: int = 1,
    n_offset_500_Hz_steps: int = 0,
    beta: float = 0.5,
    n_end_error: int = None,
):
    data = []
    for direction in ["x", "y", "z"]:
        for orientation in [-1, 1]:
            for radius in [0.05, 0.1, 0.15]:
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
                    get_penalty_mpc=PenaltyFactoryMPC.penalty_mpc("setpoint"),
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
                )
                # plot results, compute the average error
                pos_err, pos_avg_err, rot_err, rot_avg_err = show_error(
                    mpc_res=mpc_res,
                    experiment=experiment,
                    mpc_time=mpc_time,
                    end_err=n_end_error,
                )
                data.append(
                    dict(
                        direction=direction,
                        radius=radius,
                        orientation=orientation,
                        pos_err=pos_err,
                        pos_avg_err=pos_avg_err,
                        rot_err=rot_err,
                        rot_avg_err=rot_avg_err,
                        mpc_time=mpc_time,
                    )
                )
    results_df = pd.DataFrame(data).sort_values(by=["direction", "radius"])
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
    print(results_df)
    print(results_df.to_markdown())
