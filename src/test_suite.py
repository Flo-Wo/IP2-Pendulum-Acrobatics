from copy import deepcopy

import numpy as np
import pinocchio as pin

from utils.arg_parser import TestsArgParser
from utils.config.config_ip import ConfigWAM
from utils.enums import BenchmarkTestSize
from utils.experiments.setpoint.experiment_setpoint import ExperimentSetpoint
from utils.test_suite import (
    both_direction_all_orientations_single_horizon,
    both_directions_all_orientations_all_horizons,
    one_experiment,
)

# NOTE: if you want to change the defaults for test, consider:
#   utils.test_suite.test_helper
if __name__ == "__main__":
    test_args = vars(TestsArgParser().parser_args())
    test_size = test_args["test_size"]

    model = ConfigWAM(
        "rotated",
        pendulum_len=test_args["pendulum_len"],
        pendulum_density=test_args["pendulum_density"],
        use_friction_action_model=test_args["use_friction_action_model"],
    )
    time_steps = test_args["time_steps"]

    params_dict = dict(
        x_frame_id=deepcopy(model.get_pend_pole_tip_id()),
        start_point=deepcopy(model.get_pend_end_world()),
        rot_frame_id=deepcopy(model.get_pend_pole_tip_id()),
        time_steps=deepcopy(time_steps),
        rest_pos=deepcopy(model.q_config),
    )

    class ExperimentSetpointDefault(ExperimentSetpoint):
        def __init__(self, target_raw):
            super(ExperimentSetpointDefault, self).__init__(
                target_raw=target_raw, **params_dict
            )

    if test_size == BenchmarkTestSize.small:
        one_experiment(
            model=model,
            default_experiment=ExperimentSetpointDefault,
            direction=test_args["direction"],
            orientation=test_args["orientation"],
            radius=test_args["radius"],
            mpc_horizon=test_args["mpc_horizon"],
            factor_int_time=test_args["factor_int_time"],
            solver_max_iter=test_args["solver_max_iter"],
            n_fully_integrated_steps=test_args["n_fully_integrated_steps"],
            n_offset_500_Hz_steps=test_args["n_offset_500_Hz_steps"],
            beta=test_args["beta"],
            make_visualization=test_args["visualize"],
            show_plots=test_args["show_plots"],
            save_results=test_args["save_results"],
            save_plots=test_args["save_plots"],
            use_state_observer=test_args["use_state_observer"],
            use_wam_low_pass_filter=test_args["use_wam_low_pass_filter"],
            use_pend_low_pass_filter=test_args["use_pend_low_pass_filter"],
            noise_on_q_observation=test_args["noise_on_q_observation"],
            noise_on_pendulum_observation=test_args["noise_on_pendulum_observation"],
            pendulum_observation_updates_with_125_Hz=test_args[
                "pendulum_observation_updates_with_125_Hz"
            ],
            use_forward_predictor=test_args["use_forward_predictor"],
            use_friction_action_model=test_args["use_friction_action_model"],
        )

    if test_size == BenchmarkTestSize.medium:
        both_direction_all_orientations_single_horizon(
            model=model,
            mpc_horizon=test_args["mpc_horizon"],
            factor_int_time=test_args["factor_int_time"],
            solver_max_iter=test_args["solver_max_iter"],
            n_fully_integrated_steps=test_args["n_fully_integrated_steps"],
            n_offset_500_Hz_steps=test_args["n_offset_500_Hz_steps"],
            beta=test_args["beta"],
            n_end_error=test_args["n_end_error"],
        )

    if test_size == BenchmarkTestSize.large:
        both_directions_all_orientations_all_horizons(
            model=model,
            factor_int_time=test_args["factor_int_time"],
            solver_max_iter=test_args["solver_max_iter"],
            n_fully_integrated_steps=test_args["n_fully_integrated_steps"],
            n_offset_500_Hz_steps=test_args["n_offset_500_Hz_steps"],
            beta=test_args["beta"],
            n_end_error=test_args["n_end_error"],
        )
