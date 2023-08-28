import argparse
from argparse import ArgumentParser

from utils.enums import BenchmarkTestSize


class TestsArgParser:
    def __init__(self):
        self.parser = ArgumentParser()

        # PENDULUM
        self.parser.add_argument(
            "--pendulum_len",
            "--pendulum_len",
            help="Length of the pendulum.",
            type=float,
            default=0.3,
        )
        self.parser.add_argument(
            "--pendulum_density",
            "--pendulum_density",
            help="Density of the pendulum's material.",
            type=float,
            default=2700,
        )

        # TEST TYPE
        self.parser.add_argument(
            "--test_size",
            "--test_size",
            help="Size of the test suite, options are: small, medium, large.",
            type=BenchmarkTestSize,
            default=BenchmarkTestSize.small,
        )
        # TEST CONFIG
        self.parser.add_argument(
            "--time_steps",
            "--time_steps",
            help="Number of timesteps for the trajectory.",
            type=int,
            default=3000,
        )
        self.parser.add_argument(
            "--direction",
            "--direction",
            help="Direction for the setpoint reach.",
            type=str,
            default="x",
        )
        self.parser.add_argument(
            "--orientation",
            "--orientation",
            help="Orientation for the setpoint reach.",
            type=int,
            default=-1,
        )
        self.parser.add_argument(
            "--radius",
            "--radius",
            help="Radius for the setpoint reach.",
            type=float,
            default=0.15,
        )
        # OBSERVER
        self.parser.add_argument(
            "--use_state_observer",
            "--use_state_observer",
            help="Use the state observer and not the perfect observation of the mujoco simulation.",
            type=bool,
            default=True,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--use_wam_low_pass_filter",
            "--use_wam_low_pass_filter",
            help="Use a low pass filter for FD derivative wam joint velocities computation.",
            type=bool,
            default=True,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--use_pend_low_pass_filter",
            "--use_pend_low_pass_filter",
            help="Use a low pass filter for FD derivative pendulum joint velocities computation.",
            type=bool,
            default=True,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--noise_on_pendulum_observation",
            "--noise_on_pendulum_observation",
            help="Simulate noise on the Pendulums Vector Observation",
            type=float,
            default=0.0,
        )
        self.parser.add_argument(
            "--noise_on_q_observation",
            "--noise_on_q_observation",
            help="Simulate noise on the WAMs joint position observations.",
            type=float,
            default=0.0,
        )
        self.parser.add_argument(
            "--pendulum_observation_updates_with_125_Hz",
            "--pendulum_observation_updates_with_125_Hz",
            help="Get Pendulum observations on 125Hz like on the real system, instead of 500Hz in simulation.",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        # SOLVER
        self.parser.add_argument(
            "--solver_max_iter",
            "--solver_max_iter",
            help="Maximum number of iterations for the solver.",
            type=int,
            default=20,
        )
        self.parser.add_argument(
            "--beta",
            "--beta",
            help="Beta parameter to allow a larger cone of angles.",
            type=float,
            default=0.5,
        )
        self.parser.add_argument(
            "--factor_int_time",
            "--factor_int_time",
            help="integration factor",
            type=int,
            default=4,
        )
        self.parser.add_argument(
            "--mpc_horizon",
            "--mpc_horizon",
            help="mpc horizon in time steps",
            type=int,
            default=10,
        )
        self.parser.add_argument(
            "--n_fully_integrated_steps",
            "--n_fully_integrated_steps",
            help="Number of fully executed steps, solver obtains the correct state.",
            type=int,
            default=1,
        )
        self.parser.add_argument(
            "--n_offset_500_Hz_steps",
            "--n_offset_500_Hz_steps",
            help="Number of offset timesteps at 500Hz, solver does not obtain correct state anymore.",
            type=int,
            default=0,
        )
        # OBSERVER
        self.parser.add_argument(
            "--use_forward_predictor",
            "--use_forward_predictor",
            help="Use the forward predictor with pinocchio.",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        # CROCODDYL
        self.parser.add_argument(
            "--use_friction_action_model",
            "--use_friction_action_model",
            help="Use custom gravity compensation model, including friction, damping and armature.",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        # EVALUATION
        self.parser.add_argument(
            "--n_end_error",
            "--n_end_error",
            help="Number of steps from the end of the trajectory to compute the cumulative and average error.",
            type=int,
            default=300,
        )
        self.parser.add_argument(
            "--visualize",
            "--visualize",
            help="Visualize mujoco simulation",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--show_plots",
            "--show_plots",
            help="Show torques and joint curves plots.",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--save_results",
            "--save_results",
            help="Save q and qdot arrays to file.",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--save_plots",
            "--save_plots",
            help="Save plots for task space.",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )

    def parser_args(self):
        return self.parser.parse_args()
