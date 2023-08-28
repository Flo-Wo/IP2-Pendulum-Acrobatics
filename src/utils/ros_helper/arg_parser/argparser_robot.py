import argparse

from utils.ros_helper.arg_parser import ArgParserBase


class ArgParserRobot(ArgParserBase):
    def __init__(self):
        super(ArgParserRobot, self).__init__()

        # Robot custom arguments
        self.parser.add_argument(
            "--n_offset_steps",
            "--n_offset_steps",
            help="Number of offset timesteps in sync mode",
            type=int,
            default=0,
        )
        self.parser.add_argument(
            "--visualize",
            "--visualize",
            help="Visualize mujoco simulation",
            type=bool,
            default=True,
            action=argparse.BooleanOptionalAction,
        )
