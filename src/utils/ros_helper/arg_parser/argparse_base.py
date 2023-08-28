import argparse
from argparse import ArgumentParser


class ArgParserBase:
    def __init__(self):
        self.parser = ArgumentParser()

        self.parser.add_argument(
            "--pendulum_len",
            "--pendulum_len",
            help="Length of the pendulum",
            type=float,
            default=0.3,
        )

        self.parser.add_argument(
            "--with_pendulum",
            "--with_pendulum",
            help="With (True) or without (False) pendulum.",
            type=bool,
            default=True,
            action=argparse.BooleanOptionalAction,
        )

        self.parser.add_argument(
            "--control_frequency",
            "--control_frequency",
            help="Control frequency",
            type=float,
            default=125,
        )

        self.parser.add_argument(
            "--sim_frequency",
            "--sim_frequency",
            help="Simulation frequency",
            type=float,
            default=500,
        )

        self.parser.add_argument(
            "--synchronized",
            "--synchronized",
            help="simulation should run synchronized",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        self.parser.add_argument(
            "--factor_int_time",
            "--factor_int_time",
            help="Integration factor for the duration of the actions.",
            type=int,
            default=4,
        )

    def parse_args(self):
        return self.parser.parse_args()
