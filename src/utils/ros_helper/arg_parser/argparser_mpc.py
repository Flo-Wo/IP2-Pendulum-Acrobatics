from utils.enums import WarmstartType
from utils.ros_helper.arg_parser import ArgParserBase


class ArgParserMPC(ArgParserBase):
    def __init__(self):
        super(ArgParserMPC, self).__init__()

        # MPC custom arguments
        self.parser.add_argument(
            "--mpc_horizon",
            "--mpc_horizon",
            help="mpc horizon in time steps",
            type=int,
            default=10,
        )

        self.parser.add_argument(
            "--solver_max_iter",
            "--solver_max_iter",
            help="Maximum number of iterations for the solver.",
            type=int,
            default=20,
        )

        self.parser.add_argument(
            "--warmstart_type",
            "--warmstart_type",
            help="Warmstart type for the solver, options are: 'none', 'quasi_static', 'cache'",
            type=WarmstartType,
            default=WarmstartType.quasi_static,
        )
