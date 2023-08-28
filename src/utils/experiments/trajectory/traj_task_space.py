from copy import copy
from typing import Tuple

import crocoddyl
import numpy as np

from utils.config import ConfigModelBase
from utils.costs import ControlBounds
from utils.data_handling import AddData, DataSaver
from utils.experiments.trajectory import (
    ExperimentTrajectory,
    PenaltyTrajectory,
    integrated_cost_models,
)
from utils.solver import SolverResultsBase, ddp, mpc


def mpc_task_space(
    model: ConfigModelBase,
    experiment: ExperimentTrajectory,
    penalty: PenaltyTrajectory,
    solver: str = "ddp",
    solver_max_iter: int = 100,
    mpc_horizon: int = 10,
    mpc_factor_integration_time: int = 1,
    data_saver: DataSaver = None,
    show_logs: bool = False,
    ctrl_bounds: ControlBounds = None,
) -> Tuple[SolverResultsBase, int]:
    # we need copy here, otherwise the results are incorrect due to side effects
    mj_model = copy(model.mj_model)
    mj_data = copy(model.mj_data)
    pin_model = copy(model.pin_model)
    pin_data = copy(model.pin_data)

    state = crocoddyl.StateMultibody(pin_model)
    actuation_model = crocoddyl.ActuationModelFull(state)
    action_model = crocoddyl.DifferentialActionModelFreeFwdDynamics

    # NOTE: important --> we need to adjust the time constant if we work with a control
    # delay
    dt_const = 0.002

    dt_const = dt_const * mpc_factor_integration_time

    print("dt_const: ", dt_const)

    if ctrl_bounds is None:
        ctrl_bounds = ControlBounds()

    # build intermediate/running cost models
    int_cost_models = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty,
        experiment,
        dt_const,
        ctrl_bounds,
    )
    # build terminal cost models
    terminal_cost_model = crocoddyl.CostModelSum(state)
    terminal_node = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation_model, terminal_cost_model
        ),
        0,
    )

    # NOTE: we use the last model as the terminal cost model
    # terminal_node = int_cost_models[-1]

    if solver == "ddp":
        # solve via ddp solver/MPC
        results, time = ddp(
            int_cost_models,
            terminal_node,
            model.q_config,
            qdot=np.zeros(state.nv),
            solver_max_iter=solver_max_iter,
            show_logs=show_logs,
        )
    elif solver == "mpc":
        results, time = mpc(
            mj_model,
            mj_data,
            pin_model,
            pin_data,
            int_cost_models,
            terminal_node,
            mpc_horizon,
            solver_max_iter,
            show_logs=show_logs,
            factor_integration_time=mpc_factor_integration_time,
        )
    else:
        raise NotImplementedError

    add_data = AddData({"comp_time": time}, exclude=["comp_time"])

    if data_saver is not None:
        print("Run trajectory: saving")
        data_saver.save_data(model, penalty, experiment, results, add_data)
    return results, time
