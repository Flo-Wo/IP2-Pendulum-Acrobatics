import logging
from typing import List, Union

import crocoddyl

from utils.costs.control_bounds import ControlBounds
from utils.costs.residuals import ResidualBase


def integrate_residuals(
    state: crocoddyl.StateMultibody,
    action_model: crocoddyl.DifferentialActionModelAbstract,
    actuation_model: crocoddyl.ActuationModelAbstract,
    dt_const: Union[List[float], float],
    residuals: List[ResidualBase],
    ctrl_bounds: ControlBounds = None,
):
    if ctrl_bounds is None:
        ctrl_bounds = ControlBounds()
    else:
        logging.info("Control bounds are non-default.")
    cost_models = _cost_models(state, residuals)
    return _integrate(
        state,
        action_model,
        actuation_model,
        cost_models,
        dt_const,
        ctrl_bounds=ctrl_bounds,
    )


def _integrate(
    state: crocoddyl.StateMultibody,
    action_model: crocoddyl.DifferentialActionModelAbstract,
    actuation_model: crocoddyl.ActuationModelAbstract,
    cost_models: List[crocoddyl.CostModelSum],
    dt_const: Union[List[float], float],
    ctrl_bounds: ControlBounds = ControlBounds(),
) -> list:
    # create list with constant values --> if only a scalar is given
    if not isinstance(dt_const, list):
        dt_const = len(cost_models) * [dt_const]
    running_models = []
    if ctrl_bounds is not None:
        logging.info("Control bounds are given and will be set.")
    for idx, c_model in enumerate(cost_models):
        # run_model = crocoddyl.IntegratedActionModelRK4(
        # rktype="RK4",
        run_model = crocoddyl.IntegratedActionModelEuler(
            action_model(state, actuation_model, c_model),
            dt_const[idx],
        )
        if ctrl_bounds is not None:
            run_model.u_lb = ctrl_bounds.get_lower(idx)
            run_model.u_ub = ctrl_bounds.get_upper(idx)
        running_models.append(run_model)

    return running_models


def _cost_models(state: crocoddyl.StateMultibody, residuals: List[ResidualBase]):
    time_steps = max([res.len_costs for res in residuals])
    models = []
    for idx in range(time_steps):
        cost_model = crocoddyl.CostModelSum(state)
        # add the costs time step wise
        for res in residuals:
            cost_model.addCost(*res[idx])  # *(name, costs, penalty)
        models.append(cost_model)
    return models
