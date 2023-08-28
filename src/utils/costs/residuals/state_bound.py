from utils.costs.residuals import ResidualBase
import numpy as np
import crocoddyl


class StateBoundResidual(ResidualBase):
    def __init__(
        self,
        lower_bound: np.ndarray = None,
        upper_bound: np.ndarray = None,
        safety_margin: float = 1,
        **kwargs: dict
    ):
        super(StateBoundResidual, self).__init__(**kwargs)
        self._set_costs(lower_bound, upper_bound, safety_margin)

    def _set_costs(self, lb, ub, safety_margin):
        if lb is None:
            lb = np.concatenate(
                [self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]]
            )
        if ub is None:
            ub = np.concatenate(
                [self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]]
            )
        state_bound_res = crocoddyl.ResidualModelState(self.state, self.dim_control)
        state_bound_act = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lb, ub, safety_margin)
        )
        self.costs = crocoddyl.CostModelResidual(
            self.state, state_bound_act, state_bound_res
        )
        self.len_costs = 1
