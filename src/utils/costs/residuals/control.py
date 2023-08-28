import crocoddyl

from utils.costs.residuals import ResidualBase


class ControlResidual(ResidualBase):
    def __init__(self, **kwargs):
        super(ControlResidual, self).__init__(**kwargs)
        self._set_costs()

    def _set_costs(self):
        models = []
        for idx in range(len(self.act_func)):
            models.append(
                crocoddyl.CostModelResidual(
                    self.state,
                    self.act_func[idx],
                    crocoddyl.ResidualModelControl(self.state, self.state.nv),
                )
            )
        self.costs = models
        self.len_costs = len(models)
