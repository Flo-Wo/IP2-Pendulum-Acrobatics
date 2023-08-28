from utils.costs.residuals import ResidualBase
import numpy as np
import crocoddyl


class PositionTaskSpaceResidual(ResidualBase):
    def __init__(self, traj_x: np.ndarray, frame_id: int, **kwargs):
        super(PositionTaskSpaceResidual, self).__init__(**kwargs)
        self._set_costs(traj_x, frame_id)

    def _set_costs(self, traj_x: np.ndarray, frame_id: int):
        cost_models = []
        for idx in range(np.shape(traj_x)[0]):
            res_model = crocoddyl.ResidualModelFrameTranslation(
                self.state, frame_id, traj_x[idx, :], self.dim_control
            )
            cost_models.append(
                crocoddyl.CostModelResidual(
                    self.state,
                    self.act_func[idx],
                    res_model,
                )
            )

        assert len(cost_models) == len(
            traj_x[:]
        ), "Different number of residual models than trajectory points"
        self.costs = cost_models
        self.len_costs = len(cost_models)
