from utils.costs.residuals import ResidualBase
import crocoddyl
import numpy as np
from typing import List


class RotationTaskSpaceResidual(ResidualBase):
    def __init__(self, traj_rot: List[np.ndarray], frame_id: int, **kwargs):
        super(RotationTaskSpaceResidual, self).__init__(**kwargs)
        self._set_costs(traj_rot, frame_id)

    def _set_costs(self, traj_rot: List[np.ndarray], frame_id: int):
        cost_models = []
        for idx in range(np.shape(traj_rot)[0]):
            res_model = crocoddyl.ResidualModelFrameRotation(
                self.state, frame_id, traj_rot[idx], self.dim_control
            )
            cost_models.append(
                crocoddyl.CostModelResidual(
                    self.state,
                    self.act_func[idx],
                    res_model,
                )
            )

        assert len(cost_models) == len(
            traj_rot[:]
        ), "Different number of residual cost_models than trajectory points"
        self.costs = cost_models
        self.len_costs = len(cost_models)
