import crocoddyl

from utils.costs import Penalty
from utils.costs.residuals import (
    ControlResidual,
    PositionTaskSpaceResidual,
    ResidualBase,
    RotationTaskSpaceResidual,
    StateBoundResidual,
    StateSpaceResidual,
)


class ResidualFactory:
    def __init__(self, state: crocoddyl.StateMultibody, dim_control: int = 6):
        self.state = state
        self.dim_control = dim_control

    def compute_residual(
        self, name: str, res_type: str, penalty: Penalty, **kwargs
    ) -> ResidualBase:
        """Options are (with corresponding Residual Type in brackets),
        needed parameters inside kwargs listed below:
        - control (ControlResidual)
        - state_bound (StateBoundResidual)
            * optional lower_bound
            * optional upper_bound
            * optional margin
        - task_space (PositionTaskSpaceResidual)
            * traj_x
            * frame_id
        - rotation (RotationTaskSpaceResidual)
            * weights_rotation
            * traj_rot
            * frame_id
        - state_space (StateSpaceResidual)
            * q, v
        - q (StateSpaceResidual)
            * q (v = None)
        - v (StateSpaceResidual)
            * v (q = None)
        """
        shared_args = {
            "state": self.state,
            "name": name,
            "penalty": penalty.pen,
            "dim_control": self.dim_control,
            "act_func": penalty.act_func,
            "act_func_params": penalty.act_func_params,
        }
        if res_type == "control":
            shared_args["dim_residual"] = self.dim_control
            return ControlResidual(**shared_args)
        elif res_type == "state_bound":
            shared_args["dim_residual"] = self.state.nq + self.state.nv
            lower_bound = kwargs.get("lower_bound", None)
            upper_bound = kwargs.get("upper_bound", None)
            safety_margin = kwargs.get("safety_margin", 1)
            return StateBoundResidual(
                lower_bound, upper_bound, safety_margin, **shared_args
            )
        elif res_type == "task_space":
            shared_args["dim_residual"] = 3
            traj_x = kwargs.get("traj_x", None)
            frame_id = kwargs.get("frame_id")
            return PositionTaskSpaceResidual(traj_x, frame_id, **shared_args)
        elif res_type == "rotation":
            shared_args["dim_residual"] = 3
            traj_rot = kwargs.get("traj_rot", None)
            frame_id = kwargs.get("frame_id")
            return RotationTaskSpaceResidual(traj_rot, frame_id, **shared_args)
        elif res_type == "state_space":
            shared_args["dim_residual"] = self.state.nq + self.state.nv
            q = kwargs.get("q")
            v = kwargs.get("v")
            return StateSpaceResidual(q, v, **shared_args)
        elif res_type == "q":
            shared_args["dim_residual"] = self.state.nq + self.state.nv
            q = kwargs.get("q")
            print("residual factory, q: ", q)
            return StateSpaceResidual(q, None, **shared_args)
        elif res_type == "v":
            shared_args["dim_residual"] = self.state.nq + self.state.nv
            v = kwargs.get("v")
            print("residual factory v: ", v)
            return StateSpaceResidual(None, v, **shared_args)
        else:
            raise NotImplementedError
