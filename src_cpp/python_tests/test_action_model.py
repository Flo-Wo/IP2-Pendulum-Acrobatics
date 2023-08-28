import crocoddyl
import numpy as np

from utils.config.action_model_gravity_compensation import (
    DifferentialActionModelGravityCompensatedFwdDynamics,
)
from utils.config.config_ip import ConfigWAM

if __name__ == "__main__":
    model = ConfigWAM(
        model_name="rotated", pendulum_len=0.374, use_friction_action_model=True
    )
    mj_model = model.mj_model
    state = crocoddyl.StateMultibody(model.pin_model)
    actuation = crocoddyl.ActuationModelFull(state)
    runningCostModel = crocoddyl.CostModelSum(state)
    model = DifferentialActionModelGravityCompensatedFwdDynamics(
        state,
        actuation,
        runningCostModel,
        armature=np.zeros_like(mj_model.dof_armature.copy()),
        damping=np.zeros_like(mj_model.dof_damping.copy()),
        coulomb_friction=np.zeros_like(mj_model.dof_frictionloss),
        grav_comp_idxs=np.array([0, 1, 2, 3]),
    )
    data = model.createData()
    x = np.concatenate((np.array([0, -1.3, 0, 1.3, 0, 0]), np.ones(6)))
    u = np.ones(6)

    model.calc(data, x, u)
    print("xout = ", data.xout)
    model.calcDiff(data, x, u)
    print("\nFx = \n", data.Fx)
    print("\nFu = \n", data.Fu)
