from typing import Optional

import crocoddyl
import numpy as np

from utils.config.action_model_gravity_compensation import (
    DifferentialActionModelGravityCompensatedFwdDynamics,
)


class DifferentialActionModelFactory:
    @staticmethod
    def get_action_model(
        use_friction_action_model: bool = False,
        armature: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        coulomb_friction: Optional[np.ndarray] = None,
        gravity_comp_idxs: Optional[np.ndarray] = None,
    ) -> crocoddyl.DifferentialActionModelAbstract:

        # provide default arguments in the case of the custom differential action model
        if use_friction_action_model:
            return lambda state, actuation, costModel: DifferentialActionModelGravityCompensatedFwdDynamics(
                state=state,
                actuation=actuation,
                costModel=costModel,
                armature=armature,
                damping=damping,
                coulomb_friction=coulomb_friction,
                grav_comp_idxs=gravity_comp_idxs,
            )
        else:
            # no default arguments are needed
            return crocoddyl.DifferentialActionModelFreeFwdDynamics
