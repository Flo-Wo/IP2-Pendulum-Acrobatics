from typing import Optional

import crocoddyl
import numpy as np
import pinocchio as pin


class FrictionModel:
    def __init__(
        self, coulomb: np.ndarray, viscous: np.ndarray, coulomb_slope: float = 100
    ):
        self.coulomb = coulomb
        self.viscous = viscous
        self.coulomb_slope = coulomb_slope

    def calc(self, v):
        activation = np.tanh(self.coulomb_slope * v)
        return -activation * self.coulomb - v * self.viscous

    def calcDiff(self, v):
        activation_der = self.coulomb_slope * (
            1 - (np.tanh(self.coulomb_slope * v) ** 2)
        )
        return np.diag(-activation_der * self.coulomb - self.viscous)


class DifferentialActionModelGravityCompensatedFwdDynamics(
    crocoddyl.DifferentialActionModelAbstract
):
    def __init__(
        self,
        state,
        actuation,
        costModel,
        armature: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        coulomb_friction: Optional[np.ndarray] = None,
        grav_comp_idxs: Optional[np.ndarray] = None,
    ):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, state.nv, costModel.nr
        )
        self.costs = costModel
        self.actuation = actuation

        self.armature = np.zeros(self.state.nv)
        if armature is not None:
            self.armature[:] = armature

        if damping is None:
            damping = np.zeros(self.state.nv)

        if coulomb_friction is None:
            coulomb_friction = np.zeros(self.state.nv)

        if grav_comp_idxs is None:
            grav_comp_idxs = np.arange(self.state.nv)
        self.g_idxs = grav_comp_idxs

        self.friction_model = FrictionModel(coulomb_friction, damping)

    def calc(self, data, x, u=None):
        self.actuation.calc(data.multibody.actuation, x, u)
        q, v = x[: self.state.nq], x[-self.state.nv :]

        pin.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
        mass_matrix = data.pinocchio.M + np.diag(self.armature)
        g = np.zeros_like(data.pinocchio.g)
        g[self.g_idxs] = data.pinocchio.g[self.g_idxs]
        data.xout = np.linalg.solve(
            mass_matrix, u + g + self.friction_model.calc(v) - data.pinocchio.nle
        )

        pin.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pin.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]

        grav_comp_derivatives = pin.computeGeneralizedGravityDerivatives(
            self.state.pinocchio, data.pinocchio, q
        )
        pruned_grav_comp_derivates = np.zeros_like(grav_comp_derivatives)
        pruned_grav_comp_derivates[self.g_idxs] = grav_comp_derivatives[self.g_idxs]
        self.calc(data, x, u)

        dtau_dq, dtau_dv, mass_matrix = pin.computeRNEADerivatives(
            self.state.pinocchio, data.pinocchio, q, v, data.xout
        )
        mass_matrix += np.diag(self.armature)
        dacc_dq_rec = -np.linalg.solve(
            mass_matrix, dtau_dq - pruned_grav_comp_derivates
        )
        dacc_dv_rec = -np.linalg.solve(
            mass_matrix, dtau_dv - self.friction_model.calcDiff(v)
        )

        data.Fx = np.concatenate((dacc_dq_rec, dacc_dv_rec), axis=-1)
        data.Fu = np.linalg.inv(mass_matrix)
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pin.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorActMultibody(
            data.pinocchio, self.actuation.createData()
        )
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        )  # this allows us to share the memory of cost-terms of action model
        return data
