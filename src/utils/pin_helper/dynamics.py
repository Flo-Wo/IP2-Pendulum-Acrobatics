import pinocchio as pin
import numpy as np
import mujoco_py


def pin_fwd_dyn(pin_model_fwd, q, dq, tau, N_DOF):
    pin_data_fwd = pin_model_fwd.createData()
    pin.computeAllTerms(pin_model_fwd, pin_data_fwd, q, dq)
    b = pin.rnea(pin_model_fwd, pin_data_fwd, q, dq, np.zeros(N_DOF))
    return np.linalg.inv(pin_data_fwd.M) @ (tau - b)  # = ddq


def pin_inv_dyn(pin_model_inv, q, dq, ddq, N_DOF):
    pin_data_inv = pin_model_inv.createData()
    return pin.rnea(pin_model_inv, pin_data_inv, q, dq, ddq)  # = tau


def mj_fwd_dyn(sim_fwd, q, dq, tau, N_DOF):
    sim_fwd.data.qpos[:N_DOF] = q
    sim_fwd.data.qvel[:N_DOF] = dq
    sim_fwd.data.ctrl[:N_DOF] = tau
    sim_fwd.forward()
    ddq = sim_fwd.data.qacc[:N_DOF]
    return ddq


def mj_inv_dyn(sim_fwd, q, dq, ddq, N_DOF):
    sim_fwd.data.qpos[:N_DOF] = q
    sim_fwd.data.qvel[:N_DOF] = dq
    sim_fwd.data.qacc[:N_DOF] = ddq
    mujoco_py.functions.mj_inverse(sim_fwd.model, sim_fwd.data)
    tau = sim_fwd.data.qfrc_inverse[:N_DOF]
    return tau
