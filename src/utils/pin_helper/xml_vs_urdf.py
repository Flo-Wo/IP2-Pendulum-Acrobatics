"""
    mail@kaiploeger.net
    Test script to make sure the mujoco xml files and urdf files define the
    same dymaics
"""
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio as pin

from utils.config import ConfigWAM


def pin_fwd_dyn(model, q, dq, tau, N_DOF):
    data = model.createData()
    pin.computeAllTerms(model, data, q, dq)
    b = pin.rnea(model, data, q, dq, np.zeros(N_DOF))
    ddq = np.linalg.inv(data.M) @ (tau - b)
    # --> fits perfectly with the computation above
    # ddq_test = pin.aba(model, data, q, dq, tau)
    return ddq


def pin_inv_dyn(model, q, dq, ddq, N_DOF):
    data = model.createData()
    tau = pin.rnea(model, data, q, dq, ddq)
    return tau


def mj_fwd_dyn(model, sim, q, dq, tau, N_DOF):
    sim.qpos[:N_DOF] = q
    sim.qvel[:N_DOF] = dq
    sim.ctrl[:N_DOF] = tau
    mujoco.mj_forward(model, sim)
    ddq = sim.qacc[:N_DOF]
    return ddq


def mj_inv_dyn(model, sim, q, dq, ddq, N_DOF):
    sim.qpos[:N_DOF] = q
    sim.qvel[:N_DOF] = dq
    sim.qacc[:N_DOF] = ddq
    mujoco.mj_inverse(model, sim)
    tau = sim.qfrc_inverse[:N_DOF]
    return tau


def compare_models():

    with_pendulum = True
    np.random.seed(1234)
    N_SAMPLES = 10000

    if with_pendulum:
        N_DOF = 6  # without the balls' free joints in mujoco xml
        # pin_model = pin.buildModelFromUrdf("./wam/wam_pend.urdf")
        config_model = ConfigWAM("rotated")
        pin_model = config_model.pin_model
        # pin_model = pin.buildModelFromUrdf("./wam/rot_wam_pend.urdf")
    else:
        N_DOF = 4  # without the balls' free joints in mujoco xml
        pin_model = pin.buildModelFromUrdf("./wam/wam.urdf")
    # frames = list(pin_model.frames)
    # get absolute positions in the world frame via data.oMf[index],
    # for the absolute joint placements use data.oMi[index]
    # see also: (we get an element of SE3, meaning of the form (R,p) in heterogenous coordinates )
    # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/structse3_1_1Data.html
    pin_data = pin_model.createData()

    if with_pendulum:
        # mj_model = mujoco.MjModel.from_xml_path("./wam/wam_pend.xml")
        mj_model = config_model.mj_model
        # mj_model = mujoco.MjModel.from_xml_path("./wam/rot_wam_pend.xml")
    else:
        mj_model = mujoco.MjModel.from_xml_path("./wam/wam.xml")
    mj_sim = mujoco.MjData(mj_model)  # , nsubsteps=1)

    fwd_dyn_errors = np.zeros((N_SAMPLES, N_DOF))
    inv_dyn_errors = np.zeros((N_SAMPLES, N_DOF))

    fwd_count = 0
    inv_count = 0

    for i in range(N_SAMPLES):
        # random state close to neutral to avoid errors from collisions
        q = pin.randomConfiguration(pin_model)
        dq = np.random.uniform(-1, 1, [N_DOF])

        # random test inputs
        tau_test = np.random.uniform(-20, 20, [N_DOF])
        ddq_test = np.random.uniform(-5, 5, [N_DOF])

        # compare forward dynamics
        pin_ddq = pin_fwd_dyn(pin_model, q[:], dq[:], tau_test[:], N_DOF)
        mj_ddq = mj_fwd_dyn(mj_model, mj_sim, q[:], dq[:], tau_test[:], N_DOF)
        fwd_dyn_errors[i, :] = pin_ddq - mj_ddq

        # compare inverse dynamics
        pin_tau = pin_inv_dyn(pin_model, q[:], dq[:], ddq_test[:], N_DOF)
        mj_tau = mj_inv_dyn(mj_model, mj_sim, q[:], dq[:], ddq_test[:], N_DOF)
        inv_dyn_errors[i, :] = pin_tau - mj_tau
        if np.linalg.norm(fwd_dyn_errors[i, :]) >= 10e0:
            fwd_count += 1
            print("fwd:")
            print("q: ", q)
            print(pin_ddq)
            print(mj_ddq)

        if np.linalg.norm(inv_dyn_errors[i, :]) >= 10e0:
            inv_count += 1
            print("inverse: ")
            print(pin_tau)
            print(mj_tau)

    print("fwd_count: ", fwd_count)
    print("inv_count: ", inv_count)
    print(f"Test average over {N_SAMPLES} samples:")
    print(
        f"fdyn error: {np.mean(np.abs(fwd_dyn_errors), axis=0)} +- {np.std(np.abs(fwd_dyn_errors), axis=0)}"
    )
    print(
        f"idyn error: {np.mean(np.abs(inv_dyn_errors), axis=0)} +- {np.std(np.abs(inv_dyn_errors), axis=0)}"
    )

    # to check for drift
    # print(f'fdyn error: {np.mean(fwd_dyn_errors, axis=0)} +- {np.std(fwd_dyn_errors, axis=0)}')
    # print(f'idyn error: {np.mean(inv_dyn_errors, axis=0)} +- {np.std(inv_dyn_errors, axis=0)}')

    # fig, axs = plt.subplots(N_DOF, 1, sharex=True, tight_layout=True)
    # axs[0].set_title("forward dynamics errors")
    # for i in range(N_DOF):
    #     axs[i].hist(np.abs(fwd_dyn_errors[:, i]), bins=int(100))

    # fig, axs = plt.subplots(N_DOF, 1, sharex=True, tight_layout=True)
    # axs[0].set_title("inverse dynamics errors")
    # for i in range(N_DOF):
    #     axs[i].hist(np.abs(inv_dyn_errors[:, i]), bins=int(100))

    # plt.show()


if __name__ == "__main__":
    compare_models()
