import math
import os
import pathlib
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from utils.config import ConfigWAM
from utils.config.action_model_gravity_compensation import FrictionModel
from utils.solver.MPC.mpc_helper import (
    LuenbergerObserver,
    compute_additive_noise_on_observation,
)
from utils.visualize.plots.plot_traj import plot_trajectory

# load trajectory

folder = pathlib.Path(__file__).parent.parent / "data/test_suite"
state_list = np.load(os.path.join(folder, "test_observer_states.npy"))
torque_list = np.load(os.path.join(folder, "test_observer_torques.npy"))

# state_list = np.load(os.path.join(folder, "test_observer_friction_states.npy"))
# torque_list = np.load(os.path.join(folder, "test_observer_friction_torques.npy"))

# TODO: check for first index or not, maybe 1:
q_list = state_list[:, :6]
qdot_list = state_list[:, 6:]


wam_model = ConfigWAM(
    "rotated",
    pendulum_len=0.374,
    use_friction_action_model=True,
)

q_wam_start = deepcopy(wam_model.q_config[:4])
q_pend_start = deepcopy(wam_model.q_config[-2:])


def observer_traj(
    q_list: np.ndarray,
    qdot_list: np.ndarray,
    torque_list: np.ndarray,
    observer: LuenbergerObserver,
    show=False,
    noise_wam: float = 5e-4,
    noise_pend: float = 1e-3,
    err_measure: str = "avg",
) -> float:
    observer_q, observer_qdot = observer.get_solver_state(separate=True)
    q_observed = [observer_q.copy()]
    qdot_observed = [observer_qdot.copy()]
    assert (
        q_list[0, :] == observer.position
    ).all(), "Start positions have to be equal."

    assert (
        qdot_list[0, :] == observer.velocity
    ).all(), "Start velocities have to be equal."
    for time_idx in range(q_list.shape[0]):

        wam_q_noise = compute_additive_noise_on_observation(
            q_list[time_idx, :4], noise_factor=noise_wam
        )
        pend_q_noise = compute_additive_noise_on_observation(
            q_list[time_idx, -2:], noise_factor=noise_pend
        )
        tau = np.zeros(6)
        tau[:4] = torque_list[time_idx, :4]
        observer.observe(
            time_idx,
            wam_q=q_list[time_idx, :4] + wam_q_noise,
            pend_q=q_list[time_idx, -2:] + pend_q_noise if time_idx % 4 == 0 else None,
            torque=tau,
            print_error=False,
        )
        observer_q, observer_qdot = observer.get_solver_state(separate=True)
        q_observed.append(observer_q.copy())
        qdot_observed.append(observer_qdot.copy())
    q_observed = np.array(q_observed)
    qdot_observed = np.array(qdot_observed)

    if show:
        plot_trajectory(
            goal=q_list,
            experiments={"observer": q_observed, "error": q_list - q_observed[:-1, :]},
            header="Position q:\n",
        )
        plot_trajectory(
            goal=qdot_list,
            experiments={
                "observer": qdot_observed,
                "error": qdot_list - qdot_observed[:-1, :],
            },
            header="Velocities qdot:\n",
        )

    if err_measure == "avg":
        eval = np.average
    else:
        eval = np.max
    q_err = eval(np.linalg.norm(q_list - q_observed[:-1, :], axis=1))
    qdot_err = eval(np.linalg.norm(qdot_list - qdot_observed[:-1, :], axis=1))
    return q_err, qdot_err


def loop_traj():
    min_q_err = np.inf
    min_qdot_err = np.inf
    Lp = None
    Ld = None
    Li = None

    idx = 0

    """
    for lp_gain in np.arange(0.1, 1.0, 0.05):
        for ld_gain in np.arange(0, 100, 5):
            for li_gain in np.arange(0, 1000, 25):
    """
    for lp_gain in np.arange(0.6, 1.4, 0.1):
        for ld_gain in np.arange(30, 100, 5):
            for li_gain in np.arange(500, 1000, 50):
                observer = LuenbergerObserver(
                    pin_model=deepcopy(wam_model.pin_model),
                    pin_data=deepcopy(wam_model.pin_data),
                    # TODO: check this
                    q_wam=q_wam_start,
                    q_pend=q_pend_start,
                    Lp=lp_gain * np.ones(6),
                    Ld=ld_gain * np.ones(6),
                    Li=li_gain * np.ones(6),
                    integration_time_step=0.002,
                    use_friction_action_model=True,
                    armature=wam_model.mj_model.dof_armature.copy(),
                    friction_model=FrictionModel(
                        coulomb=wam_model.mj_model.dof_frictionloss.copy(),
                        viscous=wam_model.mj_model.dof_damping.copy(),
                    ),
                )
                q_err, qdot_err = observer_traj(
                    q_list, qdot_list, torque_list, observer=observer, show=False
                )
                if idx % 10 == 0:
                    print(f"Idx: {idx}")
                    print(f"{q_err}")
                    print(f"{qdot_err}")

                if qdot_err <= min_qdot_err:
                    min_qdot_err = qdot_err
                    min_q_err = q_err
                    print(f"Min: {min_qdot_err}")
                    Lp = lp_gain
                    Ld = ld_gain
                    Li = li_gain
                idx += 1

    # ==== EVALUATION ====
    print("Minimum q error: {}".format(min_q_err))
    print("Minimum qdot error: {}".format(min_qdot_err))
    print("Gains: Lp = {}, Ld = {}, Li = {}".format(Lp, Ld, Li))
    return Lp, Ld, Li


if __name__ == "__main__":
    Lp, Ld, Li = loop_traj()
    # np.random.seed(1234)
    # Lp, Ld, Li = 0.25, 5, 0
    observer = LuenbergerObserver(
        pin_model=deepcopy(wam_model.pin_model),
        pin_data=deepcopy(wam_model.pin_data),
        q_wam=q_wam_start,
        q_pend=q_pend_start,
        Lp=Lp * np.ones(6),
        Ld=Ld * np.ones(6),
        Li=Li * np.ones(6),
        integration_time_step=0.002,
        use_friction_action_model=True,
        armature=wam_model.mj_model.dof_armature.copy(),
        friction_model=FrictionModel(
            coulomb=wam_model.mj_model.dof_frictionloss.copy(),
            viscous=wam_model.mj_model.dof_damping.copy(),
        ),
    )
    err = observer_traj(q_list, qdot_list, torque_list, observer=observer, show=True)
    plt.show()
