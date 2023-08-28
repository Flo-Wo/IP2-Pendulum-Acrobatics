import os
import pathlib

import numpy as np

from utils.config.action_model_gravity_compensation import FrictionModel
from utils.config.config_ip import ConfigWAM
from utils.solver.MPC.mpc_helper import LuenbergerObserver

if __name__ == "__main__":
    np.set_printoptions(precision=9)

    model = ConfigWAM(
        model_name="rotated",
        pendulum_len=0.374,
        use_friction_action_model=True,
    )
    print("Coulomb: ", model.mj_model.dof_frictionloss.copy())
    print("Viscous: ", model.mj_model.dof_damping.copy())
    print("Armature: ", model.mj_model.dof_armature.copy())

    # test data
    q_test = model.q_config
    print("q config: ", q_test)
    qdot_test = np.zeros_like(q_test)
    curr_time = 0.0

    observer = LuenbergerObserver(
        pin_model=model.pin_model,
        pin_data=model.pin_data,
        q_wam=q_test[:4],
        q_pend=q_test[-2:],
        Lp=0.25 * np.ones(6),
        Ld=5 * np.ones(6),
        Li=0 * np.ones(6),
        integration_time_step=0.002,
        use_friction_action_model=True,
        armature=model.mj_model.dof_armature.copy(),
        friction_model=FrictionModel(
            coulomb=model.mj_model.dof_frictionloss.copy(),
            viscous=model.mj_model.dof_damping.copy(),
        ),
    )
    folder = pathlib.Path(__file__).parent.parent.parent / "data/test_suite"
    state_list = np.load(os.path.join(folder, "test_observer_states.npy"))
    torque_list = np.load(os.path.join(folder, "test_observer_torques.npy"))

    np.savetxt(os.path.join(folder, "test_observer_states.txt"), state_list)
    np.savetxt(os.path.join(folder, "test_observer_torques.txt"), torque_list)

    # observe the wam 8 times and the pendulum twice
    for time_idx in range(100):
        q = state_list[time_idx, :4]
        tau = np.zeros(6)
        tau[:4] = torque_list[time_idx, :4]
        print("\n\nIdx = {}".format(time_idx))
        print("tau = ", tau)
        print("observation = ", state_list[time_idx, : 6 if time_idx % 4 == 0 else 4])
        observer.observe(
            time_idx,
            wam_q=state_list[time_idx, :4],
            pend_q=state_list[time_idx, 4:6] if time_idx % 4 == 0 else None,
            torque=tau,
            print_error=True,
        )
        pos, vel = observer.get_solver_state(separate=True)
        print("pos = ", pos)
        print("vel = ", vel)
    print("\n\nFinal Observation: ")
    pos, vel = observer.get_solver_state(separate=True)
    print("pos = ", pos)
    print("vel = ", vel)
