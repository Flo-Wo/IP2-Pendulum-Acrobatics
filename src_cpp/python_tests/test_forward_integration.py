import numpy as np

from utils.config.action_model_gravity_compensation import FrictionModel
from utils.config.config_ip import ConfigWAM
from utils.solver.MPC.mpc_helper import ForwardIntegration, TorqueStateWithTime

if __name__ == "__main__":

    model = ConfigWAM(
        model_name="rotated",
        pendulum_len=0.374,
        use_friction_action_model=True,
    )
    print("Coulomb: ", model.mj_model.dof_frictionloss.copy())
    print("Viscous: ", model.mj_model.dof_damping.copy())
    print("Armature: ", model.mj_model.dof_armature.copy())
    forward_integrator = ForwardIntegration(
        pin_data=model.pin_data,
        pin_model=model.pin_model,
        integration_time_step=0.002,
        use_friction_action_model=True,
        friction_model=FrictionModel(
            coulomb=model.mj_model.dof_frictionloss.copy(),
            viscous=model.mj_model.dof_damping.copy(),
        ),
        armature=model.mj_model.dof_armature.copy(),
    )

    # test data
    q_test = model.q_config
    print("q config: ", q_test)
    qdot_test = np.zeros_like(q_test)
    curr_time = 0.0
    num_steps_to_integrate = 10
    factor_integration_time = 2

    tau_list = []
    state_list = []
    for time_idx in range(num_steps_to_integrate):
        tau_list.append(time_idx * 0.02 * np.ones(6))
        state_list.append(np.zeros(6))

    torque_state_time = TorqueStateWithTime(
        solver_torques=tau_list,
        solver_states=state_list,
        curr_time=curr_time,
        factor_integration_time=factor_integration_time,
    )

    # test results
    solver_state, solver_time_idx = forward_integrator.semi_implicit_euler(
        q=q_test,
        dq=qdot_test,
        torque_state_with_time=torque_state_time,
        curr_time=curr_time,
        n_500Hz_steps_to_integrate=num_steps_to_integrate,
    )
    print("Final state: ")
    print(solver_state)
    print("Final time: ")
    print(solver_time_idx)
