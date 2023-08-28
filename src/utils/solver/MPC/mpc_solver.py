import logging
from copy import deepcopy
from datetime import datetime
from typing import Union

import crocoddyl
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from utils.config.action_model_gravity_compensation import FrictionModel
from utils.costs.integrated_models import IntCostModelsBase
from utils.decorators import log_time
from utils.solver.MPC.mpc_helper import (
    ForwardIntegration,
    StateObserver,
    TorqueStateWithTime,
    compute_additive_noise_on_observation,
    get_optitrack_observations,
)
from utils.solver.MPC.mpc_results import MPCResults
from utils.solver.MPC.mpc_warmstart import MPCWarmStart


# @remove_mj_logs
@log_time
def mpc(
    mujoco_model,
    mujoco_data,
    pin_model,
    pin_data,
    int_cost_models: Union[
        list[crocoddyl.IntegratedActionModelEuler], IntCostModelsBase
    ],
    terminal_model: crocoddyl.IntegratedActionModelEuler,
    mpc_horizon: int = 6,
    solver_max_iter: int = 100,
    solver: crocoddyl.SolverAbstract = crocoddyl.SolverBoxFDDP,
    continue_end: str = "repeat",
    start_with_static_warm: bool = True,
    custom_warmstart: MPCWarmStart = None,
    num_motors: int = 4,
    show_logs: bool = True,
    factor_integration_time: int = 1,
    n_fully_integrated_steps: int = 1,
    n_offset_500_Hz_steps: int = 0,
    use_forward_predictor: bool = True,
    use_state_observer: bool = False,
    use_wam_low_pass_filter: bool = True,
    use_pend_low_pass_filter: bool = True,
    noise_on_q_observation: float = 0.0,
    noise_on_pendulum_observation: float = 0.0,
    pendulum_observation_updates_with_125_Hz: bool = False,
    use_friction_action_model: bool = False,
) -> MPCResults:
    """Model Predictive Control Solver, based on crocoddyl's DDP solver.

    Parameters
    ----------
    model : mujoco model
        Mujoco model used for internal simulations to compute the next state
        after applying the MPC computed torque.
    mujoco_data : mujoco data
        Mujoco data used for internal simulations.
    int_cost_models : list[crocoddyl.IntegratedActionModelEuler]
        List with already integrated cost models, used in a circular append
        fashion by the DDP solver.
    terminal_node : crocoddyl.IntegratedActionModelEuler
        One single terminal cost model, which is used for
        EVERY ITERATION in the ddp solver (we only append new problems but
        the terminal node stays the same). Therefore the terminal model
        should be used with zero costs.
    time_horizon : int, optional
        Time horizon used by the MPC to plan ahead, by default 6.
    max_iter : int, optional
        Maximum number of iterations for the DDP solver, by default 100.
    cont : str, optional
        Default continuation type if the time_horizon exceeds the length of
        the desired trajectory, by default "repeat".
    warm_start_x : np.ndarray, optional
        Optional list of pairs (x_d, \dot{x}_d) used as a warmstart
        for the DDP solver, by default None.
    warm_start_u : np.ndarray, optional
        Optional list of controls (u) used as a warmstart
        for the DDP solver, by default None.
    static_warmstart : bool, optional
        Initialize the solver with a list of x0 and quasiStatic commands,
        by default False.
    num_motors : float, optional
        Number of motor commands, used e.g. for the wam with attached pendulum,
        where pinocchio internally uses 6 torque commands, but in reality the
        wam only has 4 actuators.
    show_logs : bool, optional
        Optional debugging parameter to show the logs,
        by default False.
    factor_integration_time : int, optional
        Factor to multiply the integration time with, default is 1. I.e. in
        the mujoco simulation we will execute each torque/command
        (factor_integration_time - 1) times additionally in order to match the
        integration time of crocoddyl.
    n_fully_integrated_steps : int, optional
        Number of torques you want to execute in full from the solver, if one, only
        the first torque is executed for its full length. If two, also the second torque
        is executed for factor_integration_times many times.
    n_offset_500_Hz_steps : int, optional
        Number of offset steps, where the following step is executed, but the solver does
        not receive a state update. If e.g. one and n_steps=1, then the zero'th torque is
        executed factor_integration_time many times and the first torque is executed once,
        but the solver does not obtain the updated state.
    use_forward_predictor: bool, optional
        Use a forward integration to predict the state of the real system in order to
        compensate for the time offset.
    use_friction_action_model: bool, optional
        Use the custom crocoddyl action model with friction, stiction and armature and including
        gravity compensation. Thus, also the forward predictor and the mujoco simulation has
        to be adjusted.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        MPC-internal simulation results: x, \dot{x}, u.

    Raises
    ------
    NotImplementedError
        Caused by missing/wrong continuation methods.
    """
    if custom_warmstart is None:
        custom_warmstart = MPCWarmStart([], [], False)
    # start the simulation, do not make a step here
    # mujoco.mj_step(model, mujoco_data)

    # solver time as an int, since it is resetted if the simulation crashes
    solver_time_idx: int = 0
    mujoco_time_idx: int = 0
    print("begin, solver_time: ", solver_time_idx)

    # use the initial state to start planning --> will be updated in the following
    solver_state = np.concatenate((mujoco_data.qpos, mujoco_data.qvel))

    results_data = MPCResults(
        solver_state[:],
        np.zeros(num_motors + 2),
        solver_state[:],
        mujoco_data.sensordata[:],
        0,
        factor_integration_time=factor_integration_time,
        mpc_horizon=mpc_horizon,
    )

    n_500Hz_cost_models = len(int_cost_models)
    mpc_horizon_in_500Hz = mpc_horizon * factor_integration_time

    duplicate_cost_models = list(range(n_500Hz_cost_models + mpc_horizon_in_500Hz))
    print("start, end: ", duplicate_cost_models[0], duplicate_cost_models[-1])

    def get_active_models(mujoco_time_idx):
        # factor 6 and horizon = 20
        # test = list(range(1000))
        # test[0:120:6] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54,
        # 60, 66, 72, 78, 84, 90, 96, 102, 108, 114]
        # end_time_idx = mpc_horizon * factor_integration_time
        print("Mujoco time: ", mujoco_time_idx)
        mj_time_plus_mpc_horizon = mujoco_time_idx + mpc_horizon_in_500Hz
        print(
            duplicate_cost_models[
                mujoco_time_idx:mj_time_plus_mpc_horizon:factor_integration_time
            ]
        )
        return int_cost_models[
            mujoco_time_idx:mj_time_plus_mpc_horizon:factor_integration_time
        ]

    # duplicate each model --> to maybe improve the performance
    active_models = int_cost_models[:mpc_horizon_in_500Hz:factor_integration_time]
    print(duplicate_cost_models[:mpc_horizon_in_500Hz:factor_integration_time])

    # how should the mpc continue to plan ahead when reaching the end nodes
    # reapeat the current trajectory --> add simple copies to the already enqueued cost models
    # we stay on 500Hz -> factor the end
    if continue_end == "repeat":
        int_cost_models.extend(int_cost_models[:mpc_horizon_in_500Hz])
    elif continue_end == "stationary":
        int_cost_models.extend([int_cost_models[-1]] * mpc_horizon_in_500Hz)
    else:
        raise NotImplementedError
    print(len(int_cost_models))

    problem = crocoddyl.ShootingProblem(solver_state, active_models, terminal_model)

    first_call = True

    class DiffDebug:
        def __init__(self):
            self.fd_mujoco_diffs = np.zeros((1, 6))
            self.fd_observations = np.zeros((1, 6))

        def update(self, qdot_wam, fd_filtered):
            self.fd_mujoco_diffs = np.concatenate(
                (self.fd_mujoco_diffs, qdot_wam[None, :]), axis=0
            )
            self.fd_observations = np.concatenate(
                (self.fd_observations, fd_filtered[None, :]), axis=0
            )

    diff_debug = DiffDebug()
    # DEBUG
    err_mj_vs_not_integrated = []
    err_mj_vs_integrated = []
    err_mj_vs_integrated_internal_ddp = []
    err_integrated_vs_planned = []

    # np.random.seed(1234)

    # ====== HELPER FUNCTIONS: OPTI TRACK UND WAM OBSERVER ======
    def _get_state(
        separate: bool = False, observation: bool = False, noise_on_q: float = 0.0
    ) -> np.ndarray:
        """Get the current state of the mujoco simulation."""
        mj_qpos = deepcopy(mujoco_data.qpos)
        mj_qvel = deepcopy(mujoco_data.qvel)

        # we can only observe the first 4 joints
        if observation:
            # only return the position
            mj_qpos = mj_qpos[:4]
            mj_qvel = mj_qvel[:4]

            # add noise, scaled by the given factor
            if noise_on_q > 0:
                qpos_add_noise = compute_additive_noise_on_observation(
                    mj_qpos, noise_factor=noise_on_q
                )
                mj_qpos += qpos_add_noise
            return mj_qpos, mj_qvel

        if not separate:
            return np.concatenate((mj_qpos, mj_qvel))
        return mj_qpos, mj_qvel

    def _get_optitrack_data(mujoco_time: int, noise_on_pendulum: float = 0.0):
        # if we want updates with 125Hz then only every 4th step will cause an update
        if pendulum_observation_updates_with_125_Hz and not (mujoco_time % 4 == 0):
            return None
        optitrack_observation = get_optitrack_observations(mj_data=mujoco_data)
        if noise_on_pendulum > 0:
            optitrack_add_noise = compute_additive_noise_on_observation(
                optitrack_observation, noise_factor=noise_on_pendulum
            )
            optitrack_observation += optitrack_add_noise
        return optitrack_observation

    # apply the torque for the integrated duration, cache the intermediate results
    def _apply_torque(
        torque_state_time: TorqueStateWithTime,
        solved: bool,
        n_times: int,
        mujoco_time_idx: int,
    ) -> int:
        for idx in range(n_times):
            (
                torque_with_time,
                ddp_state_with_time,
            ) = torque_state_time.get_solver_torque_and_state(mujoco_time_idx)
            torque = torque_with_time.vec
            ddp_state = ddp_state_with_time.vec

            # additional gravity term compute by pinocchio (like on the real system)
            if use_friction_action_model:
                # compensate for non-planned friction
                # TODO(friction): check if this is correct
                # torque += np.sign(mujoco_data.qvel) * mujoco_model.dof_frictionloss
                pass
            mujoco_data.ctrl[:num_motors] = torque[:num_motors]  # apply torque

            # print("idx: ", idx)
            # print("curr_mujoco_time: ", mujoco_data.time)
            # print("time stamp of the torque: ", torque_with_time.time_stamp)
            # print("apply torque: ", torque)

            # perform a mujoco step
            mujoco_time_idx += 1
            mujoco.mj_step(mujoco_model, mujoco_data)

            # ======= UPDATE THE STATE OBSERVER =======
            q_wam, qdot_wam = _get_state(
                separate=True,
                observation=True,
                noise_on_q=noise_on_q_observation,
            )
            # can be None, depending on the current time step
            optitrack_data = _get_optitrack_data(
                mujoco_time=mujoco_time_idx,
                noise_on_pendulum=noise_on_pendulum_observation,
            )
            state_observer.update(
                mujoco_time=mujoco_time_idx,
                torque=torque,
                wam_qpos=q_wam,
                optitrack_observation=optitrack_data,
            )

            fd_filtered_wam = state_observer.update_wam_observation(q_pos=q_wam)

            # TODO: remove after debugging
            _, qdot_perfect = _get_state(separate=True)
            _, qdot_observation = state_observer.get_solver_state(separate=True)
            """
            print(
                "   q: ||Observe - Real|| = ",
                np.linalg.norm(
                    _get_state(separate=True)[0]
                    - state_observer.get_solver_state(separate=True)[0]
                ),
            )
            print(
                "qdot: ||Observe - Real|| = ",
                np.linalg.norm(
                    _get_state(separate=True)[1]
                    - state_observer.get_solver_state(separate=True)[1]
                ),
            )
            print(
                "      ||Observe - Real|| = ",
                np.linalg.norm(_get_state() - state_observer.get_solver_state()),
            )
            diff_debug.update(qdot_perfect, qdot_observation)
            print(
                "   q: ||Observe - Real|| = ",
                np.linalg.norm(
                    _get_state(separate=True)[0]
                    - state_observer.get_solver_state(separate=True)[0]
                ),
            )
            print(
                "qdot: ||Observe - Real|| = ",
                np.linalg.norm(
                    _get_state(separate=True)[1]
                    - state_observer.get_solver_state(separate=True)[1]
                ),
            )
            print(
                "      ||Observe - Real|| = ",
                np.linalg.norm(_get_state() - state_observer.get_solver_state()),
            )
            """

            # ======= UPDATE THE RESULTS DATA =======
            state = _get_state()
            observer_state = state_observer.get_solver_state()
            results_data.add_data(
                ddp_x=ddp_state,
                u=torque,
                state=state,
                mj_sensor=deepcopy(mujoco_data.sensordata),
                solved=solved,
                observer_state=observer_state,
            )
        return mujoco_time_idx

    # create solver for this OC problem --> only once to save memory
    ddp = solver(problem)

    # ====== OPTI TRACK AND FORWARD INTEGRATOR =======

    # model a predictor via forward integration
    mujoco_time_stepsize = 0.002
    forward_integrator = ForwardIntegration(
        pin_model=deepcopy(pin_model),
        pin_data=deepcopy(pin_data),
        integration_time_step=mujoco_time_stepsize,
        use_friction_action_model=use_friction_action_model,
        friction_model=FrictionModel(
            coulomb=mujoco_model.dof_frictionloss.copy(),
            viscous=mujoco_model.dof_damping.copy(),
        ),
        armature=mujoco_model.dof_armature.copy(),
    )

    # DEBUG
    q_list_ddp = np.zeros((n_offset_500_Hz_steps + 1, 6))
    qdot_list_ddp = np.zeros((n_offset_500_Hz_steps + 1, 6))
    # modeling of OptiTrack
    state_observer = StateObserver(
        pin_model=deepcopy(pin_model),
        pin_data=deepcopy(pin_data),
        q_wam=mujoco_data.qpos[:4],
        q_pend=np.zeros(2),
        wam_dt=mujoco_time_stepsize,
        pend_dt=mujoco_time_stepsize
        if not pendulum_observation_updates_with_125_Hz
        else 4 * mujoco_time_stepsize,
        use_filter_wam=use_wam_low_pass_filter,
        use_filter_pend=use_pend_low_pass_filter,
        use_friction_action_model=use_friction_action_model,
        friction_model=FrictionModel(
            coulomb=mujoco_model.dof_frictionloss.copy(),
            viscous=mujoco_model.dof_damping.copy(),
        ),
        armature=mujoco_model.dof_armature.copy(),
    )

    while mujoco_time_idx < n_500Hz_cost_models:
        # the problem stays the same, but the initial state changes
        problem.x0 = solver_state

        # ==== LOGGING FOR DEBUGGING ====
        if mujoco_time_idx % 100 == 0 and show_logs:
            print("\nINDEX: {}/{}".format(mujoco_time_idx, n_500Hz_cost_models))
            log = crocoddyl.CallbackLogger()
            ddp.setCallbacks([log, crocoddyl.CallbackVerbose()])

        # check for the first time the MPC is called --> static warmstart and
        # we have to define the warmstart for the next calls
        if first_call:
            if start_with_static_warm and not custom_warmstart:
                logging.info("MPC: used a static warmstart for the first step.")
                x0 = np.concatenate(
                    (mujoco_data.qpos, np.zeros(np.shape(mujoco_data.qvel)))
                )
                custom_warmstart.set_warm(
                    [x0] * (problem.T + 1), problem.quasiStatic([x0] * problem.T), True
                )
            # only perform this step once
            first_call = False

        # TODO: check only quasi static warm starts
        custom_warmstart.set_warm(
            [solver_state] * (problem.T + 1),
            problem.quasiStatic([solver_state] * problem.T),
            True,
        )
        warm_x, warm_u, is_feasible = custom_warmstart.get_warm()

        print("INDEX: {}/{}".format(mujoco_time_idx, n_500Hz_cost_models))
        print("mj_time - solver_time: {}".format(mujoco_time_idx - solver_time_idx))

        start = datetime.now()
        solved = ddp.solve(warm_x, warm_u, solver_max_iter, is_feasible)
        delta = datetime.now() - start
        # compute the time delta in seconds
        results_data.add_computation_time(delta.total_seconds())

        torque_state_time = TorqueStateWithTime(
            solver_torques=ddp.us,
            solver_states=ddp.xs,
            curr_time=solver_time_idx,
            factor_integration_time=factor_integration_time,
        )
        # solved = ddp.solve()

        # set the parameters for the next warmstart
        # custom_warmstart.set_warm(ddp.xs, ddp.us, solved)

        # ===== MUJOCO SIMULATION: CONTROL DELAY IMPLEMENTATION =====

        # ===== OBSERVABLE STEPS FOR THE SOLVER =====
        # apply the first n_steps torques fully, afterwards, we cache the state for the solver
        for idx in range(n_fully_integrated_steps):
            n_times = factor_integration_time
            print(f"n_times = {n_times}")
            mujoco_time_idx = _apply_torque(
                torque_state_time=torque_state_time,
                solved=solved,
                n_times=factor_integration_time,
                mujoco_time_idx=mujoco_time_idx,
            )

        if not use_state_observer:
            # cache the current state and the time to simulate offsets between sim and robot
            solver_state = _get_state()
        else:
            # get the solve state via the observation
            solver_state = state_observer.get_solver_state()

        solver_time_idx = mujoco_time_idx

        solver_state_before_integration = deepcopy(solver_state)
        solver_time_idx_before_integation = deepcopy(mujoco_time_idx)

        if use_forward_predictor:
            if not use_state_observer:
                # perfect observation
                obs_solver_state_qpos, obs_solver_state_qvel = _get_state(separate=True)
            else:
                (
                    obs_solver_state_qpos,
                    obs_solver_state_qvel,
                ) = state_observer.get_solver_state(separate=True)
            solver_state, solver_time_idx = forward_integrator.semi_implicit_euler(
                q=obs_solver_state_qpos.copy(),
                dq=obs_solver_state_qvel.copy(),
                torque_state_with_time=torque_state_time,
                curr_time=solver_time_idx,
                n_500Hz_steps_to_integrate=n_offset_500_Hz_steps,
            )

        # ===== NON-OBSERVABLE STEPS FOR THE SOLVER =====

        # apply the first n_steps torques fully, afterwards, we cache the state for the solver
        # execute steps individually, this time only once
        # after factor_integration_time steps, we have to increase the torque index by one

        # DEBUG: log the integration
        q_list_mujoco = np.zeros((n_offset_500_Hz_steps + 1, 6))
        qdot_list_mujoco = np.zeros((n_offset_500_Hz_steps + 1, 6))
        q_list_mujoco[0, :] = mujoco_data.qpos.copy()
        qdot_list_mujoco[0, :] = mujoco_data.qvel.copy()
        # ENDDEBUG

        for idx in range(n_offset_500_Hz_steps):
            mujoco_time_idx = _apply_torque(
                torque_state_time=torque_state_time,
                solved=solved,
                n_times=1,
                mujoco_time_idx=mujoco_time_idx,
            )
            q_list_mujoco[idx + 1, :] = mujoco_data.qpos.copy()
            qdot_list_mujoco[idx + 1, :] = mujoco_data.qvel.copy()

        # DEBUG: compute errors
        diff_mj_vs_not_integrated = np.linalg.norm(
            solver_state_before_integration - _get_state()
        )
        diff_mj_vs_integrated = np.linalg.norm(solver_state - _get_state())
        (
            _,
            solver_planned_internal_state,
        ) = torque_state_time.get_solver_torque_and_state(mujoco_time_idx)

        diff_integrated_vs_planned = np.linalg.norm(
            solver_planned_internal_state.vec - solver_state
        )
        diff_mj_vs_integrated_internal_ddp = np.linalg.norm(
            solver_planned_internal_state.vec - _get_state()
        )

        err_mj_vs_not_integrated.append(diff_mj_vs_not_integrated)
        err_mj_vs_integrated.append(diff_mj_vs_integrated)
        err_integrated_vs_planned.append(diff_integrated_vs_planned)
        # print("Mujoco state - not integrated state: ", diff_mj_vs_not_integrated)
        # print("Mujoco state - solver input: ", diff_mj_vs_integrated)
        # print("integrated state - planned state: ", diff_integrated_vs_planned)
        # print("Curr Mujoco state - DDP state: ", diff_mj_vs_integrated_internal_ddp)
        err_mj_vs_integrated_internal_ddp.append(diff_mj_vs_integrated_internal_ddp)
        # (
        #     _,
        #     solver_planned_internal_state_before_int,
        # ) = torque_state_time.get_solver_torque_and_state(
        #     solver_time_idx_before_integation
        # )
        # print(
        #     "Before integration: \n" + "DDP  - not integrated state: ",
        #     np.linalg.norm(
        #         solver_planned_internal_state_before_int.vec
        #         - solver_state_before_integration
        #     ),
        # )
        # print("finished iter")
        for next_active_model in get_active_models(mujoco_time_idx=solver_time_idx):
            problem.circularAppend(next_active_model)

    # plot fd computation
    # for x in range(6):
    #     plt.figure()
    #     plt.plot(diff_debug.fd_observations[:, x], label="filterd diff")
    #     plt.plot(diff_debug.fd_mujoco_diffs[:, x], label="mujoco perfect observation")
    #     plt.title("Index {}".format(x))
    #     plt.legend()
    # plt.show()
    # plt.figure()
    # plt.plot(err_mj_vs_not_integrated, label="Mujoco vs. not integrated")
    # plt.plot(err_mj_vs_integrated, label="Mujoco vs. integrated (pin observer)")
    # plt.plot(err_integrated_vs_planned, label="Internal DDP int vs. pin observer int")
    # plt.plot(err_mj_vs_integrated_internal_ddp, label="DDP integrated vs Mujoco")
    # plt.legend()
    # plt.show()
    print("Average errors: ")
    print("avg: s_n - tilde_s_n: ", np.average(np.array(diff_mj_vs_integrated)))
    return results_data
