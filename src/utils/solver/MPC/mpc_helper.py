import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

from utils.config.action_model_gravity_compensation import FrictionModel


class ArrayWithTimestamp:
    def __init__(self, vec: np.array, time_stamp: float):
        self.vec = vec
        self.time_stamp = time_stamp

    def __repr__(self):
        return "{}: {}".format(self.time_stamp, self.vec)


class TorqueStateWithTime:
    def __init__(
        self,
        solver_torques: List[np.array],
        solver_states: List[np.array],
        curr_time: float,
        factor_integration_time: int,
    ):

        self.torque_list: List[ArrayWithTimestamp] = []
        self.state_list: List[ArrayWithTimestamp] = []
        time = curr_time
        for idx, tau in enumerate(solver_torques):
            for _ in range(factor_integration_time):
                torque_with_time = ArrayWithTimestamp(tau, time)
                state_with_time = ArrayWithTimestamp(solver_states[idx], time)
                self.torque_list.append(torque_with_time)
                self.state_list.append(state_with_time)
                time += 1

    def get_solver_torque_and_state(self, curr_time):
        for idx, tau_time in enumerate(self.torque_list):
            if tau_time.time_stamp == curr_time:
                # tau_default = np.zeros_like(tau_time)
                # tau_default[:4] = tau_time.vec[:4]
                tau_time.vec[-2:] = np.zeros(2)
                return tau_time, self.state_list[idx]


def compute_additive_noise_on_observation(
    observation: np.ndarray,
    noise_factor: float,
    deterministic_length: bool = True,
) -> np.ndarray:
    """Return additive noise with random dir and length <= noise_factor"""
    # wam case: noise on each joint
    if observation.ndim == 1:
        add_noise = np.random.uniform(
            low=(-1) * noise_factor, high=noise_factor, size=observation.shape
        )
        return add_noise
    # optitrack case: noise per observation vector with bounded norm
    random_dir = np.random.uniform(size=observation.shape)
    norms = np.linalg.norm(random_dir, axis=1, keepdims=True)
    if not deterministic_length:
        length = np.random.uniform(
            low=(-1) * noise_factor, high=noise_factor, size=norms.shape
        )
    else:
        length = noise_factor
    add_noise = random_dir / norms * length
    return add_noise


def get_optitrack_observations(mj_data):
    """Return the last four 3D vectors simulating the optitrack position."""
    return np.array(
        [
            mj_data.sensordata[-12:-9].copy(),
            mj_data.sensordata[-9:-6].copy(),
            mj_data.sensordata[-6:-3].copy(),
            mj_data.sensordata[-3:].copy(),
        ]
    )


def compute_joint_angles(
    input_vector: np.ndarray,
):
    """Compute the joint angles given a 3D vector in space.

    This function simply reverts the above function and thus depends on the WAMs config.
    """
    r = math.sqrt(input_vector[1] ** 2 + input_vector[2] ** 2)
    phi = math.atan2(input_vector[0], r)
    theta = (-1) * math.atan2(input_vector[1], input_vector[2])

    return np.array([theta, phi])


def lstsq_on_x_coordinate(data):
    X = data[:, 1:]  # y-z are the input
    Y = data[:, 0]  # x is the target
    # first return value is the actual solution
    theta = np.array(
        np.linalg.lstsq(
            np.concatenate((X, np.ones((X.shape[0], 1))), axis=1), Y, rcond=None
        )[0]
    )
    # insert the first two datapoints, compute the difference and obtain the direction vector
    first_two_data_points_y_z = data[:2, 1:]
    first_two_data_points_x = (
        np.concatenate(
            (
                first_two_data_points_y_z,
                np.ones((first_two_data_points_y_z.shape[0], 1)),
            ),
            axis=1,
        )
        @ theta
    )

    first_two_data_points = np.concatenate(
        (first_two_data_points_x[:, None], first_two_data_points_y_z), axis=1
    )
    pend_vector_estimate = first_two_data_points[1, :] - first_two_data_points[0, :]
    return pend_vector_estimate / np.linalg.norm(pend_vector_estimate)


class LowPassFilter:
    def __init__(self, omega_p: np.ndarray, dc_gain: float = 1.0):
        self.a = np.zeros_like(omega_p)
        self.b = dc_gain * omega_p
        self.c = omega_p

        # previous state observation
        self.x_1 = np.zeros_like(omega_p)
        # running average
        self.y_1 = np.zeros_like(omega_p)

    def set_coefficients(
        self, time_step: float = 0.002, x_1: Optional[np.ndarray] = None
    ):
        # TODO: remove after debugging
        self.time_step = time_step

        if x_1 is not None:
            self.x_1 = x_1

        den = 1.0 + self.c * time_step
        self.c1 = 1.0 / den
        self.c2 = (self.a + self.b * time_step) / den
        self.c3 = self.a / den

    def compute_diff(self, x_0: np.ndarray) -> np.ndarray:
        y_0 = self.c1 * self.y_1 + self.c2 * x_0 - self.c3 * self.x_1

        self.y_1 = y_0
        self.x_1 = x_0
        return y_0


class LuenbergerObserver:
    def __init__(
        self,
        pin_model,
        pin_data,
        q_wam: np.ndarray,
        q_pend: np.ndarray,
        Lp: np.ndarray,
        Ld: np.ndarray,
        Li: np.ndarray,
        integration_time_step: float = 0.002,
        use_friction_action_model: bool = False,
        armature: Optional[np.ndarray] = None,
        friction_model: Optional[FrictionModel] = None,
    ) -> None:
        """https://en.wikipedia.org/wiki/State_observer"""
        self.pin_model = pin_model
        self.pin_data = pin_data

        self.Lp = Lp
        self.Ld = Ld
        self.Li = Li

        self.time_step = integration_time_step
        self.last_torque = np.zeros(6)
        self.last_time = 0

        # disturbance
        # pos and velocity for the wam and the pendulum
        qdot_wam = np.zeros_like(q_wam)
        qdot_pend = np.zeros_like(q_pend)

        self.position = np.concatenate((q_wam, q_pend))
        self.velocity = np.concatenate((qdot_wam, qdot_pend))

        self.state_disturbance = np.zeros_like(self.position)

        self.friction_model = friction_model
        self.armature = armature
        self.use_friction_action_model = use_friction_action_model

    def observe(
        self,
        time: float,
        wam_q: np.ndarray = None,
        # wam_qdot: np.ndarray,
        pend_q: np.ndarray = None,
        # pend_qdot: np.ndarray = None,
        torque: np.ndarray = None,
        print_error: bool = False,
    ) -> None:

        time_delta = (time - self.last_time) * self.time_step

        # save torque and time
        if torque is not None:
            self.last_torque = torque
        else:
            self.last_torque = np.zeros(6)
        # state transition update with simplectic euler integration

        # M_inv = pin.computeMinverse(self.pin_model, self.pin_data, self.position)
        # cg = pin.nonLinearEffects(
        #     self.pin_model, self.pin_data, self.position, self.velocity
        # )
        # ddq = M_inv @ (self.last_torque - cg) + self.state_disturbance
        n_substeps = 5
        delta_t = self.time_step / n_substeps

        for i in range(n_substeps):
            pin.computeAllTerms(
                self.pin_model,
                self.pin_data,
                self.position.copy(),
                self.velocity.copy(),
            )
            mass_matrix = self.pin_data.M
            right_side = self.last_torque.copy() - self.pin_data.nle
            if self.use_friction_action_model:
                mass_matrix += np.diag(self.armature)
                right_side += self.friction_model.calc(self.velocity.copy())

            ddq = np.linalg.solve(mass_matrix, right_side)

            # add state disturbance to acceleration
            ddq += self.state_disturbance

            self.velocity += ddq * delta_t
            self.position += self.velocity * delta_t

        if (pend_q is None) and (wam_q is None):
            raise NotImplementedError("Either q_wam or q_pend have to be not None.")
        if pend_q is None:
            # if we do not get an update of the pendulum, we use the predicted state
            pend_q = self.position[-2:]
        if wam_q is None:
            wam_q = self.position[:-2]

        pos = np.concatenate((wam_q, pend_q))
        err = pos - self.position

        if print_error:
            print("err = {}".format(err))
            print("||err||: {}".format(np.linalg.norm(err)))
        if np.linalg.norm(err) > 1 and print_error:
            print("large error")

        # observation update with constant observer gains
        self.position += self.Lp * err
        self.velocity += self.Ld * err
        self.state_disturbance += self.Li * err

        self.last_time = time

    def get_solver_state(self, separate: bool = False) -> np.ndarray:
        if separate:
            return self.position, self.velocity
        return np.concatenate((self.position, self.velocity))


class StateObserver:
    def __init__(
        self,
        pin_model,
        pin_data,
        q_wam: np.ndarray,
        q_pend: np.ndarray = None,
        pend_vector: np.ndarray = None,
        wam_dt: float = 0.002,
        pend_dt: float = 0.002,
        use_filter_wam: bool = True,
        use_filter_pend: bool = True,
        use_friction_action_model: bool = True,
        armature: Optional[np.ndarray] = None,
        friction_model: Optional[FrictionModel] = None,
    ) -> None:
        logging.info("Using WAM low-pas: {}".format(use_filter_wam))
        logging.info("Using Pend low-pas: {}".format(use_filter_pend))
        self.pin_model = pin_model
        self.pin_data = pin_data

        # TODO: change to measured time on real system
        self.wam_dt = wam_dt
        self.pend_dt = pend_dt

        # we can turn off the filters
        self.use_filter_wam = use_filter_wam
        self.use_filter_pend = use_filter_pend
        # we need to fill the config to compute the orientation
        self.two_zeros = np.zeros(2)

        self.curr_q_wam: np.ndarray = q_wam

        if pend_vector is not None:
            # Assumption: initial wam velocity is zero
            self.curr_q_pend: np.ndarray = self._compute_pend_q(
                q_wam, np.zeros_like(q_wam), pend_vector
            )
        else:
            self.curr_q_pend = q_pend

        # Assumption: initial wam velocity is zero
        self.curr_qdot_wam: np.ndarray = np.zeros_like(self.curr_q_wam)
        # Assumption: the initial pend velocity is zero
        self.curr_qdot_pend: np.ndarray = np.zeros_like(self.curr_q_pend)

        # define low pass filters for the wam and the pendulum velocities
        # params given by the internal WAM filter implementation, see
        # https://github.com/BarrettTechnology/libbarrett/blob/109faf6066b3961426f95fd9748a028cef5fcf16/config/wam4.conf#L79
        self.wam_low_pass = LowPassFilter(omega_p=180 * np.ones(4))
        self.wam_low_pass.set_coefficients(time_step=wam_dt)

        self.pend_low_pass = LowPassFilter(omega_p=200 * np.ones(2))
        self.pend_low_pass.set_coefficients(time_step=pend_dt)

        # Luenberger Observer to also use the internal model
        self.luenberger = LuenbergerObserver(
            pin_model=self.pin_model,
            pin_data=self.pin_data,
            q_wam=q_wam,
            q_pend=q_pend,
            # latest
            # 0.8, 50, 700
            # benchmark estimate: 0.25, 5, 0
            Lp=0.25 * np.ones(6),
            Ld=5 * np.ones(6),
            Li=0 * np.ones(6),
            integration_time_step=wam_dt,
            friction_model=friction_model,
            armature=armature,
            use_friction_action_model=use_friction_action_model,
        )

    def update(
        self,
        mujoco_time: int,
        torque: np.ndarray,
        wam_qpos: np.ndarray,
        optitrack_observation: np.ndarray = None,
    ):
        self.update_wam_observation(q_pos=wam_qpos)
        if optitrack_observation is not None:
            self.update_optitrack_observation(optitrack_observation)
            self.luenberger.observe(
                time=mujoco_time,
                wam_q=self.curr_q_wam,
                # wam_qdot=self.curr_qdot_wam,
                pend_q=self.curr_q_pend,
                # pend_qdot=self.curr_qdot_pend,
                torque=torque,
            )
        else:
            self.luenberger.observe(
                time=mujoco_time,
                wam_q=self.curr_q_wam,
                # wam_qdot=self.curr_qdot_wam,
                pend_q=None,
                # pend_qdot=None,
                torque=torque,
            )

    def update_optitrack_observation(self, optitrack_observation: np.ndarray):
        """
        - current orientation of the pendulum in the world-frame (computed via
        regression of opti-track measurements)
        """
        # compute the pendulum vector given the observations
        pend_vector = self._perform_lstsq_on_optitrack_observation(
            optitrack_observation=optitrack_observation
        )
        # compute qdot for the pendulum via finite differences
        new_q_pend = self._compute_pend_q(pend_vector)
        # fd_diff_pend = (new_q_pend - self.curr_q_pend) / self.pend_dt

        # if self.use_filter_pend:
        #     fd_diff_pend = self.pend_low_pass.compute_diff(fd_diff_pend)
        self.curr_q_pend = new_q_pend
        # self.curr_qdot_pend = fd_diff_pend

    def update_wam_observation(self, q_pos):
        """The update method receives:
        - current joint positions (without noise)
        - current joint velocities (including some noise, internally of the WAM)
        """
        fd_diff_wam = (q_pos - self.curr_q_wam) / self.wam_dt
        self.curr_q_wam = q_pos

        # if self.use_filter_wam:
        #     fd_diff_wam = self.wam_low_pass.compute_diff(fd_diff_wam)
        # self.curr_qdot_wam = fd_diff_wam

    def get_solver_state(self, separate: bool = False) -> np.ndarray:
        return self.luenberger.get_solver_state(separate=separate)
        # q = np.concatenate((self.curr_q_wam, self.curr_q_pend))
        # qdot = np.concatenate((self.curr_qdot_wam, self.curr_qdot_pend))
        # if separate:
        #     return q, qdot
        # return np.concatenate((q, qdot))

    def _perform_lstsq_on_optitrack_observation(
        self, optitrack_observation: np.ndarray
    ) -> np.ndarray:
        return lstsq_on_x_coordinate(data=optitrack_observation)

    def _compute_pend_q(self, pend_vector: np.ndarray) -> np.ndarray:
        rot_matrix = self._rot_matrix_to_local_frame(
            q_wam=self.curr_q_wam, qdot_wam=self.curr_qdot_wam
        )
        pend_vector_local_frame = rot_matrix.T @ pend_vector
        return compute_joint_angles(pend_vector_local_frame)

    def _rot_matrix_to_local_frame(
        self, q_wam: np.ndarray, qdot_wam: np.ndarray
    ) -> np.ndarray:
        """
        Reference frame in the world frame -> multiply with transpose to
        transform the pendulum vector from world-frame to reference frame.
        """
        pin.forwardKinematics(
            self.pin_model,
            self.pin_data,
            np.concatenate((q_wam, self.two_zeros)),
            np.concatenate((qdot_wam, self.two_zeros)),
        )
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        ref_frame_id = self.pin_model.getFrameId("links/pendulum/base")
        rot_matrix = self.pin_data.oMf[ref_frame_id].rotation
        return rot_matrix


class ForwardIntegration:
    def __init__(
        self,
        pin_data,
        pin_model,
        integration_time_step: float,
        use_friction_action_model: bool = False,
        armature: Optional[np.ndarray] = None,
        friction_model: Optional[FrictionModel] = None,
    ):
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.integration_time_step = integration_time_step

        # sanity checks for custom friction model
        if use_friction_action_model:
            assert use_friction_action_model == (
                friction_model is not None
            ), "Cannot use flag True without friction model and vice versa."

            assert use_friction_action_model == (
                armature is not None
            ), "Cannot use flag True without armature and vice versa."

            logging.info("\n\nForward Integration")
            print("Armature: ", armature)
            print("Damping: ", friction_model.viscous)
            print("Friction: ", friction_model.coulomb)
            logging.info("\n\n")
        self.use_friction_action_model = use_friction_action_model
        self.armature = armature
        self.friction_model = friction_model

    def semi_implicit_euler(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        torque_state_with_time: TorqueStateWithTime,
        curr_time: int,
        n_500Hz_steps_to_integrate: int,
        separate_state_return: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        q_list = np.zeros((n_500Hz_steps_to_integrate + 1, q.shape[0]))
        qdot_list = np.zeros((n_500Hz_steps_to_integrate + 1, q.shape[0]))
        q_list[0, :] = q
        qdot_list[0, :] = dq

        for idx in range(n_500Hz_steps_to_integrate):

            tau_time, _ = torque_state_with_time.get_solver_torque_and_state(
                curr_time=curr_time
            )
            tau = tau_time.vec

            # pin.forwardDynamics(self.pin_model, self.pin_data, q, dq, tau)
            if not self.use_friction_action_model:
                # perform ABA algorithm for forward dynamics
                pin.aba(self.pin_model, self.pin_data, q, dq, tau)
                ddq = self.pin_data.ddq
            else:
                # compute all terms for the dynamics
                pin.computeAllTerms(self.pin_model, self.pin_data, q, dq)
                mass_matrix = self.pin_data.M + np.diag(self.armature)

                ddq = np.linalg.solve(
                    mass_matrix,
                    tau
                    # + self.pin_data.g
                    + self.friction_model.calc(dq) - self.pin_data.nle,
                )

            dq += ddq * self.integration_time_step
            q = pin.integrate(self.pin_model, q, dq * self.integration_time_step)

            # DEBUG: log the forward integration
            q_list[idx + 1, :] = q
            qdot_list[idx + 1, :] = dq

            curr_time += 1

        if not separate_state_return:
            return np.concatenate((q, dq)), curr_time  # , q_list, qdot_list
        return q, dq, curr_time  # , q_list, qdot_list
