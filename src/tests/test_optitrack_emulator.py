import unittest
from copy import copy

import mujoco
import numpy as np
import numpy.testing as np_test
import pinocchio as pin

from utils.config.config_ip import ConfigWAM
from utils.solver.MPC.mpc_helper import (
    compute_additive_noise_on_observation,
    compute_joint_angles,
    get_optitrack_observations,
    lstsq_on_x_coordinate,
)

"""To compute the Rotation matrix in mujoco, add the following sensors to your xml-file

<framexaxis name="pend_x_axis_global" objtype="site" objname="pendulum/sensor_sites/reference_frame" reftype="site" refname="wam/ref_sites/global_origin"/>
<frameyaxis name="pend_y_axis_global" objtype="site" objname="pendulum/sensor_sites/reference_frame" reftype="site" refname="wam/ref_sites/global_origin"/>
<framezaxis name="pend_z_axis_global" objtype="site" objname="pendulum/sensor_sites/reference_frame" reftype="site" refname="wam/ref_sites/global_origin"/>

with a reference frame being the pendulums base:

<body name="pendulum/links/base" pos="-0.045 -0.35 0" euler="1.570796 0 1.570796">
    <site name="pendulum/sensor_sites/reference_frame" pos="0 0 0" size="0.005" rgba="0 1 0 1" />
"""


def _get_pend_tip_beg_world(mj_data):
    # rot_matrix = np.array(
    #     [
    #         mj_data.sensordata[9:12].copy(),
    #         mj_data.sensordata[12:15].copy(),
    #         mj_data.sensordata[15:18].copy(),
    #     ]
    # )
    return (
        mj_data.sensordata[0:3].copy(),
        mj_data.sensordata[6:9].copy(),
        # rot_matrix,
    )


def _get_rot_matrix_to_local_frame(pin_model, pin_data, qpos, qdot) -> np.ndarray:
    pin.forwardKinematics(pin_model, pin_data, qpos, qdot)
    pin.updateFramePlacements(pin_model, pin_data)
    ref_frame_id = pin_model.getFrameId("links/pendulum/base")
    rot_matrix = pin_data.oMf[ref_frame_id].rotation
    return rot_matrix


class TestPendulumAnglesInversion(unittest.TestCase):
    @classmethod
    def setUpClass(self, pendulum_len: float = 0.874):
        lower_joint_limit = -0.9
        upper_joint_limit = 0.9
        # lower_joint_limit = -np.pi / 2
        # upper_joint_limit = np.pi / 2
        grid_step_size = 0.01

        self.pendulum_len = pendulum_len

        self.angles = np.arange(
            start=lower_joint_limit,
            stop=upper_joint_limit,
            step=grid_step_size,
        )

        self.abs_err_in_noise_case = []
        self.rel_err_in_noise_case = []

    def tearDown(self) -> None:
        # Compute the average and median errors (absolute and relative)
        average_error_absolute = np.mean(self.abs_err_in_noise_case)
        median_error_absolute = np.median(self.abs_err_in_noise_case)
        average_error_relative = np.mean(self.rel_err_in_noise_case)
        median_error_relative = np.median(self.rel_err_in_noise_case)

        # Print the average and median errors
        print("==================== NOISE REPORT ====================")
        print(f"Average absolute error: {average_error_absolute}")
        print(f"Median absolute error: {median_error_absolute}")
        print(f"Average relative error: {average_error_relative}")
        print(f"Median relative error: {median_error_relative}")
        print("========================================")

    # def DEPRECATED_test_joint_angles_to_endeff(self):
    #     for phi in self.angles:
    #         for theta in self.angles:
    #             endeff = compute_endeffector_position(
    #                 pendulum_len=self.pendulum_len, joint_angles=(phi, theta)
    #             )
    #             phi_prime, theta_prime = compute_joint_angles(input_vector=endeff)
    #             endeff_prime = compute_endeffector_position(
    #                 pendulum_len=self.pendulum_len,
    #                 joint_angles=(phi_prime, theta_prime),
    #             )
    #             self._tuples_are_equal(desired=endeff, actual=endeff_prime)
    #             self._tuples_are_equal(
    #                 desired=np.array([phi, theta]),
    #                 actual=np.array([phi_prime, theta_prime]),
    #             )

    def test_perfect_joint_angles_given_tip_bottom(self):
        model = ConfigWAM("rotated", pendulum_len=0.174)
        pin_model = model.pin_model
        pin_data = model.pin_data
        mj_data = model.mj_data
        mj_model = model.mj_model

        for phi in self.angles:
            for theta in self.angles:
                q = pin.randomConfiguration(pin_model)
                q[-2:] = np.array([phi, theta])
                qdot = np.zeros(*np.shape(q))

                mj_data.qpos = q
                mj_data.qvel = qdot
                mujoco.mj_forward(mj_model, mj_data)

                (
                    pend_tip_world,
                    pend_begin_world,
                    # rot_matrix,
                ) = _get_pend_tip_beg_world(mj_data=mj_data)

                rot_matrix_pin = _get_rot_matrix_to_local_frame(
                    pin_model, pin_data, q, qdot
                )

                pend_vector = pend_tip_world - pend_begin_world
                pend_vector /= np.linalg.norm(pend_vector)
                # use the pinocchio version
                pend_vector_local_frame = rot_matrix_pin.T @ pend_vector
                # diff_vector_local_frame = rot_matrix @ diff_vector
                phi_prime, theta_prime = compute_joint_angles(
                    input_vector=pend_vector_local_frame
                )
                q_prime = copy(q)
                q_prime[-2:] = np.array([phi_prime, theta_prime])
                self._tuples_are_equal(q_prime[-2:], q[-2:])

    def test_optitrack_joint_angles(self):
        self._optitrack_joint_angles_in_mujoco(noise=False)

    def test_optitrack_joint_angles_with_noise(self):
        self._optitrack_joint_angles_in_mujoco(noise=True)

    def _optitrack_joint_angles_in_mujoco(self, noise: bool = False):
        model = ConfigWAM("rotated", pendulum_len=0.174)
        pin_model = model.pin_model
        pin_data = model.pin_data
        mj_data = model.mj_data
        mj_model = model.mj_model

        for phi in self.angles:
            for theta in self.angles:

                q = pin.randomConfiguration(pin_model)
                q[-2:] = np.array([phi, theta])
                qdot = np.zeros(*np.shape(q))

                mj_data.qpos = q
                mj_data.qvel = qdot
                mujoco.mj_forward(mj_model, mj_data)

                # old version
                (
                    pend_tip_world,
                    pend_begin_world,
                    # rot_matrix,
                ) = _get_pend_tip_beg_world(mj_data=mj_data)
                pend_vector_old = pend_tip_world - pend_begin_world
                pend_vector_old /= np.linalg.norm(pend_vector_old)

                # === new version via linear regression
                optitrack_data = get_optitrack_observations(mj_data=mj_data)

                # optitrack is accurate within 1mm
                if noise:
                    add_noise = compute_additive_noise_on_observation(
                        optitrack_data, 1e-3
                    )

                    optitrack_data += add_noise

                pend_vector = lstsq_on_x_coordinate(optitrack_data)
                if noise:
                    print(
                        "{:1.3e}".format(np.linalg.norm(pend_vector - pend_vector_old))
                    )

                if not noise:
                    self._tuples_are_equal(pend_vector, pend_vector_old)

                rot_matrix_pin = _get_rot_matrix_to_local_frame(
                    pin_model, pin_data, q, qdot
                )

                diff_vector_local_frame = rot_matrix_pin.T @ pend_vector
                phi_prime, theta_prime = compute_joint_angles(
                    input_vector=diff_vector_local_frame
                )
                q_prime = copy(q)
                q_prime[-2:] = np.array([phi_prime, theta_prime])
                self._tuples_are_equal(q_prime[-2:], q[-2:], noise=noise)

    def _tuples_are_equal(self, desired: tuple, actual: tuple, noise: bool = False):
        self.assertEqual(len(desired), len(actual))
        if not noise:
            np_test.assert_almost_equal(desired, actual, decimal=5)
        else:
            np_test.assert_allclose(
                actual=actual, desired=desired, rtol=1e-1, atol=1e-1
            )
            self.abs_err_in_noise_case.append(np.abs(actual - desired))
            self.rel_err_in_noise_case.append(np.abs((actual - desired) / desired))


def test_single_config():
    state = np.array(
        [
            7.60119690e-07,
            -7.80073704e-01,
            1.54597139e-07,
            2.37025114e00,
            -7.01935999e-04,
            -1.29420193e-06,
            3.80059845e-04,
            -3.68517624e-02,
            7.72985697e-05,
            1.25570426e-01,
            -3.50967999e-01,
            -6.47100963e-04,
        ]
    )
    q = state[:6]
    qdot = state[-6:]

    # ====== SAME TEST AS ABOVE =====
    model = ConfigWAM("rotated", pendulum_len=0.174)
    pin_model = model.pin_model
    pin_data = model.pin_data
    mj_data = model.mj_data
    mj_model = model.mj_model
    mj_data.qpos = q
    mj_data.qvel = qdot
    mujoco.mj_forward(mj_model, mj_data)

    (
        pend_tip_world,
        pend_begin_world,
        # rot_matrix,
    ) = _get_pend_tip_beg_world(mj_data=mj_data)

    rot_matrix_pin = _get_rot_matrix_to_local_frame(pin_model, pin_data, q, qdot)

    diff_vector = pend_tip_world - pend_begin_world
    print("diff vector:", diff_vector)
    diff_vector_normalized = diff_vector / np.linalg.norm(diff_vector)
    # use the pinocchio version
    diff_vector_local_frame = rot_matrix_pin.T @ diff_vector_normalized
    print("local frame: ", rot_matrix_pin.T @ diff_vector)
    # diff_vector_local_frame = rot_matrix @ diff_vector
    phi_prime, theta_prime = compute_joint_angles(input_vector=diff_vector_local_frame)
    q_prime = copy(q)
    q_prime[-2:] = np.array([phi_prime, theta_prime])
    print(q_prime)
    print(q)
    assert (
        np.isclose(q_prime - q, 0, rtol=1e-5, atol=1e-5).all() == True
    ), "Is not close."


if __name__ == "__main__":
    unittest.main()
    # input_vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example input vectors
    # fitted_vector = perform_linear_regression(input_vectors)
    # print(f"Fitted vector: {fitted_vector}")
    # print(input_vectors @ fitted_vector)
    # test_single_config()
