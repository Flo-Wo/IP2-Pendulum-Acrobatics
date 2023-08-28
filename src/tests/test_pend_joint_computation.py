import unittest
from copy import copy

import mujoco
import numpy as np
import pinocchio as pin

from utils.config.config_ip import ConfigWAM
from utils.pin_helper import compute_endeffector_position, compute_joint_angles


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

    def not_test_joint_angles_to_endeff(self):
        for phi in self.angles:
            for theta in self.angles:
                endeff = compute_endeffector_position(
                    pendulum_len=self.pendulum_len, joint_angles=(phi, theta)
                )
                phi_prime, theta_prime = compute_joint_angles(input_vector=endeff)
                endeff_prime = compute_endeffector_position(
                    pendulum_len=self.pendulum_len,
                    joint_angles=(phi_prime, theta_prime),
                )
                self._tuples_are_equal(input=endeff, input_prime=endeff_prime)
                self._tuples_are_equal(
                    input=np.array([phi, theta]),
                    input_prime=np.array([phi_prime, theta_prime]),
                )

    def test_joint_angles_in_mujoco(self):
        model = ConfigWAM("rotated", pendulum_len=0.174)
        pin_model = model.pin_model
        pin_data = model.pin_data
        mj_data = model.mj_data
        mj_model = model.mj_model

        def _get_pend_tip_beg_world(mj_data):
            rot_matrix = np.array(
                [
                    mj_data.sensordata[9:12].copy(),
                    mj_data.sensordata[12:15].copy(),
                    mj_data.sensordata[15:18].copy(),
                ]
            )
            return (
                mj_data.sensordata[0:3].copy(),
                mj_data.sensordata[6:9].copy(),
                rot_matrix,
            )

        def _get_rot_matrix_to_local_frame(qpos, qdot) -> np.ndarray:
            pin.forwardKinematics(pin_model, pin_data, qpos, qdot)
            pin.updateFramePlacements(pin_model, pin_data)
            ref_frame_id = pin_model.getFrameId("links/pendulum/base")
            rot_matrix = pin_data.oMf[ref_frame_id].rotation
            return rot_matrix

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
                    rot_matrix,
                ) = _get_pend_tip_beg_world(mj_data=mj_data)

                rot_matrix_pin = _get_rot_matrix_to_local_frame(q, qdot)

                diff_vector = pend_tip_world - pend_begin_world
                diff_vector /= np.linalg.norm(diff_vector)
                # use the pinocchio version
                diff_vector_local_frame = rot_matrix_pin.T @ diff_vector
                # diff_vector_local_frame = rot_matrix @ diff_vector
                phi_prime, theta_prime = compute_joint_angles(
                    input_vector=diff_vector_local_frame
                )
                q_prime = copy(q)
                q_prime[-2:] = np.array([phi_prime, theta_prime])
                self._tuples_are_equal(q_prime[-2:], q[-2:])

    def _tuples_are_equal(self, input: tuple, input_prime: tuple):
        self.assertEqual(len(input), len(input_prime))
        for idx in range(len(input)):
            self.assertAlmostEqual(input[idx], input_prime[idx], places=5)


if __name__ == "__main__":
    unittest.main()
