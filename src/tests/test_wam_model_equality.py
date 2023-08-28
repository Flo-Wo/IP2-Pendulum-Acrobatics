import unittest
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


class TestModelEquality(unittest.TestCase):
    def setUp(self) -> None:
        # We load both models
        model_path = Path(__file__).parent.parent / "src" / "wam"
        # self.wam1 = mujoco.MjModel.from_xml_path(str(model_path / "rot_wam.xml"))
        self.wam1 = mujoco.MjModel.from_xml_path(str(model_path / "wam4.xml"))
        self.wam1_data = mujoco.MjData(self.wam1)
        # self.wam2 = mujoco.MjModel.from_xml_path(str(model_path / "wam4_original_inertia_frames.xml"))
        # self.wam2_data = mujoco.MjData(self.wam2)

        self.pin_model, __, __ = pin.buildModelsFromUrdf(
            # str(model_path / "rot_wam.urdf"), str(model_path / "meshes")
            str(model_path / "wam4.urdf"),
            str(model_path / "meshes"),
        )

    @staticmethod
    def mj_fwd_dyn(model, data, q, dq, tau):
        data.qpos[:] = q
        data.qvel[:] = dq
        data.ctrl[:] = tau
        mujoco.mj_forward(model, data)
        ddq = data.qacc
        return ddq

    @staticmethod
    def mj_inv_dyn(model, data, q, dq, ddq):
        data.qpos[:] = q
        data.qvel[:] = dq
        data.qacc[:] = ddq
        mujoco.mj_inverse(model, data)
        tau = data.qfrc_inverse
        return tau

    @staticmethod
    def pin_fwd_dyn(model, q, dq, tau):
        data = model.createData()
        pin.computeAllTerms(model, data, q, dq)
        # Pinocchio does not model damping in its algorithms so we need to do this ourselves
        tau = tau - model.damping * dq
        b = pin.rnea(model, data, q, dq, np.zeros(4))
        ddq = np.linalg.solve(data.M, tau - b)
        return ddq

    @staticmethod
    def pin_inv_dyn(model, q, dq, ddq):
        data = model.createData()
        tau = pin.rnea(model, data, q, dq, ddq) + model.damping * dq
        return tau

    def testModels(self):
        np.random.seed(0)
        joint_positions = np.random.uniform(
            self.wam1.jnt_range[:, 0], self.wam1.jnt_range[:, 1], size=(10000, 4)
        )
        joint_velocities = np.random.uniform(-1, 1, size=(10000, 4))

        mj_ddqs = np.zeros_like(joint_positions)
        mj_taus = np.zeros_like(joint_positions)
        pin_ddqs = np.zeros_like(joint_positions)
        pin_taus = np.zeros_like(joint_velocities)
        for n, (q, dq) in enumerate(zip(joint_positions, joint_velocities)):
            # random state close to neutral to avoid errors from collisions
            # random test inputs
            tau_test = np.random.uniform(-20, 20, (4,))
            ddq_test = np.random.uniform(-5, 5, (4,))

            # compare forward dynamics
            mj_ddq1 = self.mj_fwd_dyn(self.wam1, self.wam1_data, q, dq, tau_test)
            # mj_ddq2 = self.mj_fwd_dyn(self.wam2, self.wam2_data, q, dq, tau_test)
            pin_ddq = self.pin_fwd_dyn(self.pin_model, q, dq, tau_test)
            # self.assertTrue(
            #     np.all(
            #         np.abs(mj_ddq1 - mj_ddq2)
            #         / np.maximum(1.0, 0.5 * (np.abs(mj_ddq1) + np.abs(mj_ddq2)))
            #         <= 1e-3
            #     )
            # )
            pin_ddqs[n] = pin_ddq
            # mj_ddqs[n] = 0.5 * (mj_ddq1 + mj_ddq2)
            mj_ddqs[n] = mj_ddq1

            # compare inverse dynamics
            mj_tau1 = self.mj_inv_dyn(self.wam1, self.wam1_data, q, dq, ddq_test)
            # mj_tau2 = self.mj_inv_dyn(self.wam1, self.wam1_data, q, dq, ddq_test)
            pin_tau = self.pin_inv_dyn(self.pin_model, q, dq, ddq_test)
            # self.assertTrue(
            #     np.all(
            #         np.abs(mj_tau1 - mj_tau2)
            #         / np.maximum(1.0, 0.5 * (np.abs(mj_tau1) + np.abs(mj_tau2)))
            #         <= 1e-3
            #     )
            # )
            pin_taus[n] = pin_tau
            # mj_taus[n] = 0.5 * (mj_tau1 + mj_tau2)
            mj_taus[n] = mj_tau1

        rel_pin_ddq_errors = np.mean(np.abs(pin_ddqs - mj_ddqs)) / np.maximum(
            1.0, np.abs(mj_ddqs)
        )
        rel_pin_tau_errors = np.mean(np.abs(pin_taus - mj_taus)) / np.maximum(
            1.0, np.abs(mj_taus)
        )

        print("rel tau errors: ", np.percentile(rel_pin_tau_errors, 95))
        print("rel ddq errors: ", np.percentile(rel_pin_ddq_errors, 95))
        self.assertTrue(np.percentile(rel_pin_tau_errors, 95) <= 1e-5)
        self.assertTrue(np.percentile(rel_pin_ddq_errors, 95) <= 1e-3)


if __name__ == "__main__":
    unittest.main()
