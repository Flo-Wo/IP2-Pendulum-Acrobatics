import math
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import interp1d

from utils.config import ConfigWAM
from utils.visualize.plots.projection_plots import plot_2D_projected_trajectory
from utils.visualize.plots.torques import plot_torques


# helper functions
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


def compute_pend_pos_world(
    pin_model, pin_data, q_wam: np.ndarray, pend_vector: np.ndarray
) -> np.ndarray:
    q_pend = compute_pend_q(pin_model, pin_data, q_wam, pend_vector)
    q_full = np.concatenate((q_wam, q_pend))

    pin.forwardKinematics(pin_model, pin_data, q_full)
    pin.updateFramePlacements(pin_model, pin_data)

    pole_tip_id = pin_model.getFrameId("links/pendulum/pole_tip")
    return pin_data.oMf[pole_tip_id].translation


def compute_pend_q(
    pin_model, pin_data, q_wam: np.ndarray, pend_vector: np.ndarray
) -> np.ndarray:
    rot_matrix = _rot_matrix_to_local_frame(pin_model, pin_data, q_wam=q_wam)
    pend_vector_local_frame = rot_matrix.T @ pend_vector
    return compute_joint_angles(pend_vector_local_frame)


def _rot_matrix_to_local_frame(pin_model, pin_data, q_wam: np.ndarray) -> np.ndarray:
    """
    Reference frame in the world frame -> multiply with transpose to
    transform the pendulum vector from world-frame to reference frame.
    """
    pin.forwardKinematics(
        pin_model,
        pin_data,
        np.concatenate((q_wam, np.array([0, 0]))),
    )
    pin.updateFramePlacements(pin_model, pin_data)
    ref_frame_id = pin_model.getFrameId("links/pendulum/base")
    rot_matrix = pin_data.oMf[ref_frame_id].rotation
    return rot_matrix


def eval_tracking(save: bool = True):
    model = ConfigWAM(model_name="rotated", pendulum_len=0.374)

    ref_pole_tip_pos = model.get_pend_end_world().copy()

    pin_model = model.pin_model
    pin_data = model.pin_data

    curr_folder = Path(__file__).resolve().parent / "tracking"

    target_trajectory_input = np.load(curr_folder / "eight_trajectories.npz")["pos"][
        :, 0, :
    ]
    target_trajectory = np.tile(target_trajectory_input, (4, 1))

    achieved_data_path = curr_folder / "achieved_trajectory_looped.npz"
    observed_data_path = curr_folder / "observed_trajectory_looped.npz"

    achieved_data = np.load(achieved_data_path)
    observed_data = np.load(observed_data_path)

    print(achieved_data["torques"].shape)

    assert achieved_data["pos"].shape[0] == observed_data["pos"].shape[0]
    pole_pos = []
    achieved_pos = achieved_data["pos"]
    achieved_ts = achieved_data["ts"]

    achieved_torques = achieved_data["torques"]

    observed_pos = observed_data["pos"]

    target_ts = 0.008 * np.arange(target_trajectory.shape[0])

    for ts in range(achieved_pos.shape[0]):
        pole_pos.append(
            compute_pend_pos_world(
                pin_model, pin_data, achieved_pos[ts, :4], achieved_pos[ts, 4:]
            ).copy()
        )

    acp_fn = interp1d(
        achieved_ts,
        pole_pos,
        bounds_error=False,
        fill_value="extrapolate",
        axis=0,
    )
    mask = target_ts <= achieved_ts[-1]
    achieved_cartesian_pos = acp_fn(target_ts[mask])

    print(
        "Average Tracking Error: {:.4e}".format(
            np.mean(
                np.linalg.norm(
                    achieved_cartesian_pos - target_trajectory[mask, :], axis=-1
                )
            )
        )
    )
    plot_2D_projected_trajectory(
        actual_pole_tip_positions=achieved_cartesian_pos,
        ref_pole_tip_pos=ref_pole_tip_pos,
        target_pole_tip_positions=target_trajectory,
        time_step=0.008,
        save=save,
        filename="Real_LQR_tracking.pdf",
        path="../../report/imgs/LQR_real_system/",
    )
    planned_torques = np.load(curr_folder / "solver_results.npy")
    planned_torques = planned_torques[:8100, :]
    print(achieved_ts[-1])

    plot_torques(
        torques=achieved_torques,
        planned_torques=planned_torques,
        x_axis=achieved_ts,
        filename="Real_LQR_torques.pdf",
        save=save,
        torque_duration=0.008,
        path="../../report/imgs/LQR_real_system/",
    )


def eval_stabilization(save: bool = True):

    model = ConfigWAM(model_name="rotated", pendulum_len=0.374)

    ref_pole_tip_pos = model.get_pend_end_world().copy()
    pin_model = model.pin_model
    pin_data = model.pin_data

    curr_folder = Path(__file__).resolve().parent / "stabilization"
    achieved_data_path = curr_folder / "achieved_trajectory_vid1.npz"
    observed_data_path = curr_folder / "observed_trajectory_vid1.npz"

    achieved_data = np.load(achieved_data_path)
    observed_data = np.load(observed_data_path)

    assert achieved_data["pos"].shape[0] == observed_data["pos"].shape[0]
    pole_pos = []
    achieved_pos = achieved_data["pos"]
    observed_pos = observed_data["pos"]
    for ts in range(achieved_pos.shape[0]):
        pole_pos.append(
            compute_pend_pos_world(
                pin_model, pin_data, achieved_pos[ts, :4], achieved_pos[ts, 4:]
            ).copy()
        )
    pole_pos_arr = np.array(pole_pos)

    pole_offset = pole_pos_arr - ref_pole_tip_pos

    target = np.zeros_like(pole_offset)
    print(
        f"Average Tracking Error: {np.mean(np.linalg.norm(pole_offset - target, axis=-1)): .4e}"
    )

    plot_2D_projected_trajectory(
        actual_pole_tip_positions=pole_pos_arr,
        ref_pole_tip_pos=ref_pole_tip_pos,
        # planned_pole_tip_positions=pole_positions,
        time_step=0.008,
        save=save,
        filename="Real_LQR_stabilization.pdf",
        path="../../report/imgs/LQR_real_system/",
    )


if __name__ == "__main__":
    eval_tracking(save=False)
    # eval_stabilization(save=False)
