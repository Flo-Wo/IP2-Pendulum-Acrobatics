import pinocchio as pin
import numpy as np
from typing import List
from copy import copy


def frame_translation_global(
    pin_model, pin_data, q: np.array, frame_idx: int, q_dot: np.array = None
) -> np.array:
    all_frames = all_frames_pos_global(pin_model, pin_data, q, q_dot=q_dot)
    return all_frames[frame_idx].translation


def task_space_traj(
    pin_model, pin_data, q_list: List[np.array], frame_idx: int
) -> List[np.array]:
    x_traj = np.zeros((np.shape(q_list)[0], 3))
    for idx, q in enumerate(q_list):
        x_traj[idx, :] = frame_translation_global(
            pin_model, pin_data, q, frame_idx=frame_idx
        )
    return x_traj


def all_frames_pos_global(pin_model, pin_data, q, q_dot=None) -> List[np.array]:
    """Execute one forward kinematics pass and get frame positions
    w.r.t. global frame.

    Parameters
    ----------
    pin_model : pin.ModelTpl
        Pinocchio model.
    pin_data : pin.DataTpl
        Pinocchio data corresponding to the model above.
    q : np.array
        Joint configuration
    q_dot : np.arrray, optional
        Joint velocities, by default None and thus set
        to zero vector

    Returns
    -------
    List[np.array]
        Return pin_data.oMf, can be used to compute
        .translation or .rotation
    """

    if q_dot is None:
        q_dot = np.zeros(pin_model.nv)
    pin.forwardKinematics(pin_model, pin_data, q, q_dot)
    pin.updateFramePlacements(pin_model, pin_data)
    return copy(pin_data.oMf)
