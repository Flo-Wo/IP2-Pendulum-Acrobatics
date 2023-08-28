from typing import List

import numpy as np

from utils.pin_helper.world_frame import frame_translation_global


def angles_traj(
    pin_model,
    pin_data,
    q_list: List[np.array],
    upper_frame_id: int,
    lower_frame_id: int,
    reference_vector: np.array = np.array([0, 0, 1]),
    q_dot: List[np.array] = None,
    radian: bool = False,
    project_2D: bool = False,
):
    angles = np.zeros(len(q_list))
    if q_dot is None:
        q_dot = [None] * len(q_list)
    for idx, q in enumerate(q_list):
        angles[idx] = angle_frames_to_vec(
            pin_model,
            pin_data,
            q,
            upper_frame_id,
            lower_frame_id,
            reference_vector,
            q_dot[idx],
            radian=radian,
            project_2D=project_2D,
        )
    return angles


def angle_frames_to_vec(
    pin_model,
    pin_data,
    q: np.array,
    upper_frame_id: int,
    lower_frame_id: int,
    reference_vector: np.array = np.array([0, 0, 1]),
    q_dot: np.array = None,
    radian: bool = False,
    project_2D: bool = False,
):
    """Angle between the connection vector of two frames and a reference vector.

    Parameters
    ----------
    pin_model : pin.Model
        Pinocchio Model
    pin_data : pin.Data
        Pinocchio data to the model above.
    q : np.array
        Joint configuration
    frame1_id : int
        Id of the upper frame
    frame2_id : int
        Id of the lower frame
    reference_vector : np.array, optional
        Vector to compute angle between, by default np.array([0, 0, 1])
        i.e. the z-axis.
    q_dot : np.array, optional
        Joint velocities, by default None.
    radian : bool, optional
        Angles are by default in degrees, by default False i.e. in degrees.

    Returns
    -------
    float
        Angle in degrees (default) or radian (if radian=True is set).
    """
    frame1_glob = frame_translation_global(
        pin_model, pin_data, q, upper_frame_id, q_dot=q_dot
    )

    frame2_glob = frame_translation_global(
        pin_model, pin_data, q, lower_frame_id, q_dot=q_dot
    )
    return compute_angle(
        frame1_glob - frame2_glob,
        reference_vector,
        radian=radian,
        project_2D=project_2D,
    )


def compute_angle(
    vec1: np.array,
    vec2: np.array,
    radian: bool = False,
    project_2D: bool = False,
):
    signum = 1
    if project_2D:
        vec1 = np.take(vec1, indices=(0, 2), axis=0)
        signum = np.sign(np.take(vec1, indices=0, axis=0))
        vec2 = np.take(vec2, indices=(0, 2), axis=0)
    vec1_norm = normalize_vec(vec1)
    vec2_norm = normalize_vec(vec2)

    angle = signum * np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
    if radian:
        return angle
    return angle * 180 / np.pi


def normalize_vec(vec: np.array) -> np.array:
    return vec / np.linalg.norm(vec)
