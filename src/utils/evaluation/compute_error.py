import numpy as np


def _task_space_error(x_traj: np.array, goal_traj: np.array) -> np.array:
    return x_traj - goal_traj


def _rot_error(pen_top: np.array, pen_beg: np.array, radian: bool = False) -> np.array:
    diff = pen_top - pen_beg
    z_axis = np.array([0, 0, 1])
    angles = np.arccos(
        diff @ z_axis / (np.linalg.norm(z_axis) * np.linalg.norm(diff, axis=1))
    )
    if radian:
        return angles
    # convert to degrees
    return angles * 180 / np.pi


# TODO: check axis
def _l2_error_along_axis(diff: np.array, ord=2, axis=1, use: str = "norm"):
    if use == "norm":
        avg_l2 = np.average(np.linalg.norm(diff, axis=axis, ord=ord))
    elif use == "abs":
        avg_l2 = np.average(np.abs(diff))
    else:
        raise NotImplementedError
    return avg_l2
