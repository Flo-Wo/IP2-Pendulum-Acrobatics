import numpy as np


def circle(
        radius: float,
        rest_point: float,
        frequency: float = 1,
        n_steps: int = 2000,
        start_pos: str = "bottom",
        rot_dir: int = 1,
        plane: str = "y",
        smoothing: bool = False
):
    if smoothing:
        n_steps += 500

    domain = rot_dir * frequency * np.linspace(0, 2 * np.pi, num=n_steps)

    # position is always relative to the position of the circle's center in the
    # projective plane (project into the two planes exlcuding the given plane)
    offset_map = {"right": 0, "top": 0.5, "left": 1, "bottom": 3 / 2}
    offset = offset_map[start_pos] * np.pi

    # start = rest_point - np.array([radius * np.cos(offset), 0, radius * np.sin(offset)])

    cos_move = radius * np.cos(domain + offset)[None, :]
    sin_move = radius * np.sin(domain + offset)[None, :]
    zeros = np.zeros(n_steps)[None, :]

    if plane == "x":
        rotation = stack_vectors(zeros, cos_move, sin_move)
    elif plane == "y":
        rotation = stack_vectors(cos_move, zeros, sin_move)
    elif plane == "z":
        rotation = stack_vectors(cos_move, sin_move, zeros)

    if smoothing:
        end_smooth = 500
        smooth_factor = np.linspace(0, 1, end_smooth, endpoint=True)
        rotation[:, :end_smooth] *= smooth_factor

    start = rest_point - rotation[:, 0]

    traj_task_space = (start[:, None] + rotation).T
    return traj_task_space


def spiral(
        radius: float,
        max_height: float,
        rest_point: float,
        circle_frequency: float = 1,
        n_steps: int = 2000,
        start_pos: str = "bottom",
        circle_orientation: int = 1,
        spiral_orientation: int = 1,
        plane: str = "y",
        smoothing: bool = False
):
    if smoothing:
        n_steps += 500

    circle_domain = (
            circle_orientation * circle_frequency * np.linspace(0, 2 * np.pi, num=n_steps)
    )

    spiral_domain_up = (
            spiral_orientation * max_height * np.linspace(0, 1, num=int(n_steps / 2))
    )
    spiral_domain_down = (
            spiral_orientation
            * max_height
            * np.linspace(1, 0, num=n_steps - int(n_steps / 2))
    )
    # position is always relative to the position of the circle's center in the
    # projective plane (project into the two planes exlcuding the given plane)
    offset_map = {"right": 0, "top": 0.5, "left": 1, "bottom": 3 / 2}
    offset = offset_map[start_pos] * np.pi

    # start = rest_point - np.array([radius * np.cos(offset), 0, radius * np.sin(offset)])

    # cos_move = radius * np.cos(circle_domain + offset)[None, :]
    cos_move = radius * np.cos(circle_domain + offset)[None, :]
    sin_move = radius * np.sin(circle_domain + offset)[None, :]
    height_move = np.concatenate((spiral_domain_up, spiral_domain_down))[None, :]

    if plane == "x":
        rotation = stack_vectors(height_move, cos_move, sin_move)
    elif plane == "y":
        rotation = stack_vectors(cos_move, height_move, sin_move)
    elif plane == "z":
        rotation = stack_vectors(cos_move, sin_move, height_move)

    if smoothing:
        end_smooth = 500
        smooth_factor = np.linspace(0, 1, end_smooth, endpoint=True)
        rotation[:, :end_smooth] *= smooth_factor

    start = rest_point - rotation[:, 0]

    traj_task_space = (start[:, None] + rotation).T
    return traj_task_space


def figure_eight(
        radius: float,
        rest_point: float,
        frequency: float = 1,
        n_steps: int = 2000,
        rot_dir: int = 1,
):
    # parametrization of a lissajous-curve, see:
    # https://en.wikipedia.org/wiki/Lissajous_curve
    # the parametrization (i.e. the phase shift) is set in such a way, that
    # the resting position is always the center of the lissajous-curve
    domain = rot_dir * frequency * np.linspace(np.pi / 2, 2.5 * np.pi, num=n_steps)

    start = rest_point
    rotation = radius * np.vstack(
        (
            np.sin(domain + np.pi / 2)[None, :],
            np.zeros(n_steps)[None, :],
            np.sin(2 * domain)[None, :],
        )
    )
    traj_task_space = (start[:, None] + rotation).T
    return traj_task_space


def stack_vectors(x: np.array, y: np.array, z: np.array):
    return np.vstack((x, y, z))
