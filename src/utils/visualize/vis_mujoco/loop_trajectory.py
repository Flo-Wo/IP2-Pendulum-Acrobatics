from copy import copy, deepcopy

import mujoco
import numpy as np

from utils.config import ConfigModelBase
from utils.visualize.vis_mujoco.camera import make_viewer
from utils.visualize.vis_mujoco.mj_viewer_base import MujocoViewer


def loop_mj_vis(
    model: ConfigModelBase,
    controls: list = None,
    q: np.array = None,
    q_dot: np.array = None,
    num_motors: int = 4,
    cam_type: str = "standard",
):

    mj_model = deepcopy(model.mj_model)
    mj_data = deepcopy(model.mj_data)

    if controls is not None:
        n = len(controls)
    else:
        assert (
            np.shape(q)[0] == np.shape(q_dot)[0]
        ), "Lengths of q and q_dot must be equal."
        n = np.shape(q)[0]
    # print("controls: ")
    # print(controls)

    def run_step(controls, q, q_dot):
        if controls is not None:

            def one_step(controls, q, q_dot, step):
                mj_data.ctrl[:num_motors] = controls[step, :num_motors]

        elif q is not None and q_dot is not None:

            def one_step(controls, q, q_dot, step):
                mj_data.qpos[:] = q[step, :]
                mj_data.qvel[:] = q_dot[step, :]

        else:
            raise NotImplementedError("Give either list of controls or (q AND q_dot)")

        return one_step

    viewer = make_viewer(mj_model, mj_data, cam_type=cam_type)
    step = 0
    # reset the model and execute the computed torques
    mujoco.mj_resetData(mj_model, mj_data)

    mj_data.qpos[:] = model.q_config
    mj_data.qvel[:] = np.zeros(np.shape(mj_data.qvel))

    one_step = run_step(controls, q, q_dot)

    # mujoco.mj_step(mj_model, mj_data)
    while step < n:
        one_step(controls, q, q_dot, step)
        # mj_data.ctrl[:num_motors] = controls[step, :num_motors]
        mujoco.mj_step(mj_model, mj_data)
        viewer.render()
        step += 1
        if step % n == 0:
            mujoco.mj_resetData(mj_model, mj_data)
            mj_data.qpos[:] = model.q_config
            mj_data.qvel[:] = np.zeros(np.shape(mj_data.qvel))
            # mujoco.mj_step(mj_model, mj_data)
            step = 0
