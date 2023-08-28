import logging
from copy import copy

import mujoco
import numpy as np

from utils.config import ConfigModelBase
from utils.visualize.video import frames_to_video
from utils.visualize.vis_mujoco import make_camera, make_viewer

# original source: https://github.com/kploeger/python_sandbox/blob/master/simulation/mujoco_render_trace.py


def get_site_speed(model, data, site_name):
    """Returns the speed of a geom."""
    site_vel = np.zeros(6)
    site_type = mujoco.mjtObj.mjOBJ_SITE
    site_id = data.site(site_name).id
    mujoco.mj_objectVelocity(model, data, site_type, site_id, site_vel, 0)
    return np.linalg.norm(site_vel)


def modify_scene(scn, positions, speeds):
    """Draw position trace, speed modifies width and colors."""
    if len(positions) > 1:
        for i in range(len(positions) - 1):
            rgba = np.array(
                (
                    np.clip(speeds[i] / 10, 0, 1),
                    np.clip(1 - speeds[i] / 10, 0, 1),
                    0.5,
                    1.0,
                )
            )
            radius = 0.005 / (1 + np.sqrt(speeds[i]))
            if scn.ngeom < scn.maxgeom:
                scn.ngeom += 1  # increment ngeom
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(9),
                    rgba.astype(np.float32),
                )
                mujoco.mjv_makeConnector(
                    scn.geoms[scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    radius,
                    positions[i][0],
                    positions[i][1],
                    positions[i][2],
                    positions[i + 1][0],
                    positions[i + 1][1],
                    positions[i + 1][2],
                )


def visualize_tip_position(
    model: ConfigModelBase,
    filename: str,
    path: str = "../videos/",
    controls: list = None,
    q: np.array = None,
    q_dot: np.array = None,
    num_motors: int = 4,
    cam=None,
    save_last_frame: bool = True,
    mj_timestep: float = 0.002,
    frame_path: str = "../videos/last_frame/",
    show_logs: bool = True,
    cam_type: str = "standard",
):
    mj_model = copy(model.mj_model)
    mj_data = copy(model.mj_data)
    pin_model = copy(model.pin_model)

    if controls is not None:
        n = len(controls)
        print("len = ", n)
    else:
        assert (
            np.shape(q)[0] == np.shape(q_dot)[0]
        ), "Lengths of q and q_dot must be equal."
        n = np.shape(q)[0]

    # compute the upper bound for the simulation and the video
    duration = n * mj_timestep

    def run_step(controls, q, q_dot):
        if controls is not None:

            def one_step(controls, q, q_dot, step):
                mj_data.ctrl[:num_motors] = controls[step][:num_motors]

        elif q is not None and q_dot is not None:

            def one_step(controls, q, q_dot, step):
                mj_data.qpos[:] = q[step, :]
                mj_data.qvel[:] = q_dot[step, :]

        else:
            raise NotImplementedError("Give either list of controls or (q AND q_dot)")

        return one_step

    tool_x_hist = []
    tool_dx_hist = []

    cam = make_camera(cam_type)
    viewer = make_viewer(mj_model, mj_data, cam_type=cam_type)
    viewer_cb = lambda m, d, s: modify_scene(s, tool_x_hist, tool_dx_hist)
    viewer.set_pre_render_callback(viewer_cb)

    mujoco.mj_resetData(mj_model, mj_data)  # Reset state and time.

    mj_data.qpos[:] = model.q_config
    mj_data.qvel[:] = np.zeros(pin_model.nv)

    one_step = run_step(controls, q, q_dot)

    # mujoco.mj_forward(mj_model, mj_data)
    # mujoco.mj_step(mj_model, mj_data)

    # IP1 config
    # site_name = "wam/sensor_sites/pend_endeff"
    site_name = "pendulum/sensor_sites/pole_tip"

    step = 0
    while step < n:
        one_step(controls, q, q_dot, step)
        mujoco.mj_step(mj_model, mj_data)
        tool_x_hist.append(copy(mj_data.site_xpos[mj_data.site(site_name).id]))
        # tool_x_hist.append(mj_data.site_xpos[-1].copy())
        tool_dx_hist.append(copy(get_site_speed(mj_model, mj_data, site_name)))
        viewer.render()
        step += 1
    viewer.close()
    viewer = None

    # reset again
    mujoco.mj_resetData(mj_model, mj_data)  # Reset state and time.
    mj_data.qpos[:] = model.q_config
    mj_data.qvel[:] = np.zeros(pin_model.nv)

    # mujoco.mj_forward(mj_model, mj_data)
    # mujoco.mj_step(mj_model, mj_data)
    resolution = [2560, 1440]

    mj_model.vis.global_.offwidth = resolution[0]
    mj_model.vis.global_.offheight = resolution[1]
    renderer = mujoco.Renderer(mj_model, width=resolution[0], height=resolution[1])

    t0 = mj_data.time
    frames = []
    tool_x_hist = []
    tool_dx_hist = []

    framerate = 60  # (Hz)

    step = 0
    if show_logs:
        logging.info("Working on the saving")
    while step < n:
        one_step(controls, q, q_dot, step)
        mujoco.mj_step(mj_model, mj_data)
        tool_x_hist.append(mj_data.site_xpos[mj_data.site(site_name).id].copy())
        tool_dx_hist.append(get_site_speed(mj_model, mj_data, site_name))
        if len(frames) < (mj_data.time - t0) * framerate:
            renderer.update_scene(mj_data, camera=cam)
            modify_scene(renderer.scene, tool_x_hist, tool_dx_hist)  # draw things!
            frames.append(renderer.render().copy())
        if step % 100 == 0:
            logging.info("Frame: {}/{}".format(step, n))
        step += 1
    frames_to_video(
        frames,
        filename=filename,
        path=path,
        save_last_frame=save_last_frame,
        frame_path=frame_path,
    )
