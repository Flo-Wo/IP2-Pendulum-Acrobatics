import mujoco

from utils.visualize.vis_mujoco.mj_viewer_base import MujocoViewer


def make_viewer(model, data, cam_type: str = "standard"):
    viewer = MujocoViewer(model, data, hide_menu=False)
    viewer.cam = make_camera(cam_type=cam_type)
    return viewer


def make_camera(cam_type: str = "standard"):
    cam = mujoco.MjvCamera()
    if cam_type == "standard":
        cam.azimuth = -60
        cam.elevation = -20
        cam.distance = 4
        cam.lookat[:] = [0.2, 0.2, 2.4]
    elif cam_type == "proj_xy":
        cam.azimuth = 180
        cam.elevation = -90
        cam.distance = 0.7
        cam.lookat[:] = [-0.2, 0.1, 3.5]
    elif cam_type == "proj_yz":
        cam.azimuth = -90
        cam.elevation = -0
        cam.distance = 2.5
        cam.lookat[:] = [-0.3, 0.14, 2.7]
    elif cam_type == "proj_yz_close_up":
        cam.azimuth = -90
        cam.elevation = -0
        cam.distance = 1
        cam.lookat[:] = [-0.3, 0.14, 3.2]
    elif cam_type == "lab_setting":
        cam.azimuth = -150
        cam.elevation = -10
        cam.distance = 2.5
        cam.lookat[:] = [-0.3, 0.14, 2.7]
    else:
        raise NotImplementedError(f"Camera type {cam_type} not implemented.")
    return cam
