import logging
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np
from pinocchio import RobotWrapper

from utils.config.prepro_calculations import cylinder_inertia_tensor


def read_model(
    model_type: str,
    file_end: str = ".urdf",
    pendulum_len: float = 0.3,
    pendulum_radius: float = 0.005,
    pendulum_mass: float = 0.5,
):
    """Read the model given the model type, the file ending and the pendulum length.
    This function will read the metafile, substitute the length of the pendulum and
    then read a temporary file with the set pendulum length.

    Parameters
    ----------
    model_type : str
        Type of the model, currently either "rot" or "normal" or "without_pendulum".
    file_end : str, optional
        File ending, currently either ".urdf" (for pinocchio)
        or ".xml" (for mujoco), by default ".urdf".
    pendulum_len : float, optional
        Length of the pendulum, by default 0.3.
    pendulum_radius : float, optional
        Radius of the pendulum, by default 0.015.
    pendulum_mass : float, optional
        Mass of the pendulum, by default 0.5.

    Returns
    -------
    Mujoco/Pinocchio Model
        Robot model, either for mujoco or pinocchio.

    Raises
    ------
    NotImplementedError
        If you enter the wrong model type.
    NotImplementedError
        If you enter an incorrect file ending.
    """
    if model_type == "rot":
        logging.info("READ ROTATED")
        filename = "rot_wam_pend"
    elif model_type == "normal":
        logging.info("READ NORMAL")
        filename = "wam_pend"
    elif model_type == "without_pendulum":
        logging.info("READ WAM WITHOUT PENDULUM.")
        raise NotImplementedError
        # filename = "rot_wam"
    else:
        raise NotImplementedError

    tmp_end = _set_len_radius_inertia(
        filename,
        file_end,
        pendulum_len,
        pendulum_radius=pendulum_radius,
        pendulum_mass=pendulum_mass,
    )

    if file_end == ".urdf":
        file_reader = lambda fn: RobotWrapper.BuildFromURDF(
            filename=fn, package_dirs=str(Path(fn).parent / "meshes")
        )
    elif file_end == ".xml":
        file_reader = mujoco.MjModel.from_xml_path
    else:
        raise NotImplementedError

    return _read_raw(
        filename=filename, tmp_end=tmp_end, file_end=file_end, reader=file_reader
    )


def _set_len_radius_inertia(
    filename: str,
    file_end: str,
    pendulum_len: float,
    pendulum_radius: float = 0.005,
    pendulum_mass: float = 0.5,
) -> str:
    filedata = _read_file(filename, file_end)
    replaced_data = _insert_inertia_pend_len_optitrack(
        filedata,
        pendulum_len,
        pendulum_radius=pendulum_radius,
        pendulum_mass=pendulum_mass,
    )
    tmp_end = "_{}_temp".format(pendulum_len)
    return _save_file(filename, tmp_end, file_end, replaced_data)


def _read_raw(
    filename: str,
    tmp_end: str,
    file_end: str,
    reader: callable,
):
    full_path = str(
        Path(__file__).parent.parent.parent / "wam" / (filename + tmp_end + file_end)
    )
    return reader(full_path)


def _read_file(
    filename: str,
    file_end: str,
) -> str:
    full_path = str(
        Path(__file__).parent.parent.parent / "wam/metafiles" / (filename + file_end)
    )
    with open(full_path, "r") as file:
        filedata = file.read()
    return filedata


def _save_file(
    filename: str,
    tmp_end: str,
    file_end: str,
    replaced_data: str,
):
    full_path = str(
        Path(__file__).parent.parent.parent / "wam" / (filename + tmp_end + file_end)
    )
    with open(full_path, "w") as file:
        file.write(replaced_data)
    return tmp_end


class InertiaTensorFormat:
    def __init__(self, inertia_tensor: np.ndarray):
        self.x = "{:10.16f}".format(inertia_tensor[0])
        self.y = "{:10.16f}".format(inertia_tensor[1])
        self.z = "{:10.16f}".format(inertia_tensor[2])

    def tensor_to_str(self) -> str:
        return " ".join([self.x, self.y, self.z])

    def __repr__(self) -> str:
        return self.tensor_to_str()


def _insert_inertia_pend_len_optitrack(
    model_descr: str,
    pendulum_len: float = 0.3,
    pendulum_radius: float = 0.015,
    pendulum_mass: float = 0.5,
):
    inertia_tensor = InertiaTensorFormat(
        cylinder_inertia_tensor(
            mass=pendulum_mass,
            height=pendulum_len,
            radius=pendulum_radius,
        )
    )

    optitrack_markers = _optitrack_marker_pos(pendulum_len=pendulum_len)

    # PENDULUM SETTINGS
    logging.info("Pendulum length: {}".format(pendulum_len))
    logging.info("Pendulum radius: {}".format(pendulum_radius))
    logging.info("Pendulum mass: {}".format(pendulum_mass))

    # INERTIA SETTINGS
    logging.info("Inertia position: {}".format(pendulum_len / 2))
    logging.info("Inertia tensor: {}".format(inertia_tensor))

    # optitrack settings
    logging.info("Optitrack One: {}".format(optitrack_markers["one"]))
    logging.info("Optitrack Two: {}".format(optitrack_markers["two"]))
    logging.info("Optitrack Three: {}".format(optitrack_markers["three"]))
    logging.info("Optitrack Four: {}".format(optitrack_markers["four"]))

    # replace the inertia of the pendulum
    model_descr = model_descr.replace(
        "pendulum_inertia_tensor_full", inertia_tensor.tensor_to_str()
    )
    model_descr = model_descr.replace("pendulum_inertia_tensor_xx", inertia_tensor.x)
    model_descr = model_descr.replace("pendulum_inertia_tensor_yy", inertia_tensor.y)
    model_descr = model_descr.replace("pendulum_inertia_tensor_zz", inertia_tensor.z)

    # replace the mass of the pendulum
    model_descr = model_descr.replace("pendulum_mass", str(pendulum_mass))

    # replace the length of the pendulum and the position of the sensors
    model_descr = model_descr.replace("pendulum_length_full", str(pendulum_len))

    # replace the position of the inertia mass
    model_descr = model_descr.replace(
        "pendulum_length_half", str(float(pendulum_len / 2))
    )
    # replace the optitrack markers
    for marker_idx, pos in optitrack_markers.items():
        model_descr = model_descr.replace(
            "optitrack_marker_{}_pos".format(marker_idx), "{:4.5f}".format(pos)
        )

    # print(model_descr)
    return model_descr


def _optitrack_marker_pos(
    pendulum_len: float = 0.3,
) -> Dict[str, float]:
    """Define the positions of the optitrack markers."""
    return dict(
        one=pendulum_len / 6,
        two=pendulum_len / 3,
        three=3 * pendulum_len / 4,
        four=pendulum_len - 0.01,
    )
