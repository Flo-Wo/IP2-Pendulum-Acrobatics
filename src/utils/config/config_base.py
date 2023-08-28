import logging

import mujoco
import numpy as np

from utils import DataBase
from utils.config.config_utils import read_model
from utils.config.prepro_calculations import cylinder_mass_from_density


class ConfigModelBase(DataBase):
    def __init__(
        self,
        q_config: np.ndarray,
        model_name: str,
        model_type: str = "normal",
        pendulum_len: float = 0.174,
        pendulum_radius: float = 0.005,
        pendulum_density: float = 2700,
        use_friction_action_model: bool = False,
    ):
        """Custom Config Model for our IP Experiments, including Pinocchio and Mujoco models

        Parameters
        ----------
        q_config : np.ndarray
            Start configuration of the robot, set for pinocchio and mujoco.
        model_name : str
            Name of the model, usually rotated.
        model_type : str, optional
            Either normal or rotated, by default "normal"
        pendulum_len : float, optional
            Length of the pendulum in [m], by default 0.174.
        pendulum_radius : float, optional
            Radius of the pendulum in [m], by default 0.005.
        pendulum_density : float, optional
            Density of the pendulum in [kg/m^3], by default 2700
        use_friction_action_model : bool, optional
            Use custom gravity compensation model, by default False. If True,
            we do not ignore fricton, damping and armature anymore. If False,
            these values are set to zero and thus ignored.
        """

        self.model_type = model_type
        # set the pendulum length
        self.len_pendulum = pendulum_len
        pendulum_mass = cylinder_mass_from_density(
            height=pendulum_len, radius=pendulum_radius, density=pendulum_density
        )
        print("Pendulum density: ", pendulum_density)
        print("Pendulum length: ", pendulum_len)
        print("Pendulum radius: ", pendulum_radius)
        print("Pendulum mass: ", pendulum_mass)
        self.pendulum_mass = pendulum_mass
        self.pendulum_radius = pendulum_radius

        # read the models
        self.mj_model = read_model(
            model_type,
            ".xml",
            pendulum_len=pendulum_len,
            pendulum_radius=pendulum_radius,
            pendulum_mass=pendulum_mass,
        )
        self.mj_data = mujoco.MjData(self.mj_model)

        if not use_friction_action_model:
            logging.info("\n\nSetting: ")

            logging.info("Damping to zero.")
            self.mj_model.dof_damping = np.zeros(np.shape(self.mj_model.dof_damping))

            logging.info("Aramture to zero.")
            self.mj_model.dof_armature = np.zeros(np.shape(self.mj_model.dof_armature))

            logging.info("Friction to zero.")
            self.mj_model.dof_frictionloss = np.zeros(
                np.shape(self.mj_model.dof_frictionloss)
            )

        logging.info("\n\n CONFIG BASE")
        print("Damping: ", self.mj_model.dof_damping)
        print("Armature: ", self.mj_model.dof_armature)
        print("Friction: ", self.mj_model.dof_frictionloss)
        logging.info("\n\n")

        self.pin_robot = read_model(
            model_type,
            ".urdf",
            pendulum_len=pendulum_len,
            pendulum_radius=pendulum_radius,
            pendulum_mass=pendulum_mass,
        )
        self.pin_model = self.pin_robot.model
        self.pin_data = self.pin_robot.data
        # self.pin_data = self.pin_model.createData()

        # set the robot to it's start configuration
        self.mj_data.qpos[:] = q_config
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.q_config = q_config
        # define a string representation for automatic filename generation
        self.model_name = model_name

    # DataBase method's implementations, a config is uniquely defined by the model type,
    # the config string and the length of the pendulum
    def __str__(self):
        return "{}_{}_{}".format(self.model_type, self.model_name, self.len_pendulum)

    def save_to_file(self) -> dict:
        return {}

    def save_to_metadata(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "len_pendulum": self.len_pendulum,
            "mass_pendulum": self.pendulum_mass,
            "radius_pendulum": self.pendulum_radius,
        }

    def get_pend_end_world(self):
        return self.mj_data.sensordata[0:3]

    def get_pend_pole_tip_id(self):
        return self.pin_model.getFrameId("links/pendulum/pole_tip")

    def get_pend_beg_id(self):
        return self.pin_model.getFrameId("links/pendulum/rotating_x_axis")
