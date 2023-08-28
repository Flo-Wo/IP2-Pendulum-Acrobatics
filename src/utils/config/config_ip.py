import numpy as np

from utils.config.config_base import ConfigModelBase


class ConfigWAM(ConfigModelBase):
    def __init__(
        self,
        model_name: str,
        pendulum_len: float = 0.174,
        pendulum_radius: float = 0.005,
        pendulum_density: float = 2700,
        use_friction_action_model: bool = False,
    ):
        """Configuration class for our WAM models used in all experiments.

        Parameters
        ----------
        model_name : str
            A unique model name which is mapped to a specific configuration.
        len_pendulum : float, optional
            Length of the pendulum, by default 0.3.
        use_friction_action_model : bool, optional
            Use custom gravity compensation model, by default False. If True,
            we do not ignore fricton, damping and armature anymore. If False,
            these values are set to zero and thus ignored.

        Raises
        ------
        NotImplementedError
            If you choose a wrong model_name.
        """
        if model_name == "standard":
            q_config = np.array([0, -0.3, 0, 0.3, 0, 0])
            model_type = "normal"
        elif model_name == "standard_angled":
            q_config = np.array([0, -1.7, 0, 1.7, 0, 0])
            model_type = "normal"
        elif model_name == "rotated":
            # q_config = np.array([0, -0.78, 0, 2.37, 0, 0])
            q_config = np.array([0.0, -1.3, 0.0, 1.3, 0.0, 0.0])
            model_type = "rot"
        elif model_name == "human":
            q_config = np.array([0, -1.6, 1.55, 1.6, 0, 1.55])
            model_type = "rot"
        elif model_name == "rotated_without_pendulum":
            q_config = np.array([0, -0.78, 0, 2.37])
            model_type = "without_pendulum"
        else:
            raise NotImplementedError("Wrong argument for the WAM model.")

        super(ConfigWAM, self).__init__(
            q_config,
            model_name,
            model_type=model_type,
            pendulum_len=pendulum_len,
            pendulum_radius=pendulum_radius,
            pendulum_density=pendulum_density,
            use_friction_action_model=use_friction_action_model,
        )
