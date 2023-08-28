import threading

import numpy as np
import pinocchio as pin
from rospy import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from utils.ros_helper.state_observer import StateObserverBase


class StateObserverPinSim(StateObserverBase):
    def __init__(self, pin_model, time_interval: float, time_step: float = 0.005):
        super(StateObserverBase, self).__init__()

        self.pin_model = pin_model
        self.pin_data = self.pin_model.createData()
        # self.pin_model, self.pin_data = self._setup_pinocchio(pin_model)

        self.time_step = time_step
        self.num_int_steps = int(np.ceil(time_interval / time_step))

        self.lock = threading.Lock()
        self.state: JointState = None
        self._state_available = False

    def get_joint_state(self) -> JointState:
        q, dq, tau, header = self._get_curr_state()
        q_new, dq_new = self._semi_implicit_euler(q=q, dq=dq, tau=tau)

        return JointState(
            header=Header(
                stamp=header.stamp + Duration(secs=self.time_step * self.num_int_steps)
            ),
            position=q_new,
            velocity=dq_new,
        )

    def _get_curr_state(self) -> np.ndarray:
        with self.lock:
            return (
                np.array(self.state.position),
                np.array(self.state.velocity),
                np.array(self.state.effort),
                self.state.header,
            )

    def _setup_pinocchio(self, path_to_urdf: str):
        pin_model = pin.buildModelFromUrdf(path_to_urdf)
        pin_data = pin_model.createData()
        return pin_model, pin_data

    def _semi_implicit_euler(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
    ):
        """Semi-implicit Euler method, based on https://en.wikipedia.org/wiki/Semi-implicit_Euler_method.

        Parameters
        ----------
        q : np.ndarray
            Current configuration.
        dq : np.ndarray
            Current joint velocity.
        ddq : np.ndarray
            Current joint acceleration.

        Returns
        -------
        Tuple
            q, dq at the time: t0 + time_step
        """
        q_next, dq_next = q, dq

        for time_step in range(self.num_int_steps):

            ddq = pin.aba(self.pin_model, self.pin_data, q_next, dq_next, tau)
            dq_next = dq + self.time_step * ddq
            q_next = q + self.time_step * dq_next

        return q_next, dq_next
