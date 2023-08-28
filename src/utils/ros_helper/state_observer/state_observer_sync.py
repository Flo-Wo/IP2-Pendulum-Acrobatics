import multiprocessing
import threading
from copy import deepcopy

from sensor_msgs.msg import JointState

from utils.ros_helper.state_observer import StateObserverBase


class StateObserverSync(StateObserverBase):
    state: JointState

    def __init__(self):
        super(StateObserverBase, self).__init__()
        self.cv = multiprocessing.Condition()
        self.state = None
        self._state_available = False

    def get_joint_state(self) -> JointState:
        with self.cv:
            self.cv.wait_for(self.state_is_available)
            return deepcopy(self.state)
