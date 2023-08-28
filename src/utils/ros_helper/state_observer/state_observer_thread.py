import threading
from copy import deepcopy

from sensor_msgs.msg import JointState

from utils.ros_helper.state_observer import StateObserverBase


class StateObserverThread(StateObserverBase):
    state: JointState

    def __init__(self):
        super(StateObserverBase, self).__init__()
        self.lock = threading.Lock()
        self.state = None
        self._state_available = False

    def get_joint_state(self) -> JointState:
        return deepcopy(self.state)
