import multiprocessing
from abc import ABC, abstractmethod

import rospy
from sensor_msgs.msg import JointState


class StateObserverBase(ABC):
    state: JointState
    cv: multiprocessing.Condition
    _state_available: bool

    @abstractmethod
    def get_joint_state(self) -> JointState:
        """Return the joint state as a joint state message.

        Returns
        -------
        JointState
            ROS joint state message.

        Raises
        ------
        NotImplementedError
            Abstract class.
        """
        raise NotImplementedError

    def update_state(self, joint_state: JointState) -> None:
        rospy.loginfo(
            "\nRECEIVED NEW STATE: {}.{}".format(
                joint_state.header.stamp.secs, joint_state.header.stamp.nsecs
            )
        )
        with self.cv:
            self.state = joint_state
            self._state_available = True
            self.cv.notify_all()

    def state_is_available(self) -> bool:
        return self._state_available

    def invalidate_data(self):
        self._state_available = False
