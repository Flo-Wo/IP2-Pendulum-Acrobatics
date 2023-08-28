import copy
import threading

import rospy
from trajectory_msgs.msg import JointTrajectory


class TorqueObserver:
    def __init__(self, block_until_first_data: bool = False):
        self.lock = threading.Lock()
        self.joint_traj_msg: JointTrajectory = None

        self._torques_available: bool = not block_until_first_data
        self._received_first_torques: bool = False

    def torques_available(self):
        return self._torques_available

    def received_data(self):
        return self._received_first_torques

    def write_torques(self, joint_traj_msg: JointTrajectory):
        with self.lock:
            self.joint_traj_msg = joint_traj_msg
            time = joint_traj_msg.header.stamp
            print("\n")
            rospy.loginfo("RECEIVED NEW TORQUES: {}.{}".format(time.secs, time.nsecs))
            print("\n")
            self._torques_available = True
            self._received_first_torques = True

    def read_torques(self):
        with self.lock:
            return copy.deepcopy(self.joint_traj_msg)
