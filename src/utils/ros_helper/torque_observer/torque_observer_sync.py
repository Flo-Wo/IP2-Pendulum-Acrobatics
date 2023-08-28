import multiprocessing
from copy import deepcopy

import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory


class TorqueObserverSync:
    def __init__(self, block_until_first_data: bool = True):
        self.cv = multiprocessing.Condition()
        self.joint_traj_msg: JointTrajectory = None

        self._torques_available: bool = not block_until_first_data
        self._received_first_torques: bool = False
        self.synchronized = block_until_first_data

    def write_torques(self, joint_traj_msg: JointTrajectory) -> None:
        if (
            self.joint_traj_msg is not None
            and joint_traj_msg.header.stamp <= self.joint_traj_msg.header.stamp
        ):
            # catch that the first response is sent twice due to parallelism
            return
        with self.cv:
            # write torque updates with a thread lock
            self.joint_traj_msg = joint_traj_msg
            self.cv.notify_all()
            self._new_torques_available = True
            self._received_first_torques = True
            rospy.loginfo(
                "NEW TORQUES WRITTEN: {}.{}".format(
                    joint_traj_msg.header.stamp.secs, joint_traj_msg.header.stamp.nsecs
                )
            )

    def new_torques_available(self) -> bool:
        return self._new_torques_available

    def received_first_torques(self) -> bool:
        return self._received_first_torques

    def invalidate_data(self) -> None:
        self._new_torques_available = False

    def get_joint_traj(self) -> JointTrajectory:
        with self.cv:
            self.cv.wait_for(self.new_torques_available)
            # if self.synchronized:
            #     self._new_torques_available = False
            return deepcopy(self.joint_traj_msg)
