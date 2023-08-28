// #include <ros/ros.h>
// #include <trajectory_msgs/JointTrajectory.h>
// #include <trajectory_msgs/JointTrajectoryPoint.h>
#ifndef TORQUES_WITH_TIME_HPP
#define TORQUES_WITH_TIME_HPP

#include <iostream>
#include <vector>

#include "../logger/Logger.hpp"
#include "Eigen/Dense"

namespace pendulum_acrobatics {
class ArrayWithDuration {
 public:
  ArrayWithDuration(const Eigen::VectorXd& array, double time_stamp,
                    double time_duration);
  ArrayWithDuration(const ArrayWithDuration& array_with_duration);
  ArrayWithDuration(int size);
  const Eigen::VectorXd& getArray() const;
  double getTimestamp() const;
  double getDuration() const;

 private:
  Eigen::VectorXd array_;
  double time_duration_;
  double time_stamp_;
};

class TorquesWithTime {
 public:
  TorquesWithTime(double header_time, std::vector<Eigen::VectorXd>& torque_list,
                  int factor_integration_time, double integration_time = 0.002);
  TorquesWithTime(int mpc_horizon, int factor_integration_time,
                  double integration_time = 0.002, int torque_size = 6);
  void update_solution_torques(double header_time,
                               const std::vector<Eigen::VectorXd>& torque_list);
  ArrayWithDuration get_current_torque(double time_stamp);

 private:
  double header_time_;
  int factor_integration_time_;
  double integration_time_;
  std::vector<ArrayWithDuration> torque_buffer_1_;
  std::vector<ArrayWithDuration> torque_buffer_2_;
  std::vector<ArrayWithDuration>* active_torque_buffer;
};
}  // namespace pendulum_acrobatics
#endif