#include "TorquesWithTime.hpp"

#include <iostream>

namespace pendulum_acrobatics {
ArrayWithDuration::ArrayWithDuration(const Eigen::VectorXd& array,
                                     double time_stamp, double time_duration) {
  this->array_ = array;
  this->time_stamp_ = time_stamp;
  this->time_duration_ = time_duration;
};
ArrayWithDuration::ArrayWithDuration(const ArrayWithDuration& other) {
  this->array_ = other.array_;
  this->time_stamp_ = other.time_stamp_;
  this->time_duration_ = other.time_duration_;
}

ArrayWithDuration::ArrayWithDuration(int size) {
  this->array_ = Eigen::VectorXd::Zero(size);
  this->time_stamp_ = 0.0;
  this->time_duration_ = 0.0;
}
const Eigen::VectorXd& ArrayWithDuration::getArray() const {
  return this->array_;
}
double ArrayWithDuration::getTimestamp() const { return this->time_stamp_; }
double ArrayWithDuration::getDuration() const { return this->time_duration_; }
// trajectory_msgs::JointTrajectoryPoint as_joint_traj_point() const {
//   trajectory_msgs::JointTrajectoryPoint joint_traj_point;
//   joint_traj_point.positions.clear();
//   joint_traj_point.positions.reserve(array_.size());
//   for (int i = 0; i < array_.size(); ++i) {
//     joint_traj_point.positions.push_back(array_[i]);
//   }
//   joint_traj_point.time_from_start = ros::Duration(time_stamp_);

//   return joint_traj_point;
// };

TorquesWithTime::TorquesWithTime(int mpc_horizon, int factor_integration_time,
                                 double integration_time, int torque_size)
    : header_time_(0.0),
      factor_integration_time_(factor_integration_time),
      integration_time_(integration_time),
      torque_buffer_1_(factor_integration_time * mpc_horizon,
                       ArrayWithDuration(torque_size)),
      torque_buffer_2_(factor_integration_time * mpc_horizon,
                       ArrayWithDuration(torque_size)),
      active_torque_buffer(&torque_buffer_1_){};

TorquesWithTime::TorquesWithTime(double header_time,
                                 std::vector<Eigen::VectorXd>& torque_list,
                                 int factor_integration_time,
                                 double integration_time)
    : header_time_(header_time),
      factor_integration_time_(factor_integration_time),
      integration_time_(integration_time) {
  // replicate the torques --> integration factor
  double time(this->header_time_);
  torque_buffer_1_.reserve(torque_list.size() * factor_integration_time);
  torque_buffer_2_.reserve(torque_list.size() * factor_integration_time);
  this->active_torque_buffer = &torque_buffer_1_;

  for (size_t idx = 0; idx < torque_list.size(); ++idx) {
    Eigen::VectorXd tau(torque_list[idx]);
    // The pendulum is not actutated
    tau.tail(2).setZero();
    for (int i = 0; i < this->factor_integration_time_; ++i) {
      const ArrayWithDuration torque_with_time(tau, time,
                                               this->integration_time_);
      torque_buffer_1_.emplace_back(torque_with_time);
      torque_buffer_2_.emplace_back(torque_with_time);
      // std::cout << "adding, time: " << time << "\n";
      time += integration_time_;
    }
    // capacity -> is now constant
    // std::cout << this->torques_with_duration_.capacity() << "\n";
    Logger::log(LogLevel::DEBUG,
                "TorquesWithTime::TorquesWithTime, buffer 1, size: " +
                    std::to_string(this->torque_buffer_1_.size()) + "\n");
    Logger::log(LogLevel::DEBUG,
                "TorquesWithTime::TorquesWithTime, buffer 2, size: " +
                    std::to_string(this->torque_buffer_1_.size()) + "\n");
  }
};
void TorquesWithTime::update_solution_torques(
    double header_time, const std::vector<Eigen::VectorXd>& torque_list) {
  this->header_time_ = header_time;
  double time(this->header_time_);

  int index = 0;

  // determine the inactive torque buffer
  std::vector<ArrayWithDuration>* inactive_torque_buffer =
      (this->active_torque_buffer == &torque_buffer_1_) ? &torque_buffer_2_
                                                        : &torque_buffer_1_;

  for (size_t idx = 0; idx < torque_list.size(); ++idx) {
    Eigen::VectorXd tau(torque_list[idx]);
    // The pendulum is not actutated
    tau.tail(2).setZero();
    for (int i = 0; i < this->factor_integration_time_; ++i) {
      const ArrayWithDuration torque_with_time(tau, time,
                                               this->integration_time_);
      (*inactive_torque_buffer)[index] = torque_with_time;
      time += this->integration_time_;
      index++;
    }
    // capacity -> is now constant
    Logger::log(LogLevel::DEBUG,
                "TorquesWithTime::TorquesWithTime, size: " +
                    std::to_string(this->active_torque_buffer->size()) + "\n");
  }
  // switch buffer
  this->active_torque_buffer = inactive_torque_buffer;
};
ArrayWithDuration TorquesWithTime::get_current_torque(double time_stamp) {
  double min_time_diff = std::numeric_limits<double>::max();
  ArrayWithDuration torque_with_min_timediff(6);

  Logger::log(LogLevel::DEBUG,
              "TorquesWithTime::get_current_torque, time input: " +
                  std::to_string(time_stamp) + "\n");
  // std::cout << "length: " << torques_with_duration_.size() << "\n";
  for (const auto& torque_with_duration : *active_torque_buffer) {
    double time_diff =
        std::abs(torque_with_duration.getTimestamp() - time_stamp);
    // std::cout << "time_diff: " << time_diff << "\n";
    if (time_diff < min_time_diff) {
      min_time_diff = time_diff;
      // std::cout << "update min torque time: "
      //           << torque_with_duration.getTimestamp() << "\n";
      // std::cout << "min timediff: " << min_time_diff << "\n";
      torque_with_min_timediff = torque_with_duration;
    }
  }

  return torque_with_min_timediff;
};
// trajectory_msgs::JointTrajectory to_joint_trajectory_message() {
//   trajectory_msgs::JointTrajectory joint_trajectory;
//   joint_trajectory.header.stamp = ros::Time(header_time_);

//   for (const auto& torque_with_duration : torques_with_duration_) {
//     trajectory_msgs::JointTrajectoryPoint joint_traj_point;
//     joint_traj_point.positions.resize(torque_with_duration.getArray().size());
//     for (int i = 0; i < torque_with_duration.getArray().size(); ++i) {
//       joint_traj_point.positions[i] = torque_with_duration.getArray()(i);
//     }
//     joint_traj_point.time_from_start =
//         ros::Duration(torque_with_duration.getDuration());
//     joint_trajectory.points.push_back(joint_traj_point);
//   }

//   return joint_trajectory;
// };
}  // namespace pendulum_acrobatics

/*
int main() {
  // Example usage of TorquesWithTime

  // Some example data for torques
  std::vector<Eigen::VectorXd> torque_list;
  for (int i = 0; i < 20; ++i) {
    Eigen::VectorXd tau(6);
    tau << i * 0.1, i * 0.2, i * 0.3, i * 0.4, i * 0.5, i * 0.6;
    torque_list.push_back(tau);
    std::cout << torque_list.size() << "\n";
  }

  // Create an instance of TorquesWithTime
  double header_time = 0.0;
  int mpc_horizon = 20;
  int factor_integration_time = 2;
  float integration_time = 0.002;

  // pendulum_acrobatics::TorquesWithTime torques_with_time(
  //     header_time, torque_list, factor_integration_time, integration_time);
  pendulum_acrobatics::TorquesWithTime torques_with_time(
      mpc_horizon, factor_integration_time);
  torques_with_time.update_solution_torques(header_time, torque_list);

  // Test get_current_torque method
  for (int i = 0; i < 20; i += 5) {
    double time_stamp = 0.002 * i;
    Eigen::VectorXd current_torque =
        torques_with_time.get_current_torque(time_stamp).getArray();
    std::cout << "Torque at time stamp " << time_stamp << ": "
              << current_torque.transpose() << std::endl;
  }
  std::vector<Eigen::VectorXd> torque_list_update;
  for (int i = 0; i < 200; i += 10) {
    Eigen::VectorXd tau(6);
    tau << i * 0.1, i * 0.2, i * 0.3, i * 0.4, i * 0.5, i * 0.6;
    torque_list_update.push_back(tau);
    std::cout << torque_list_update.size() << "\n";
  }
  torques_with_time.update_solution_torques(0.0, torque_list_update);
  // Test get_current_torque method
  for (int i = 0; i < 20; i += 5) {
    double time_stamp = 0.002 * i;
    Eigen::VectorXd current_torque =
        torques_with_time.get_current_torque(time_stamp).getArray();
    std::cout << "Torque at time stamp " << time_stamp << ": "
              << current_torque.transpose() << std::endl;
  }
  // Test to_joint_trajectory_message method
  // trajectory_msgs::JointTrajectory joint_traj_msg =
  //     torques_with_time.to_joint_trajectory_message();

  // // Display the trajectory points in the joint trajectory message
  // for (const auto& point : joint_traj_msg.points) {
  //   std::cout << "Positions: ";
  //   for (const auto& pos : point.positions) {
  //     std::cout << pos << " ";
  //   }
  //   std::cout << ", Time from start: " << point.time_from_start.toSec()
  //             << " seconds" << std::endl;
  // }

  return 0;
}
*/