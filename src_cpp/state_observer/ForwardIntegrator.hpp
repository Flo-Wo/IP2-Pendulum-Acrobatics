
#ifndef PENDULUM_ACROBATICS_FORWARD_INTEGRATOR_HPP
#define PENDULUM_ACROBATICS_FORWARD_INTEGRATOR_HPP

#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include "../action_model/FrictionModel.hpp"
#include "../datatypes/TorquesWithTime.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/fwd.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
// filesystem
#include <boost/filesystem.hpp>
// logging and datatypes
#include "../datatypes/TorquesWithTime.hpp"
#include "../logger/Logger.hpp"
namespace pendulum_acrobatics {

class ForwardIntegrator {
 public:
  ForwardIntegrator(const std::string relative_path_urdf,
                    const Eigen::VectorXd& armature,
                    const Eigen::VectorXd& damping,
                    const Eigen::VectorXd& coulomb_friction,
                    double integration_time_step);

  double semi_implicit_euler(Eigen::VectorXd q, Eigen::VectorXd dq,
                             TorquesWithTime& torques_with_time,
                             int num_of_500Hz_steps_to_integrate,
                             double curr_time, Eigen::VectorXd& return_state);

 protected:
  pinocchio::Model pin_model_;
  pinocchio::Data* pin_data_;
  Eigen::MatrixXd armature_;
  FrictionModel friction_model_;
  double integration_time_step_;
};
}  // namespace pendulum_acrobatics
#endif  // PENDULUM_ACROBATICS_FORWARD_INTEGRATOR_HPP