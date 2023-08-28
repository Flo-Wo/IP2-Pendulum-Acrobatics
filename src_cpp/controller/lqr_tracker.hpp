#ifndef BARRETT_TRACKING_ROS_LQR_TRACKER_HPP
#define BARRETT_TRACKING_ROS_LQR_TRACKER_HPP

// crocoddyl inputs
#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

// cost models
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/multibody/residuals/frame-rotation.hpp>
#include <crocoddyl/multibody/residuals/frame-translation.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>

// integration
#include <crocoddyl/core/integrator/euler.hpp>

// Solver
#include <crocoddyl/core/solvers/box-fddp.hpp>
// Callback methods
#include <Eigen/Core>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <vector>

// filesystem
#include <boost/filesystem.hpp>
#include <iostream>

// pinocchio inputs
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

// crocoddyl inputs
#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/activations/weighted-quadratic-barrier.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

// cost models
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/multibody/residuals/frame-rotation.hpp>
#include <crocoddyl/multibody/residuals/frame-translation.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>

// integration
#include <crocoddyl/core/integrator/euler.hpp>

// Solver
#include <crocoddyl/core/solvers/box-fddp.hpp>
// Callback methods
#include <crocoddyl/core/utils/callbacks.hpp>

// own methods and classes
#include "../action_model/ActionModel.hpp"
#include "../datatypes/State.hpp"
#include "../datatypes/TorquesWithTime.hpp"
#include "../file_reader/FileReader.hpp"
#include "../logger/Logger.hpp"

namespace barrett_pendulum_control {

class LQRTracker {
 public:
  LQRTracker(std::string urdf_path, double control_time,
             std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
             const Eigen::VectorXd& init_pos);
  virtual ~LQRTracker();
  virtual Eigen::VectorXd get_torques(double cur_time,
                                      const Eigen::VectorXd& grav_comp_torques);

 private:
  pinocchio::Model pin_model_;
  pinocchio::Data* pin_data_;
  double integration_time_;
  double start_time;

  std::vector<Eigen::VectorXd> x_ffs;
  std::vector<Eigen::VectorXd> u_ffs;
  std::vector<Eigen::MatrixXd> k_mats;
  std::vector<Eigen::VectorXd> k_vecs;

  std::tuple<std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>,
             boost::shared_ptr<crocoddyl::ActionModelAbstract>>
  setUpCostModels(std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
                  const Eigen::VectorXd& init_pos);
};

}  // namespace barrett_pendulum_control

#endif
