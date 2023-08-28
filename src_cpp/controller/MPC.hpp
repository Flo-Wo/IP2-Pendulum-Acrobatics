#include <Eigen/Core>
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
#include "../datatypes/TorquesWithTime.hpp"
#include "../file_reader/FileReader.hpp"
#include "../logger/Logger.hpp"

// load pi constant
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef MPC_NODE
#define MPC_NODE
namespace pendulum_acrobatics {
class MPC {
 public:
  MPC(const std::string& pin_model_filepath,
      std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
      std::vector<float>& angleGoal, const Eigen::VectorXd& init_position,
      const Eigen::VectorXd& armature, const Eigen::VectorXd& damping,
      const Eigen::VectorXd& coulomb_friction,
      const Eigen::VectorXd& grav_comp_idxs, int factor_integration_time,
      int solver_max_iter = 20, int mpc_horizon = 20,
      float integration_time_500Hz = 0.002);

  void solve(Eigen::VectorXd state, double time, int time_idx,
             std::vector<Eigen::VectorXd>& xs,
             std::vector<Eigen::VectorXd>& us);
  const std::vector<Eigen::VectorXd>& get_xs();
  const std::vector<Eigen::VectorXd>& get_us();
  int get_mpc_horizon();
  int get_integration_factor();
  const Eigen::VectorXd& get_torque_to_curr_time(double curr_time);
  void update_active_cost_models(int time_idx);
  void append_active_cost_models(int time_idx);
  void load_active_cost_models(int time_idx);
  void setUpSolver(const Eigen::VectorXd& init_pos);
  void setUpActiveCostModels();
  boost::shared_ptr<crocoddyl::ActionModelAbstract> setUpCostModels(
      std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
      std::vector<float>& angleGoal,
      const Eigen::VectorXd& init_pos = {0.0, -1.3, 0.0, 1.3, 0.0, 0.0});

  // class attributes
  int solver_max_iter_;
  int mpc_horizon_;
  int factor_integration_time_;
  float integration_time_;
  const Eigen::VectorXd& armature_;
  const Eigen::VectorXd& damping_;
  const Eigen::VectorXd& coulomb_friction_;
  const Eigen::VectorXd& grav_comp_idxs_;
  pinocchio::Model pin_model_;
  pinocchio::Data* pin_data_;
  // cost models
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
      running_models_;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminal_model_;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
      active_cost_models_;
  // store warmstart solution
  std::vector<Eigen::VectorXd> warmstart_xs_;
  // store problem, solver and solutions
  boost::shared_ptr<crocoddyl::ShootingProblem> problem;
  boost::shared_ptr<crocoddyl::SolverBoxFDDP> solver;

  // store the computed torques with timestamps
  TorquesWithTime torques_with_time_;

  // store last time
  int last_time_idx_;
  bool first_call_;
};

}  // namespace pendulum_acrobatics
#endif