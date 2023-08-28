// main File to use the controller and the observer
// use ROS

#include <boost/filesystem.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <iostream>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
// new
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <crocoddyl/core/activations/quadratic.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/multibody/costs/state.hpp>
#include <iostream>

int main() {
  // Get the current path (executable or working directory)
  boost::filesystem::path currentPath = boost::filesystem::current_path();

  // Construct the relative path to the URDF file
  std::string urdfFileName =
      "rot_wam_pend_0.374_temp.urdf";  // Replace with your URDF file name
  boost::filesystem::path urdfFilePath =
      currentPath / "../src/wam" / urdfFileName;
  pinocchio::Model pin_model_;
  std::cout << "Using " << urdfFilePath.string() << "\n";
  pinocchio::urdf::buildModel(urdfFilePath.string(), pin_model_);
  pinocchio::Data pin_data = pinocchio::Data(pin_model_);

  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(
          boost::make_shared<pinocchio::Model>(pin_model_));
  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
      boost::make_shared<crocoddyl::ActuationModelFull>(state);
  std::cout << "working";
  // Create state, actuation, and model
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation =
      boost::make_shared<crocoddyl::ActivationModelQuad>(3);

  // Create cost models (we'll use two quadratic costs for demonstration
  // purposes)
  boost::shared_ptr<crocoddyl::CostModelAbstract> cost1 =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelState>(
                     state, state->zero()));

  // Add the cost models to the cost sum (you can add as many as you need)
  boost::shared_ptr<crocoddyl::CostModelSum> cost_sum =
      boost::make_shared<crocoddyl::CostModelSum>(state);
  cost_sum->addCost("cost1", cost1, 1.0);  // Weight 1.0 for the first cost

  // Create terminal cost model (similar to the cost models above)
  boost::shared_ptr<crocoddyl::CostModelAbstract> terminal_cost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelState>(
                     state, state->zero()));

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          state, actuation, cost_sum);
  // Create the action model using the cost sum for intermediate nodes and the
  // terminal cost for the final node
  boost::shared_ptr<crocoddyl::ActionModelAbstract> action_model =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
          model, 0.1);  // Here, integration time is set to 0.1

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > cost_models;
  cost_models.push_back(action_model);

  // Set lower and upper control bounds (not shown in the example)
  // ...

  // Create initial state (not shown in the example)
  // ...
  // Create the initial state with 12 zeros
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(12);

  // Create the ShootingProblem
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, cost_models,
                                                     action_model);

  // Solve the problem using your preferred solver (not shown in the example)
  // ...

  std::cout << "working 2";
  return 0;
};