#include "lqr_tracker.hpp"

namespace barrett_pendulum_control {

LQRTracker::LQRTracker(std::string urdf_path, double control_time,
                       std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
                       const Eigen::VectorXd& init_pos) {
  boost::filesystem::path currentPath_new = boost::filesystem::current_path();
  boost::filesystem::path urdfFilePath_new = currentPath_new / urdf_path;
  // pinocchio::Model pin_model_;
  Logger::log(LogLevel::INFO, "Using " + urdfFilePath_new.string() + "\n");
  pinocchio::urdf::buildModel(urdfFilePath_new.string(), pin_model_);
  // this->pin_model_.effortLimit = 3 * 0.005 * this->pin_model_.effortLimit;
  this->pin_data_ = new pinocchio::Data(this->pin_model_);
  this->integration_time_ = control_time;

  // build problem and solver
  Eigen::VectorXd x0(init_pos.size() + init_pos.size());
  x0 << init_pos, Eigen::VectorXd::Zero(init_pos.size());

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> cost_models;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminal_model;

  // Logger::setLogLevel(LogLevel::DEBUG);
  std::tie(cost_models, terminal_model) =
      setUpCostModels(pend_tip_pos_goal, init_pos);
  auto problem = boost::make_shared<crocoddyl::ShootingProblem>(x0, cost_models,
                                                                terminal_model);
  Logger::log(LogLevel::DEBUG, "MPC::setUpSolver: Building Solver.\n");
  auto solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem);
  // Logger::setLogLevel(LogLevel::DEBUG);
  std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
  boost::shared_ptr<crocoddyl::CallbackVerbose> cb =
      boost::make_shared<crocoddyl::CallbackVerbose>();
  cb->set_precision(10);
  cbs.push_back(cb);
  solver->setCallbacks(cbs);

  problem->set_x0(x0);
  Logger::log(LogLevel::DEBUG, "MPC::solve: WORKED set x0.\n");

  // provide space to save the results
  std::size_t T = problem->get_T();
  // std::cout << "T = " << T << "\n";
  Logger::log(LogLevel::DEBUG,
              "MPC::solve: WORKED get T: " + std::to_string(T) + ".\n");

  // Eigen::VectorXd warmstart_state(state);
  // warmstart_state << state.head(6), Eigen::VectorXd::Zero(6);

  Eigen::IOFormat fmt(Eigen::FullPrecision, 0, ", ", "\n", "[", "]");
  // std::cout << "x0 = " << x0.format(fmt) << "\n";
  std::vector<Eigen::VectorXd> xs_warm(T, x0), xs(T + 1, x0),
      us(T, Eigen::VectorXd::Zero(6));

  problem->quasiStatic(us, xs_warm);
  solver->setCandidate(xs, us, true);
  // auto u = us.back();
  // std::cout << "u = " << u.format(fmt) << "\n";

  bool success = solver->solve(xs, us, 5000, true);

  // for (auto xs : solver->get_xs()){
  //     std::cout << "xs = " << Logger::matrixToString(xs) << "\n";
  // }
  x_ffs = solver->get_xs();
  u_ffs = solver->get_us();
  for (uint32_t i = 0; i < solver->get_Quu_inv().size(); ++i) {
    k_mats.push_back(solver->get_Quu_inv()[i] *
                     solver->get_Qxu()[i].transpose());
    k_vecs.push_back(solver->get_Quu_inv()[i] * solver->get_Qu()[i]);
    if (i % 500 == 0) {
      std::cout << "mat and vec, index = " << i << "\n";
      std::cout << "solver.xs = " << solver->get_xs()[i].format(fmt) << "\n";
      std::cout << "solver.us = " << solver->get_us()[i].format(fmt) << "\n";

      std::cout << "Quu_inv = " << solver->get_Quu_inv()[i].format(fmt) << "\n";
      std::cout << "Qxu = " << solver->get_Qxu()[i].transpose().format(fmt)
                << "\n";
      std::cout << "Qu = " << solver->get_Qu()[i].format(fmt) << "\n";
      std::cout << "K = " << k_mats.back().format(fmt) << "\n";
      std::cout << "k = " << k_vecs.back().format(fmt) << "\n";
    }
  }

  start_time = -1;
}

LQRTracker::~LQRTracker() { delete this->pin_data_; }

Eigen::VectorXd LQRTracker::get_torques(
    double cur_time, const Eigen::VectorXd& grav_comp_torques) {
  // : we already pass in the relative time
  /*
  if (start_time < 0) {
      start_time = cur_time;
  }

  double obs_time = this->observer->get_state(this->state);
  double rel_time = (obs_time - cur_time);
  if (rel_time < 0) {
      rel_time = 0;
  }
  */
  int idx = (int)std::floor(cur_time / this->integration_time_);

  // std::cout << "IDX: " << idx << "\n";

  Eigen::VectorXd full_state(12);
  pendulum_acrobatics::State state;
  full_state << state.position, state.velocity;
  full_state -= this->x_ffs[idx];

  // std::cout << "solver_torques = " << this->u_ffs[idx] << "\n";
  // std::cout << "x_ffs = " << this->x_ffs[idx] << "\n";

  // std::cout << "K * state = " << this->k_mats[idx] * full_state << "\n";
  // std::cout << "k = " << this->k_vecs[idx] << "\n";
  // std::cout << "gravity = " << grav_comp_torques << "\n";

  // std::cout << "torque = "
  //           << this->u_ffs[idx] - this->k_mats[idx] * full_state -
  //                  this->k_vecs[idx] + grav_comp_torques
  //           << "\n";
  return this->u_ffs[idx] - this->k_mats[idx] * full_state - this->k_vecs[idx] +
         grav_comp_torques;
}

std::tuple<std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>,
           boost::shared_ptr<crocoddyl::ActionModelAbstract>>
LQRTracker::setUpCostModels(std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
                            const Eigen::VectorXd& init_pos) {
  // ==== CROCODDYL ====
  Logger::log(LogLevel::DEBUG, "Setup solver cost models started.\n");

  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(
          boost::make_shared<pinocchio::Model>(this->pin_model_));
  Logger::log(LogLevel::DEBUG, "State is built correctly.\n");

  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
      boost::make_shared<crocoddyl::ActuationModelFull>(state);
  Logger::log(LogLevel::DEBUG, "Actuation is built correctly.\n");

  pinocchio::FrameIndex target_link_id = this->pin_model_.getFrameId(
      "links/pendulum/pole_tip", pinocchio::FrameType::BODY);

  // std::cout << "Frame ID: " << target_link_id << "\n";

  // ==== INIT POSITION AND OFFSET ====
  Logger::log(LogLevel::DEBUG, "Computing the initial position.\n");
  // make shared pointers
  pinocchio::forwardKinematics(this->pin_model_, *(this->pin_data_), init_pos);
  pinocchio::updateFramePlacements(this->pin_model_, *(this->pin_data_));

  const Eigen::VectorXd target_offset(
      this->pin_data_->oMf[target_link_id].translation());
  Logger::log(LogLevel::DEBUG,
              "target offset " + Logger::matrixToString(target_offset) + ".\n");
  std::cout << "target_offset: \n" << target_offset << "\n";

  std::size_t nu = this->pin_model_.nv;

  // define the init position
  Eigen::VectorXd x0(init_pos.size() + init_pos.size());
  x0 << init_pos, Eigen::VectorXd::Zero(init_pos.size());

  // ==== COST MODELS ====
  // Costs that are shared between all time-steps
  Logger::log(LogLevel::DEBUG, "Building shared cost models. \n");
  // STATE penalty
  boost::shared_ptr<crocoddyl::CostModelAbstract> stateRegCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          // boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(
          //     state_pen_weights),
          boost::make_shared<crocoddyl::ResidualModelState>(
              state, x0, actuation->get_nu()));
  // CONTROL penalty
  boost::shared_ptr<crocoddyl::CostModelAbstract> uRegCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelControl>(
                     state, actuation->get_nu()));
  // limit costs
  Eigen::VectorXd lb = this->pin_model_.lowerPositionLimit.array() + 0.1;
  Eigen::VectorXd ub = this->pin_model_.upperPositionLimit.array() - 0.1;
  Eigen::VectorXd velocityLimit = this->pin_model_.velocityLimit;

  Eigen::VectorXd damping(6);
  damping.setZero();

  Eigen::VectorXd armature(6);
  armature.setZero();

  Eigen::VectorXd coulomb_friction(6);
  coulomb_friction.setZero();

  // Concatenate the position and velocity limits
  Eigen::VectorXd lb_all(this->pin_model_.nv + this->pin_model_.nv);
  lb_all << lb, -velocityLimit;
  Eigen::VectorXd ub_all(this->pin_model_.nv + this->pin_model_.nv);
  ub_all << ub, velocityLimit;

  crocoddyl::ActivationBounds bounds(lb_all, ub_all, 1.0);

  boost::shared_ptr<crocoddyl::CostModelAbstract> limitCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
              bounds),
          boost::make_shared<crocoddyl::ResidualModelState>(
              state, state->zero(), actuation->get_nu()));

  Logger::log(LogLevel::DEBUG, "Building shared cost models. \n");

  Logger::log(LogLevel::DEBUG, "WORKED: Building shared cost models. \n");
  // TIME DEPENDENT COST MODELS
  std::size_t traj_size = pend_tip_pos_goal.size();

  Logger::log(LogLevel::DEBUG, "Starting Loop. \n");
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
      running_models_;
  for (int time_idx = 0; time_idx < traj_size; ++time_idx) {
    // positional error of the pendulum
    Eigen::VectorXd target_pos(pend_tip_pos_goal[time_idx] + target_offset);
    if (time_idx % 100 == 0) {
      std::cout << "time_idx = " << time_idx << "\n";
      std::cout << "target = " << target_pos << "\n";
      std::cout << "dt = " << this->integration_time_ << "\n";
      Logger::log(LogLevel::DEBUG,
                  "Loop index: " + std::to_string(time_idx) + "\n");
      Logger::log(LogLevel::DEBUG,
                  "target: " + Logger::matrixToString(target_pos) + "\n");
    }

    boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
        boost::make_shared<crocoddyl::CostModelResidual>(
            state, boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                       state, target_link_id, target_pos, actuation->get_nu()));

    boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel =
        boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
    runningCostModel->addCost("pendulum_pos_costs", goalTrackingCost, 1e3);
    runningCostModel->addCost("xReg", stateRegCost, 1e-1);
    runningCostModel->addCost("uReg", uRegCost, 1e-2);
    runningCostModel->addCost("jointLimits", limitCost, 1e6);

    Logger::log(LogLevel::DEBUG, "WORKED: cost model sum\n");
    // integrate the cost models
    // boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics>
    //     runningDAM = boost::make_shared<
    //         crocoddyl::DifferentialActionModelFreeFwdDynamics>(
    //         state, actuation, runningCostModel);

    boost::shared_ptr<
        pendulum_acrobatics::DifferentialActionModelFrictionFwdDynamics>
        runningDAM = boost::make_shared<
            pendulum_acrobatics::DifferentialActionModelFrictionFwdDynamics>(
            state, actuation, runningCostModel, armature, damping,
            coulomb_friction);

    Logger::log(LogLevel::DEBUG, "WORKED: action model constructor\n");

    boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModelEuler =
        boost::make_shared<crocoddyl::IntegratedActionModelEuler>(runningDAM,
                                                                  0.008);
    Logger::log(LogLevel::DEBUG, "WORKED: integration\n");
    runningModelEuler->set_u_lb((-1.0) * this->pin_model_.effortLimit);
    runningModelEuler->set_u_ub(this->pin_model_.effortLimit);

    running_models_.push_back(runningModelEuler);
    Logger::log(LogLevel::DEBUG, "WORKED: push back\n");
  }

  // build the target cost model

  Eigen::VectorXd terminal_target_pos(pend_tip_pos_goal.back() + target_offset);
  boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
              state, target_link_id, terminal_target_pos, actuation->get_nu()));
  boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel =
      boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  terminalCostModel->addCost("pendulum_pos_costs", goalTrackingCost, 1e3);
  terminalCostModel->addCost("xReg", stateRegCost, 1e-1);
  terminalCostModel->addCost("jointLimits", limitCost, 1e6);

  // integrate the cost models
  boost::shared_ptr<
      pendulum_acrobatics::DifferentialActionModelFrictionFwdDynamics>
      terminalDAM = boost::make_shared<
          pendulum_acrobatics::DifferentialActionModelFrictionFwdDynamics>(
          state, actuation, terminalCostModel, armature, damping,
          coulomb_friction);
  // boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics>
  //     terminalDAM =
  //         boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
  //             state, actuation, terminalCostModel);
  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminal_model =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(terminalDAM,
                                                                0.0);

  return std::make_tuple(running_models_, terminal_model);
}

}  // namespace barrett_pendulum_control

int main(int argc, char* argv[]) {
  Logger::setLogLevel(LogLevel::INFO);
  if (argc > 3) {
    std::string arg = argv[3];
    if (arg == "DEBUG") {
      Logger::setLogLevel(LogLevel::DEBUG);
    }
  }
  std::string pin_model_filepath = argv[1];
  // Vector of 3D test vectors
  // std::vector<Eigen::VectorXd> pend_tip_pos_goal;

  boost::filesystem::path current_path = boost::filesystem::current_path();
  std::string target_path = argv[2];
  boost::filesystem::path states_path =
      current_path / "controller/data/" / target_path;
  std::vector<Eigen::VectorXd> pend_tip_pos_goal =
      pendulum_acrobatics::FileReader::readNumpyTxt(states_path.string());

  // for (int i = 0; i < num_cost_models; ++i) {
  //   // Creating a 3D test vector (replace these values with your desired test
  //   // data)
  //   Eigen::VectorXd test_vector(3);
  //   // test_vector << i * 0.1, i * 0.2, i * 0.3;
  //   test_vector.setZero();
  //   pend_tip_pos_goal.push_back(test_vector);
  // }

  Eigen::VectorXd init_position(6);  // Add test data as needed
  init_position << 0.0, -1.3, 0.0, 1.3, 0.0, 0.0;

  std::cout << "Before Class constructor\n";
  // Create an instance of the MPC class using the constructor
  barrett_pendulum_control::LQRTracker lqr(pin_model_filepath, 0.008,
                                           pend_tip_pos_goal, init_position);
  std::cout << "After Class constructor\n";
  lqr.get_torques(0.0, Eigen::VectorXd::Zero(6));
  return 0;
}