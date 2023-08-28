
#include "MPC.hpp"

// testing
// #include <gperftools/profiler.h>

#include "crocoddyl/core/utils/timer.hpp"
void print_results(const std::vector<Eigen::VectorXd>& vectors) {
  for (const auto& vector : vectors) {
    for (int i = 0; i < vector.size(); ++i) {
      std::cout << vector[i];
      if (i < vector.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
}

namespace pendulum_acrobatics {
MPC::MPC(const std::string& pin_model_filepath,
         std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
         std::vector<float>& angleGoal, const Eigen::VectorXd& init_position,
         const Eigen::VectorXd& armature, const Eigen::VectorXd& damping,
         const Eigen::VectorXd& coulomb_friction,
         const Eigen::VectorXd& grav_comp_idxs, int factor_integration_time,
         int solver_max_iter, int mpc_horizon, float integration_time_500Hz)
    : solver_max_iter_(solver_max_iter),
      armature_(armature),
      damping_(damping),
      coulomb_friction_(coulomb_friction),
      grav_comp_idxs_(grav_comp_idxs),
      pin_model_(),
      mpc_horizon_(mpc_horizon),
      factor_integration_time_(factor_integration_time),
      integration_time_(integration_time_500Hz),
      warmstart_xs_(mpc_horizon, Eigen::VectorXd::Zero(12)),
      torques_with_time_(mpc_horizon, factor_integration_time,
                         integration_time_500Hz) {
  // make path relative to the file location
  boost::filesystem::path currentPath_new = boost::filesystem::current_path();
  boost::filesystem::path urdfFilePath_new =
      currentPath_new / pin_model_filepath;
  // pinocchio::Model pin_model_;
  Logger::log(LogLevel::INFO, "Using " + urdfFilePath_new.string() + "\n");
  pinocchio::urdf::buildModel(urdfFilePath_new.string(), pin_model_);
  // this->pin_model_ = pin_model_;
  Logger::log(LogLevel::INFO, "Pinocchio Model built successfully.\n");
  Logger::log(LogLevel::DEBUG,
              "NQ: " + std::to_string(this->pin_model_.nq) + ".\n");
  // pinocchio::Data
  // this->pin_data_ = pinocchio::Data(this->pin_model_);
  this->pin_data_ = new pinocchio::Data(this->pin_model_);
  // this->pin_data_ = pin_data_;
  Logger::log(LogLevel::INFO, "Pinocchio Data built successfully.\n");

  this->terminal_model_ =
      this->setUpCostModels(pend_tip_pos_goal, angleGoal, init_position);
  this->setUpActiveCostModels();
  Logger::log(LogLevel::INFO, "Set up costs succesfully.\n");
  this->setUpSolver(init_position);
  Logger::log(LogLevel::INFO, "Set up solver succesfully.\n");
};

int MPC::get_mpc_horizon() { return this->mpc_horizon_; }

int MPC::get_integration_factor() { return this->factor_integration_time_; }

// load the torques with the closest time difference w.r.t. current time
const Eigen::VectorXd& MPC::get_torque_to_curr_time(double curr_time) {
  return this->torques_with_time_.get_current_torque(curr_time).getArray();
};

const std::vector<Eigen::VectorXd>& MPC::get_xs() {
  return this->solver->get_xs();
}
const std::vector<Eigen::VectorXd>& MPC::get_us() {
  return this->solver->get_us();
}
void MPC::solve(Eigen::VectorXd state, double time, int time_idx,
                std::vector<Eigen::VectorXd>& xs,
                std::vector<Eigen::VectorXd>& us) {
  Logger::log(LogLevel::DEBUG, "MPC::solve: load active cost models.\n");
  this->update_active_cost_models(time_idx);

  // Logger::log(LogLevel::DEBUG, "MPC::solve: Append active cost models.\n");
  // this->append_active_cost_models();
  // Logger::log(LogLevel::DEBUG,
  //             "MPC::solve: WORKED append active cost models.\n");

  this->problem->set_x0(state);
  Logger::log(LogLevel::DEBUG, "MPC::solve: WORKED set x0.\n");

  // provide space to save the results
  // TODO: preallocate memory?
  std::size_t T = problem->get_T();
  if (!(xs.size() == T + 1 && us.size() == T)) {
    std::cerr << "MPC::solve: xs or us has the wrong shape.\n";
  }
  Logger::log(LogLevel::DEBUG, "MPC::solve: WORKED get T.\n");

  // Eigen::VectorXd warmstart_state(state);
  // warmstart_state << state.head(6), Eigen::VectorXd::Zero(6);

  // fill the warmstart solutions
  std::fill(this->warmstart_xs_.begin(), this->warmstart_xs_.end(), state);

  this->problem->quasiStatic(us, this->warmstart_xs_);
  this->solver->setCandidate(xs, us, true);
  Logger::log(LogLevel::DEBUG, "MPC::solve: WORKED set candidate solution.\n");

  // Solve the problem
  bool success = this->solver->solve(xs, us, this->solver_max_iter_, true);
  Logger::log(LogLevel::DEBUG, "MPC::solve: WORKED solve call.\n");
  this->torques_with_time_.update_solution_torques(time,
                                                   this->solver->get_us());
};

void MPC::update_active_cost_models(int time_idx) {
  // use updateNode

  int highest_index_needed =
      time_idx + this->factor_integration_time_ * this->mpc_horizon_;

  if (highest_index_needed > this->running_models_.size()) {
    std::cout << "TIME INDEX EXCEEDS THE NUMBER OF COST MODEL"
              << "\n";
    return;
  }

  for (std::size_t i = 0, idx = time_idx; i < this->mpc_horizon_;
       ++i, idx += this->factor_integration_time_) {
    if (!this->running_models_[idx]) {
      std::cerr << "\n\n MPC::update_active_cost_models: null pointer \n\n";
    } else {
      Logger::log(LogLevel::DEBUG, "adding cost model: " + std::to_string(idx) +
                                       " at index: " + std::to_string(i) +
                                       "\n");
      this->problem->updateModel(i, this->running_models_[idx]);
    }
  }
  Logger::log(LogLevel::DEBUG,
              "load_active_cost_models:: WORKED: loading cost models\n");
  this->last_time_idx_ = time_idx;
};

void MPC::append_active_cost_models(int time_idx) {
  // use circular append

  Logger::log(LogLevel::DEBUG, "append_active_cost_models:: STARTING\n");
  if (this->active_cost_models_.empty()) {
    Logger::log(LogLevel::DEBUG,
                "append_active_cost_models:: active_cost_models_ is empty\n");
  }
  Logger::log(
      LogLevel::DEBUG,
      "length: " + std::to_string(this->active_cost_models_.size()) + "\n");
  for (const boost::shared_ptr<crocoddyl::ActionModelAbstract>& cost_model :
       this->active_cost_models_) {
    if (cost_model) {
      this->problem->circularAppend(cost_model);
    } else {
      Logger::log(LogLevel::ERROR,
                  "MPC::append_active_cost_models: encountered a null shared "
                  "pointer.\n");
    }
  }
  Logger::log(LogLevel::DEBUG, "append_active_cost_models:: worked");
};

void MPC::load_active_cost_models(int time_idx) {
  Logger::log(LogLevel::DEBUG,
              "len(running_models): " +
                  std::to_string(this->running_models_.size()) + "\n");
  for (std::size_t i = 0, idx = time_idx; i < this->mpc_horizon_;
       ++i, idx += this->factor_integration_time_) {
    Logger::log(LogLevel::DEBUG,
                "MPC::load_active_cost_models:: adding cost model: " +
                    std::to_string(idx) + " at index: " + std::to_string(i) +
                    "\n");
    this->active_cost_models_[i] = this->running_models_[idx];
    if (!this->active_cost_models_[i]) {
      std::cerr << "\n\n null pointer \n\n";
    }
  }
  Logger::log(LogLevel::DEBUG,
              "load_active_cost_models:: WORKED: loading cost models\n");
};

void MPC::setUpActiveCostModels() {
  this->active_cost_models_ =
      std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>(
          this->mpc_horizon_);
}

void MPC::setUpSolver(const Eigen::VectorXd& init_pos) {
  Logger::log(LogLevel::DEBUG, "MPC::setUpSolver: Running Setup Solver.\n");
  // define the init position
  Eigen::VectorXd x0(init_pos.size() + init_pos.size());
  x0 << init_pos, Eigen::VectorXd::Zero(init_pos.size()).cast<double>();

  Logger::log(LogLevel::DEBUG,
              "MPC::setUpSolver: Loading active cost models.\n");
  this->load_active_cost_models(0);
  Logger::log(LogLevel::DEBUG,
              "MPC::setUpSolver: Worked loading active cost models.\n");

  Logger::log(LogLevel::DEBUG,
              "MPC::setUpSolver: Building shooting problem.\n");
  if (this->terminal_model_) {
    Logger::log(LogLevel::DEBUG, "MPC::setUpSolver: terminal model is valid\n");
  }
  if (!this->active_cost_models_.empty()) {
    Logger::log(LogLevel::DEBUG,
                "MPC::setUpSolver: active cost models are valid\n");
  }
  // build problem and solver
  this->problem = boost::make_shared<crocoddyl::ShootingProblem>(
      x0, this->active_cost_models_, terminal_model_);

  Logger::log(LogLevel::DEBUG, "MPC::setUpSolver: Building Solver.\n");

  this->solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem);

  // CALLBACKS
  // std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
  // cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
  // this->solver->setCallbacks(cbs);

  Logger::log(LogLevel::DEBUG, "MPC::setUpSolver: WORKED Setup solver.\n");
};

boost::shared_ptr<crocoddyl::ActionModelAbstract> MPC::setUpCostModels(
    std::vector<Eigen::VectorXd>& pend_tip_pos_goal,
    std::vector<float>& angleGoal, const Eigen::VectorXd& init_pos) {
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

  // ==== INIT POSITION AND OFFSET ====
  Logger::log(LogLevel::DEBUG, "Computing the initial position.\n");
  // make shared pointers
  pinocchio::forwardKinematics(this->pin_model_, *(this->pin_data_), init_pos);
  pinocchio::updateFramePlacements(this->pin_model_, *(this->pin_data_));

  const Eigen::VectorXd target_offset(
      this->pin_data_->oMf[target_link_id].translation());

  std::size_t nu = this->pin_model_.nv;

  // define the init position
  Eigen::VectorXd x0(init_pos.size() + init_pos.size());
  x0 << init_pos, Eigen::VectorXd::Zero(init_pos.size()).cast<double>();

  Eigen::VectorXd vel_pen_target(init_pos.size() + init_pos.size());
  vel_pen_target << Eigen::VectorXd::Zero(init_pos.size() + init_pos.size());
  Eigen::VectorXd vel_pen_weights(init_pos.size() + init_pos.size());
  vel_pen_weights << Eigen::VectorXd::Zero(init_pos.size()),
      Eigen::VectorXd::Ones(init_pos.size());
  Logger::log(LogLevel::DEBUG,
              "Computation of initial position and weighting worked. \n");

  // ==== COST MODELS ====
  // Costs that are shared between all time-steps

  Logger::log(LogLevel::DEBUG, "Building shared cost models. \n");
  // STATE penalty
  boost::shared_ptr<crocoddyl::CostModelAbstract> qRegCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelState>(
                     state, x0, actuation->get_nu()));
  // VELOCITY penalty
  boost::shared_ptr<crocoddyl::CostModelAbstract> vRegCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(
              vel_pen_weights),
          boost::make_shared<crocoddyl::ResidualModelState>(
              state, vel_pen_target, actuation->get_nu()));
  // CONTROL penalty
  boost::shared_ptr<crocoddyl::CostModelAbstract> uRegCost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelControl>(
                     state, actuation->get_nu()));

  // TODO: needed?

  // Create a bound constraint that restricts the feasible space
  // Eigen::VectorXd lb = this->pin_model_.lowerPositionLimit.array() + 0.1;
  // Eigen::VectorXd ub = this->pin_model_.upperPositionLimit.array() - 0.1;
  // Eigen::VectorXd velocityLimit = this->pin_model_.velocityLimit;

  // Concatenate the position and velocity limits
  // Eigen::VectorXd lb_all(this->pin_model_.nv + this->pin_model_.nv);
  // lb_all << lb, -velocityLimit;
  // Eigen::VectorXd ub_all(this->pin_model_.nv + this->pin_model_.nv);
  // ub_all << ub, velocityLimit;

  // TODO: shared pointers?
  // crocoddyl::ActivationBounds bounds(lb_all, ub_all, 1.0);

  // crocoddyl::ResidualModelState xLimitResidual(state, state.zero(), nu);
  // crocoddyl::ActivationModelQuadraticBarrier xLimitActivation(bounds);
  // crocoddyl::CostModelResidual limitCost(state, xLimitActivation,
  //                                        xLimitResidual);

  Logger::log(LogLevel::DEBUG, "WORKED: Building shared cost models. \n");
  // TIME DEPENDENT COST MODELS
  std::size_t traj_size = pend_tip_pos_goal.size();

  Eigen::VectorXd rotation_weighted_barrier(3);
  rotation_weighted_barrier << 1.0, 1.0, 0.0;

  Logger::log(LogLevel::DEBUG, "Starting Loop. \n");
  for (int time_idx = 0; time_idx < traj_size; ++time_idx) {
    Logger::log(LogLevel::DEBUG,
                "Loop index: " + std::to_string(time_idx) + "\n");
    // positional error of the pendulum
    Eigen::VectorXd target_pos(pend_tip_pos_goal[time_idx] + target_offset);

    boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
        boost::make_shared<crocoddyl::CostModelResidual>(
            state, boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                       state, target_link_id, target_pos, actuation->get_nu()));

    Logger::log(LogLevel::DEBUG, "WORKED: Translational costs \n");
    // rotational error of the pendulum
    float curr_angle(angleGoal[time_idx]);
    Eigen::VectorXd angle_lower_bound(3);
    angle_lower_bound << -1.0 * curr_angle, -1.0 * curr_angle, 2.0 * M_PI;

    Eigen::VectorXd angle_upper_bound(3);
    angle_upper_bound << curr_angle, curr_angle, 2.0 * M_PI;

    crocoddyl::ActivationBounds rotation_activation_bounds(angle_lower_bound,
                                                           angle_upper_bound);
    Logger::log(LogLevel::DEBUG, "Rotational activation bounds \n");
    boost::shared_ptr<crocoddyl::CostModelAbstract> goalRotationCost =
        boost::make_shared<crocoddyl::CostModelResidual>(
            state,
            boost::make_shared<
                crocoddyl::ActivationModelWeightedQuadraticBarrier>(
                rotation_activation_bounds, rotation_weighted_barrier),
            boost::make_shared<crocoddyl::ResidualModelFrameRotation>(
                state, target_link_id,
                Eigen::MatrixXd::Identity(3, 3).cast<double>(),
                actuation->get_nu()));

    Logger::log(LogLevel::DEBUG, "WORKED: Rotational activation costs\n");
    // add all cost models to the running cost model
    boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel =
        boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

    runningCostModel->addCost("pendulum_pos_costs", goalTrackingCost, 6e5);
    runningCostModel->addCost("pendulum_rot_costs", goalRotationCost, 1e6);
    runningCostModel->addCost("xReg", qRegCost, 1e2);
    runningCostModel->addCost("uReg", uRegCost, 1e-1);
    runningCostModel->addCost("vReg", vRegCost, 1e0);

    Logger::log(LogLevel::DEBUG, "WORKED: cost model sum\n");
    // integrate the cost models
    boost::shared_ptr<DifferentialActionModelFrictionFwdDynamics> runningDAM =
        boost::make_shared<DifferentialActionModelFrictionFwdDynamics>(
            state, actuation, runningCostModel, this->armature_, this->damping_,
            this->coulomb_friction_);
    // , this->grav_comp_idxs_);
    Logger::log(LogLevel::DEBUG, "WORKED: action model constructor\n");
    // integrate the cost models

    boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModelEuler =
        boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
            runningDAM,
            // integration time is factored
            this->factor_integration_time_ * this->integration_time_);
    Logger::log(LogLevel::DEBUG, "WORKED: integration\n");

    runningModelEuler->set_u_lb((-1.0) * this->pin_model_.effortLimit);
    runningModelEuler->set_u_ub(this->pin_model_.effortLimit);

    running_models_.push_back(runningModelEuler);
    Logger::log(LogLevel::DEBUG, "WORKED: push back\n");
  }

  boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel =
      boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  // terminalCostModel->addCost("xReg", qRegCost, 1e2);
  // terminalCostModel->addCost("uReg", uRegCost, 1e-1);
  // terminalCostModel->addCost("vReg", vRegCost, 1e0);
  // integrate the cost models
  boost::shared_ptr<DifferentialActionModelFrictionFwdDynamics> terminalDAM =
      boost::make_shared<DifferentialActionModelFrictionFwdDynamics>(
          state, actuation, terminalCostModel, this->armature_, this->damping_,
          this->coulomb_friction_);
  //, this->grav_comp_idxs_);

  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminalModelEuler =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
          terminalDAM,
          // integration time is zero
          0.0);

  terminalModelEuler->set_u_lb((-1.0) * this->pin_model_.effortLimit);
  terminalModelEuler->set_u_ub(this->pin_model_.effortLimit);
  Logger::log(LogLevel::DEBUG,
              "Init pos size: " + std::to_string(init_pos.size()) + "\n");
  return terminalModelEuler;
};
}  // namespace pendulum_acrobatics

// ====================
// MAIN METHOD
// ====================

int main(int argc, char* argv[]) {
  // Parse command line arguments to set the log level
  Logger::setLogLevel(LogLevel::INFO);
  if (argc > 2) {
    std::string arg = argv[2];
    if (arg == "DEBUG") {
      Logger::setLogLevel(LogLevel::DEBUG);
    }
  }

  // Create the logger with the specified log level
  std::cout << "Starting main\n";
  // Define the required data for the MPC constructor
  std::string pin_model_filepath = argv[1];

  // Vector of 3D test vectors
  std::vector<Eigen::VectorXd> pend_tip_pos_goal;
  int num_cost_models = 3000;

  for (int i = 0; i < num_cost_models; ++i) {
    // Creating a 3D test vector (replace these values with your desired test
    // data)
    Eigen::VectorXd test_vector(3);
    // test_vector << i * 0.1, i * 0.2, i * 0.3;
    test_vector.setZero();
    pend_tip_pos_goal.push_back(test_vector);
  }

  // Vector of 1D test angles
  std::vector<float> angleGoal;

  for (int i = 0; i < num_cost_models; ++i) {
    // Creating a 1D test angle (replace these values with your desired test
    // data)
    float test_angle = 0.0;
    angleGoal.push_back(test_angle);
  }

  Eigen::VectorXd init_position(6);  // Add test data as needed
  init_position << 0.0, -1.3, 0.0, 1.3, 0.0, 0.0;

  // friction model constants
  Eigen::VectorXd armature(6);  // Assuming 6 degrees of freedom
  armature << 0.111266, 0.053249, 0.0564972, 0.0182617, 0., 0.;
  Eigen::VectorXd damping(6);  // Assuming 6 degrees of freedom
  damping << 1.31174e+00, 4.92821e-01, 1.48827e+00, 1.50353e-01, 1.00000e-03,
      1.00000e-03;
  Eigen::VectorXd coulomb_friction(6);  // Assuming 6 degrees of freedom
  // coulomb_friction << 1.70477, 1.43072, 0.685072, 0.854358, 0., 0.;
  coulomb_friction.setZero();

  Eigen::VectorXd grav_comp_idxs = Eigen::VectorXd();
  // grav_comp_idxs << 1, 2, 3, 4;

  int factor_integration_time = 1;
  int solver_max_iter = 20;
  int mpc_horizon = 20;
  float integration_time_500Hz = 0.002;

  std::cout << "Before Class constructor\n";
  // Create an instance of the MPC class using the constructor
  pendulum_acrobatics::MPC mpc_node(
      pin_model_filepath, pend_tip_pos_goal, angleGoal, init_position, armature,
      damping, coulomb_friction, grav_comp_idxs, factor_integration_time,
      solver_max_iter, mpc_horizon, integration_time_500Hz);

  std::cout << "After Class constructor\n";

  // READ test data for example states
  // read sample data and observe it
  boost::filesystem::path current_path = boost::filesystem::current_path();
  boost::filesystem::path states_path =
      current_path / "state_observer/test_data/test_observer_states.txt";
  std::vector<Eigen::VectorXd> states_list =
      pendulum_acrobatics::FileReader::readNumpyTxt(states_path.string());

  // Example: Call the solve() function with test data
  Eigen::VectorXd state(12);  // Add test data as needed
  state.setZero();            // Add test data as needed
  std::vector<Eigen::VectorXd> xs(mpc_node.get_mpc_horizon() + 1, state);
  std::vector<Eigen::VectorXd> us(mpc_node.get_mpc_horizon(),
                                  Eigen::VectorXd::Zero(6));
  float current_time;

  // compute the average computation time
  Eigen::ArrayXd duration(100);
  std::cout << "Starting Benchmark test: \n";
  // ProfilerStart("mpc_profiler.prof");
  for (int time_idx = 0, i = 0; time_idx < 1000; time_idx += 10, i += 1) {
    // std::cout << "\n\nTime IDX: " << time_idx << "\n";
    state = states_list[time_idx];
    // std::cout << "input state = \n" << state << "\n";
    std::vector<Eigen::VectorXd> xs(mpc_node.get_mpc_horizon() + 1, state);
    std::vector<Eigen::VectorXd> us(mpc_node.get_mpc_horizon(),
                                    Eigen::VectorXd::Zero(6));
    crocoddyl::Timer timer;
    mpc_node.solve(state, 0.002 * time_idx, time_idx, xs, us);
    duration[i] = timer.get_duration();
    // std::cout << "Solver results: \n";
    // print_results(mpc_node.get_us());
  }
  // ProfilerStop();
  double avrg_duration = duration.sum() / 100;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration
            << "-" << max_duration << ")" << std::endl;

  // // Example: Call the solve() function with test data
  // Eigen::VectorXd state(12);  // Add test data as needed
  // state.setZero();            // Add test data as needed
  // float current_time = 0.0;   // Add test data as needed
  // std::vector<Eigen::VectorXd> xs(mpc_node.get_mpc_horizon() + 1, state);
  // // TODO: maybe not hardcode the 6
  // std::vector<Eigen::VectorXd> us(mpc_node.get_mpc_horizon(),
  // Eigen::VectorXd::Zero(6));
  // mpc_node.solve(state, current_time, xs, us);
  std::cout << "After solver call\n";

  return 0;
}