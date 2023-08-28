#include "LuenbergObserver.hpp"

#include "../file_reader/FileReader.hpp"

namespace pendulum_acrobatics {
Observer::Observer(const std::string relative_path_urdf) {
  boost::filesystem::path current_path = boost::filesystem::current_path();
  boost::filesystem::path full_path = current_path / relative_path_urdf;
  // pinocchio::Model pin_model_;
  std::cout << "Observer::Observer: file_path = " << full_path.string() << "\n";
  pinocchio::urdf::buildModel(full_path.string(), this->model_);
  this->data_ = new pinocchio::Data(this->model_);
  this->numDof_ = this->model_.nq;
  this->target_link_id = this->model_.getFrameId("links/pendulum/base",
                                                 pinocchio::FrameType::BODY);
  // we start with state buffer 1
  this->write_state_buffer_ = &this->state_buffer_1_;
};

int Observer::getNDOF() { return this->numDof_; }

State* Observer::get_active_buffer() {
  return (this->write_state_buffer_ == &this->state_buffer_1_)
             ? &this->state_buffer_2_
             : &this->state_buffer_1_;
};

void Observer::reset(const State& current, double time) {
  if (this->numDof_ == 0) {
    throw std::invalid_argument(
        "Observer::reset: Observer must be initialized before reset");
  }

  // position
  if (current.position.size() == this->numDof_) {
    this->state_buffer_1_.position = current.position;
    this->last_observed_.position = current.position;
  } else if (current.position.size() == 0) {
    this->state_buffer_1_.position = Eigen::VectorXd::Zero(this->numDof_);
    this->last_observed_.position = Eigen::VectorXd::Zero(this->numDof_);
  } else {
    throw std::invalid_argument(
        "Observer::reset: current.position must be the same size as the number "
        "of joints");
  }

  // velocity
  if (current.velocity.size() == this->numDof_) {
    this->state_buffer_1_.velocity = current.velocity;
    this->last_observed_.velocity = current.velocity;
  } else if (current.velocity.size() == 0) {
    this->state_buffer_1_.velocity = Eigen::VectorXd::Zero(this->numDof_);
    this->last_observed_.velocity = Eigen::VectorXd::Zero(this->numDof_);
  } else {
    throw std::invalid_argument(
        "Observer::reset: current.velocity must be empty or the same size as "
        "the number of joints");
  }

  // acceleration
  if (current.acceleration.size() == this->numDof_) {
    this->state_buffer_1_.acceleration = current.acceleration;
    this->last_observed_.acceleration = current.acceleration;
  } else if (current.acceleration.size() == 0) {
    this->state_buffer_1_.acceleration = Eigen::VectorXd::Zero(this->numDof_);
    this->last_observed_.acceleration = Eigen::VectorXd::Zero(this->numDof_);
  } else {
    throw std::invalid_argument(
        "Observer::reset: current.acceleration must be empty or the same size "
        "as the number of joints");
  }

  // disturbance
  if (current.disturbance.size() == this->numDof_) {
    this->state_buffer_1_.disturbance = current.disturbance;
    this->last_observed_.disturbance = current.disturbance;
  } else if (current.disturbance.size() == 0) {
    this->state_buffer_1_.disturbance = Eigen::VectorXd::Zero(this->numDof_);
    this->last_observed_.disturbance = Eigen::VectorXd::Zero(this->numDof_);
  } else {
    throw std::invalid_argument(
        "Observer::reset: disturbance must be empty or the same size as the "
        "number of joints");
  }

  // lastTorque
  this->lastTorque_ = Eigen::VectorXd::Zero(this->numDof_);

  // time
  this->lastTime_ = time;

  // update state buffer 2
  this->state_buffer_2_ = this->state_buffer_1_;

  // set reset flag
  this->reset_ = true;
  Logger::log(LogLevel::DEBUG, "Observer::reset reset finished");
};

// ----------------------------------------------------------------------------
// ---------- ExtendedStateLuenbergerObserver
// ----------------------------------------------
// ----------------------------------------------------------------------------

ExtendedStateLuenbergerObserver::ExtendedStateLuenbergerObserver(
    const std::string relative_path_urdf, const Eigen::VectorXd& Lp,
    const Eigen::VectorXd& Ld, const Eigen::VectorXd& Li,
    const Eigen::VectorXd& armature, const Eigen::VectorXd& damping,
    const Eigen::VectorXd& coulomb_friction)
    : Observer(relative_path_urdf), friction_model_(coulomb_friction, damping) {
  // Friction Model
  this->armature_ = armature;
  // this->friction_model_ = FrictionModel(coulomb_friction, damping);

  this->state_buffer_1_.disturbance = Eigen::VectorXd::Zero(this->numDof_);
  this->state_buffer_2_.disturbance = Eigen::VectorXd::Zero(this->numDof_);
  this->Lp_.diagonal() = Lp;
  this->Ld_.diagonal() = Ld;
  this->Li_.diagonal() = Li;
  this->initialized_ = true;
};

void ExtendedStateLuenbergerObserver::observe_wam(
    const double time, const Eigen::VectorXd& wam_joint_pos,
    const Eigen::VectorXd& lastTorque) {
  // assert that the observed state is valid
  if (wam_joint_pos.size() != 4) {
    throw std::invalid_argument(
        "LuenbergerObserver::observe_wam: observed.position must be the same "
        "size "
        "as the number of actuated joints");
  }
  this->observe(time, wam_joint_pos, lastTorque);
};

void ExtendedStateLuenbergerObserver::observe_optitrack(
    const double time, const Eigen::Vector3d& lsqt_orientation,
    const Eigen::VectorXd& lastTorque) {
  // Compute the joint angles given a 3D vector in space after least squares
  double r = std::sqrt(lsqt_orientation[1] * lsqt_orientation[1] +
                       lsqt_orientation[2] * lsqt_orientation[2]);
  double phi = std::atan2(lsqt_orientation[0], r);
  double theta = -std::atan2(lsqt_orientation[1], lsqt_orientation[2]);

  // we only have the positions, use the current internal state as observation
  this->observe(time, Eigen::Vector2d(theta, phi), lastTorque);
};
Eigen::Vector2d ExtendedStateLuenbergerObserver::optitrack_to_joint_pos(
    const Eigen::VectorXd& wam_pos, const Eigen::Vector3d& lsqt_orientation) {
  pinocchio::forwardKinematics(this->model_, *(this->data_), wam_pos);
  pinocchio::updateFramePlacements(this->model_, *(this->data_));
  // rotate the vector to the local frame
  const Eigen::MatrixXd rot_matrix_to_local_frame(
      this->data_->oMf[this->target_link_id].rotation());
  Eigen::Vector3d lsqt_orientation_local_frame =
      rot_matrix_to_local_frame.transpose() * lsqt_orientation;

  // Compute the joint angles given a 3D vector in space after least squares
  double r = std::sqrt(
      lsqt_orientation_local_frame[1] * lsqt_orientation_local_frame[1] +
      lsqt_orientation_local_frame[2] * lsqt_orientation_local_frame[2]);
  double phi = std::atan2(lsqt_orientation_local_frame[0], r);
  double theta = -std::atan2(lsqt_orientation_local_frame[1],
                             lsqt_orientation_local_frame[2]);

  return Eigen::Vector2d(theta, phi);
}

void ExtendedStateLuenbergerObserver::observe(
    const double time, const Eigen::VectorXd& partial_observation,
    const Eigen::VectorXd& lastTorque) {
  // assert reset flag
  if (!this->reset_) {
    throw std::runtime_error(
        "LuenbergerObserver::observe: Observer needs to be reset to an initial "
        "state before it can be used.");
  }

  // check control command
  if (lastTorque.size() != 0 && lastTorque.size() != this->numDof_) {
    throw std::invalid_argument(
        "LuenbergerObserver::observe: lastTorque must be empty or the same "
        "size as the number of joints");
  }
  // assume no torque if not given
  if (lastTorque.size() == this->numDof_) {
    this->lastTorque_ = lastTorque;
  } else {
    this->lastTorque_ = Eigen::VectorXd::Zero(this->numDof_);
  }
  Logger::log(LogLevel::DEBUG, "LuenObserver::observer set torques" +
                                   Logger::matrixToString(this->lastTorque_) +
                                   "\n");

  int n_substeps = 5;

  double time_delta_ = time - this->lastTime_;
  double dt = time_delta_ / n_substeps;

  Logger::log(LogLevel::DEBUG,
              "Time difference computation: " + std::to_string(dt) + ".\n");
  Eigen::MatrixXd M_inv(this->data_->M);

  pinocchio::computeAllTerms(this->model_, *(this->data_),
                             this->write_state_buffer_->position,
                             this->write_state_buffer_->velocity);
  this->data_->M.diagonal() += armature_;
  pinocchio::cholesky::decompose(this->model_, *(this->data_));
  M_inv.setZero();
  pinocchio::cholesky::computeMinv(this->model_, *(this->data_), M_inv);
  Eigen::VectorXd cg = pinocchio::nonLinearEffects(
      this->model_, *(this->data_), this->write_state_buffer_->position,
      this->write_state_buffer_->velocity);

  Logger::log(LogLevel::DEBUG, "Computed pinocchio terms.\n");

  this->write_state_buffer_->acceleration =
      M_inv * (this->lastTorque_ +
               this->friction_model_.calc(this->write_state_buffer_->velocity) -
               cg) +
      this->write_state_buffer_->disturbance;

  this->write_state_buffer_->velocity +=
      this->write_state_buffer_->acceleration * dt;

  this->write_state_buffer_->position +=
      this->write_state_buffer_->velocity * dt;

  for (int i = 0; i < n_substeps - 1; i++) {
    pinocchio::computeAllTerms(this->model_, *(this->data_),
                               this->write_state_buffer_->position,
                               this->write_state_buffer_->velocity);
    this->data_->M.diagonal() += armature_;
    pinocchio::cholesky::decompose(this->model_, *(this->data_));
    M_inv.setZero();
    pinocchio::cholesky::computeMinv(this->model_, *(this->data_), M_inv);
    Eigen::VectorXd cg = pinocchio::nonLinearEffects(
        this->model_, *(this->data_), this->write_state_buffer_->position,
        this->write_state_buffer_->velocity);

    this->write_state_buffer_->acceleration =
        M_inv *
            (this->lastTorque_ +
             this->friction_model_.calc(this->write_state_buffer_->velocity) -
             cg) +
        this->write_state_buffer_->disturbance;
    this->write_state_buffer_->velocity +=
        this->write_state_buffer_->acceleration * dt;
    this->write_state_buffer_->position +=
        this->write_state_buffer_->velocity * dt;
  }

  Eigen::VectorXd observed_position(6);
  if (partial_observation.size() == 4) {
    // wam observation
    Logger::log(LogLevel::DEBUG,
                "Luen::Observer: Observed WAM: " +
                    Logger::matrixToString(partial_observation));
    observed_position << partial_observation,
        this->write_state_buffer_->position.tail(2);
  } else if (partial_observation.size() == 2) {
    Logger::log(LogLevel::DEBUG,
                "Luen::Observer: Observed OPTITRACK: " +
                    Logger::matrixToString(partial_observation));
    // optitrack observation
    observed_position << this->write_state_buffer_->position.head(4),
        partial_observation;
  } else if (partial_observation.size() == 6) {
    observed_position << partial_observation;
    Logger::log(LogLevel::DEBUG,
                "Luen::Observer: Full observation: " +
                    Logger::matrixToString(partial_observation));
  } else {
    throw std::invalid_argument(
        "LuenObserver::observe: Invalid size for observation argument.");
  }
  Logger::log(LogLevel::DEBUG, "Observed position: " +
                                   Logger::matrixToString(observed_position) +
                                   ".\n");

  // full observation -> write to the active buffer
  Eigen::VectorXd error =
      observed_position - this->write_state_buffer_->position;
  Logger::log(LogLevel::DEBUG,
              "internal position: " +
                  Logger::matrixToString(this->write_state_buffer_->position) +
                  ".\n");
  Logger::log(LogLevel::DEBUG,
              "error: " + Logger::matrixToString(error) + ".\n");
  this->write_state_buffer_->position += this->Lp_ * error;
  this->write_state_buffer_->velocity += this->Ld_ * error;
  this->write_state_buffer_->disturbance += this->Li_ * error;

  // write state buffer contains newest updates

  // read and write state data from the active buffer
  State* read_state_buffer = this->get_active_buffer();

  // swap pointers
  this->write_state_buffer_ = read_state_buffer;
  // pointer status:
  // read -> updated => can read the correct status from now on
  // write -> non-updated

  // update internal values of state buffer again by copying it to ensure
  // correct updates via pinocchio
  read_state_buffer = this->get_active_buffer();
  *(this->write_state_buffer_) = *(read_state_buffer);

  // remember current time stamp
  this->lastTime_ = time;
};

double ExtendedStateLuenbergerObserver::get_state(State& predicted) {
  // read from inactive state buffer
  State* read_state_buffer_ = this->get_active_buffer();
  predicted = *read_state_buffer_;  // call copy constructor
  return this->lastTime_;
};

}  // namespace pendulum_acrobatics

int main(int argc, char* argv[]) {
  Logger::setLogLevel(LogLevel::INFO);
  if (argc > 2) {
    std::string arg = argv[2];
    if (arg == "DEBUG") {
      Logger::setLogLevel(LogLevel::DEBUG);
    }
  }
  // Initialize the observer with gains
  // std::string relative_path_urdf = "../src/wam/rot_wam_pend_0.374_temp.urdf";
  std::string relative_path_urdf = argv[1];
  Eigen::VectorXd Lp = Eigen::VectorXd::Constant(6, 0.25);
  Eigen::VectorXd Ld = Eigen::VectorXd::Constant(6, 5);
  Eigen::VectorXd Li = Eigen::VectorXd::Constant(6, 0.0);

  // friction model constants
  Eigen::VectorXd armature(6);  // Assuming 6 degrees of freedom
  armature << 0.111266, 0.053249, 0.0564972, 0.0182617, 0., 0.;
  Eigen::VectorXd damping(6);  // Assuming 6 degrees of freedom
  damping << 1.31174e+00, 4.92821e-01, 1.48827e+00, 1.50353e-01, 1.00000e-03,
      1.00000e-03;
  Eigen::VectorXd coulomb_friction(6);  // Assuming 6 degrees of freedom
  coulomb_friction << 1.70477, 1.43072, 0.685072, 0.854358, 0., 0.;

  pendulum_acrobatics::ExtendedStateLuenbergerObserver observer(
      relative_path_urdf, Lp, Ld, Li, armature, damping, coulomb_friction);

  // init pos, zero velocity and zero initial torques
  Eigen::VectorXd q(6);
  // q << 0., -0.78, 0., 2.37, 0., 0.;
  q << 0., -1.3, 0., 1.3, 0., 0.;
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(6, 0.0);
  pendulum_acrobatics::State reset_state(q, dq,
                                         Eigen::VectorXd::Constant(6, 0));

  observer.reset(reset_state, 0.0);

  // read sample data and observe it
  boost::filesystem::path current_path = boost::filesystem::current_path();
  boost::filesystem::path states_path =
      current_path / "state_observer/test_data/test_observer_states.txt";
  boost::filesystem::path torques_path =
      current_path / "state_observer/test_data/test_observer_torques.txt";
  std::vector<Eigen::VectorXd> states_list =
      pendulum_acrobatics::FileReader::readNumpyTxt(states_path.string());
  std::vector<Eigen::VectorXd> torques_list =
      pendulum_acrobatics::FileReader::readNumpyTxt(torques_path.string());

  double time = 0.002;
  Eigen::VectorXd partial_observation;
  pendulum_acrobatics::State internal_state;
  const Eigen::IOFormat fmt(9, Eigen::DontAlignCols, "\t", " ", "", "", "", "");
  for (int time_idx = 0; time_idx < 100; ++time_idx) {
    if (time_idx % 4 == 0) {
      // full observation
      partial_observation.resize(6);
      partial_observation.setZero();
      partial_observation << states_list[time_idx].head(6);
    } else {
      // only wam observation
      partial_observation.resize(4);
      partial_observation.setZero();
      partial_observation << states_list[time_idx].head(4);
    }
    Eigen::VectorXd torque = Eigen::VectorXd::Constant(6, 0.0);
    torque.head(4) = torques_list[time_idx].head(4);
    std::cout << "\n\ntime idx = " << time_idx << "\n";
    // std::cout << "tau = " << torque << "\n";
    // std::cout << "observation = " << partial_observation << "\n";
    observer.observe(time, partial_observation, torque);
    double int_time = observer.get_state(internal_state);
    std::cout << "Internal position: " << internal_state.position.format(fmt)
              << "\n";
    std::cout << "Internal velocity: " << internal_state.velocity.format(fmt)
              << "\n";
    time += 0.002;
  }

  Eigen::Vector3d test_orientation;
  test_orientation << 0, 1, 1;
  observer.optitrack_to_joint_pos(q, test_orientation);
  // Get the internal state
  double int_time = observer.get_state(internal_state);

  // Print the state value

  std::cout << "Final results: \n\n";
  std::cout << "Internal position: " << internal_state.position.format(fmt)
            << std::endl;
  std::cout << "Internal velocity: " << internal_state.velocity.format(fmt)
            << std::endl;
  std::cout << "worked";
  return 0;
}