#include "ForwardIntegrator.hpp"

#include <iostream>

namespace pendulum_acrobatics {
ForwardIntegrator::ForwardIntegrator(const std::string relative_path_urdf,
                                     const Eigen::VectorXd& armature,
                                     const Eigen::VectorXd& damping,
                                     const Eigen::VectorXd& coulomb_friction,
                                     double integration_time_step)
    : armature_(armature),
      integration_time_step_(integration_time_step),
      friction_model_(coulomb_friction, damping)

{
  boost::filesystem::path current_path = boost::filesystem::current_path();
  boost::filesystem::path full_path = current_path / relative_path_urdf;

  Logger::log(LogLevel::DEBUG, "ForwardIntegrator::ForwardIntegrator, using: " +
                                   full_path.string() + "\n");
  pinocchio::urdf::buildModel(full_path.string(), this->pin_model_);
  this->pin_data_ = new pinocchio::Data(this->pin_model_);
};

double ForwardIntegrator::semi_implicit_euler(
    Eigen::VectorXd q, Eigen::VectorXd dq, TorquesWithTime& torques_with_time,
    int num_of_500Hz_steps_to_integrate, double curr_time,
    Eigen::VectorXd& return_state) {
  for (int idx = 0; idx < num_of_500Hz_steps_to_integrate; ++idx) {
    pinocchio::computeAllTerms(this->pin_model_, *(this->pin_data_), q, dq);

    // add armature to the diagonal mass matrix
    this->pin_data_->M.diagonal() += armature_;
    Logger::log(LogLevel::DEBUG,
                "M = " + Logger::matrixToString(this->pin_data_->M) + "\n");

    pinocchio::cholesky::decompose(this->pin_model_, *(this->pin_data_));

    // set M_inverse to zero
    Eigen::MatrixXd M_inv(this->pin_data_->M);
    M_inv.setZero();

    pinocchio::cholesky::computeMinv(this->pin_model_, *(this->pin_data_),
                                     M_inv);
    Logger::log(LogLevel::DEBUG,
                "M^-1 = " + Logger::matrixToString(M_inv) + "\n");

    Eigen::VectorXd cg = pinocchio::nonLinearEffects(this->pin_model_,
                                                     *(this->pin_data_), q, dq);
    Logger::log(LogLevel::DEBUG, "nle = " + Logger::matrixToString(cg) + "\n");

    ArrayWithDuration tau = torques_with_time.get_current_torque(curr_time);
    Logger::log(LogLevel::DEBUG,
                "tau = " + Logger::matrixToString(tau.getArray()) +
                    " with time: " + std::to_string(tau.getTimestamp()) + "\n");
    Logger::log(
        LogLevel::DEBUG,
        "friction = " + Logger::matrixToString(this->friction_model_.calc(dq)) +
            "\n");
    Eigen::VectorXd ddq =
        M_inv * (tau.getArray() + this->friction_model_.calc(dq) - cg);
    Logger::log(LogLevel::DEBUG, "ddq = " + Logger::matrixToString(ddq) + "\n");

    // semi implicit euler method
    dq += ddq * integration_time_step_;
    curr_time += integration_time_step_;
    Eigen::VectorXd q_next(q);
    pinocchio::integrate(this->pin_model_, q, dq * integration_time_step_,
                         q_next);
    Logger::log(LogLevel::DEBUG,
                "ForwardIntegrator::semi_implicit_euler, q_next= " +
                    Logger::matrixToString(q_next) + "\n");

    q = q_next;
  }
  // Eigen::VectorXd state(q.size() + dq.size());
  return_state << q, dq;
  return curr_time;
};

};  // namespace pendulum_acrobatics
int main(int argc, char* argv[]) {
  // Example input data
  // std::string relative_path_urdf = "path/to/urdf/file.urdf";

  // length has to be 0.374
  std::string relative_path_urdf = argv[1];
  // Parse command line arguments to set the log level
  Logger::setLogLevel(LogLevel::INFO);
  if (argc > 2) {
    std::string arg = argv[2];
    if (arg == "DEBUG") {
      Logger::setLogLevel(LogLevel::DEBUG);
    }
  }

  /*
  Coulomb:  [1.70477  1.43072  0.685072 0.854358 0.       0.      ]
  Viscous:
  [1.31174e+00 4.92821e-01 1.48827e+00 1.50353e-01 1.00000e-03 1.00000e-03]
  Armature:  [0.111266  0.053249  0.0564972 0.0182617 0.        0.       ]
  */
  Eigen::VectorXd armature(6);  // Assuming 6 degrees of freedom
  armature << 0.111266, 0.053249, 0.0564972, 0.0182617, 0., 0.;
  Eigen::VectorXd damping(6);  // Assuming 6 degrees of freedom
  damping << 1.31174e+00, 4.92821e-01, 1.48827e+00, 1.50353e-01, 1.00000e-03,
      1.00000e-03;
  Eigen::VectorXd coulomb_friction(6);  // Assuming 6 degrees of freedom
  coulomb_friction << 1.70477, 1.43072, 0.685072, 0.854358, 0., 0.;

  double integration_time_step = 0.002;  // Time step for integration

  // Create the ForwardIntegrator object
  pendulum_acrobatics::ForwardIntegrator integrator(
      relative_path_urdf, armature, damping, coulomb_friction,
      integration_time_step);

  // Example initial joint position and velocity
  Eigen::VectorXd q(6);  // Assuming 6 degrees of freedom
  q << 0., -0.78, 0., 2.37, 0., 0.;
  Eigen::VectorXd dq =
      Eigen::VectorXd::Constant(6, 0.0);  // Assuming 6 degrees of freedom
  // Eigen::VectorXd dq(6);  // Assuming 6 degrees of freedom
  // dq << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  int num_of_500Hz_steps_to_integrate(10);
  int factor_integration_time(2);
  double curr_time(0.0);

  // Example torques with time
  std::vector<Eigen::VectorXd> torque_list;
  for (int i = 0; i < num_of_500Hz_steps_to_integrate; ++i) {
    torque_list.push_back(
        Eigen::VectorXd::Constant(6, i * 0.02));  // Set all torques to 10.0
  }
  pendulum_acrobatics::TorquesWithTime torques_with_time(
      curr_time, torque_list, factor_integration_time, integration_time_step);

  // Initialize return_state vector
  Eigen::VectorXd return_state(12);  // Assuming 6 degrees of freedom

  std::cout << "calling semi-implicit Euler\n";
  // Call the semi_implicit_euler method
  double solver_time = integrator.semi_implicit_euler(
      q, dq, torques_with_time, num_of_500Hz_steps_to_integrate, curr_time,
      return_state);

  // Print the final joint positions after integration
  std::cout << "Final joint positions:" << std::endl;
  std::cout << return_state << std::endl;

  std::cout << "Final solver time:" << std::endl;
  std::cout << solver_time << std::endl;
  return 0;
}
