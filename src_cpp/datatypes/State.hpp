#ifndef STATES_HPP
#define STATES_HPP

#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

namespace pendulum_acrobatics {

struct State {
  State() : position(), velocity(), acceleration(), disturbance() {}
  State(int numDof)
      : position(numDof),
        velocity(numDof),
        acceleration(numDof),
        disturbance(numDof) {}
  State(Eigen::VectorXd position, Eigen::VectorXd velocity,
        Eigen::VectorXd acceleration, Eigen::VectorXd disturbance) {
    this->position = position;
    this->velocity = velocity;
    this->acceleration = acceleration;
    this->disturbance = disturbance;
  }
  State(Eigen::VectorXd position, Eigen::VectorXd velocity,
        Eigen::VectorXd acceleration) {
    this->position = position;
    this->velocity = velocity;
    this->acceleration = acceleration;
    this->disturbance = Eigen::VectorXd();
  }
  // copy constructor
  State(const State& other)
      : position(other.position),
        velocity(other.velocity),
        acceleration(other.acceleration),
        disturbance(other.disturbance) {}
  Eigen::VectorXd position;
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
  Eigen::VectorXd disturbance;
};

}  // namespace pendulum_acrobatics

#endif  // STATE