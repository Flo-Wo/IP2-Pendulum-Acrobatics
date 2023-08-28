#ifndef FRICTION_MODEL_HPP
#define FRICTION_MODEL_HPP

#include <Eigen/Dense>

#include "iostream"

class FrictionModel {
 public:
  FrictionModel(const Eigen::Ref<const Eigen::VectorXd>& coulomb_friction,
                const Eigen::Ref<const Eigen::VectorXd>& damping,
                float coulomb_slope = 100.0)
      : coulomb_friction_(coulomb_friction),
        damping_(damping),
        activation_der_(),
        coulomb_slope_(coulomb_slope) {}

  // calculation methods
  Eigen::VectorXd calc(const Eigen::Ref<const Eigen::VectorXd>& v) const {
    return (-1) *
               (coulomb_friction_.array() * (coulomb_slope_ * v).array().tanh())
                   .matrix() -
           v.cwiseProduct(damping_);
  }

  Eigen::MatrixXd calcDiff(const Eigen::Ref<const Eigen::VectorXd>& v) const {
    this->activation_der_ =
        (coulomb_slope_ * (1 - (coulomb_slope_ * v).array().tanh().square()));
    return ((-1) * this->activation_der_.cwiseProduct(coulomb_friction_) -
            damping_)
        .asDiagonal();
  }

 private:
  Eigen::VectorXd coulomb_friction_;
  Eigen::VectorXd damping_;
  mutable Eigen::VectorXd activation_der_;
  float coulomb_slope_;
};
#endif  // FRICTION_MODEL_HPP