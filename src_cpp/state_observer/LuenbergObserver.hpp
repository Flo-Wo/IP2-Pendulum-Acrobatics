#include <iostream>

// own module
#include "../action_model/FrictionModel.hpp"
#include "../datatypes/State.hpp"
#include "../logger/Logger.hpp"
// pinocchio
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"

// compute dynamics
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
// filesystem
#include <boost/filesystem.hpp>

namespace pendulum_acrobatics {

class Observer {
 public:
  Observer(const std::string relative_path_urdf);
  void reset(const State& current, double time);

  int getNDOF();
  State* get_active_buffer();
  virtual void observe(const double time,
                       const Eigen::VectorXd& partial_observation,
                       const Eigen::VectorXd& lastTorque) = 0;
  virtual double get_state(State& internal_state) = 0;

 protected:
  bool initialized_ = false;
  bool reset_ = false;

  pinocchio::Model model_;
  pinocchio::Data* data_;

  double numDof_;
  std::vector<std::string> jointNames_;

  State state_buffer_1_;
  State state_buffer_2_;
  State* write_state_buffer_;
  State last_observed_;
  Eigen::VectorXd lastTorque_;
  double lastTime_;
  pinocchio::FrameIndex target_link_id;
};

class ExtendedStateLuenbergerObserver : public Observer {
  using Observer::Observer;

 public:
  ExtendedStateLuenbergerObserver(const std::string relative_path_urdf,
                                  const Eigen::VectorXd& Lp,
                                  const Eigen::VectorXd& Ld,
                                  const Eigen::VectorXd& Li,
                                  const Eigen::VectorXd& armature,
                                  const Eigen::VectorXd& damping,
                                  const Eigen::VectorXd& coulomb_friction);
  void observe_wam(const double time, const Eigen::VectorXd& wam_joint_pos,
                   const Eigen::VectorXd& lastTorque);
  void observe_optitrack(const double time,
                         const Eigen::Vector3d& lsqt_orientation,
                         const Eigen::VectorXd& lastTorque);

  Eigen::Vector2d optitrack_to_joint_pos(
      const Eigen::VectorXd& wam_pos, const Eigen::Vector3d& lsqt_orientation);
  void observe(const double time, const Eigen::VectorXd& partial_observation,
               const Eigen::VectorXd& lastTorque);
  double get_state(State& internal_state);

 private:
  FrictionModel friction_model_;
  Eigen::VectorXd armature_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lp_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Ld_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Li_;
};
}  // namespace pendulum_acrobatics