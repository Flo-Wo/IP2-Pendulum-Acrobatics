#ifndef ACTION_MODEL_HPP
#define ACTION_MODEL_HPP

#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/actuation-base.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/data-collector-base.hpp>
#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "../logger/Logger.hpp"
#include "FrictionModel.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"

namespace pendulum_acrobatics {

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorXs;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

class DifferentialActionModelFrictionFwdDynamics
    : public crocoddyl::DifferentialActionModelAbstract {
 public:
  DifferentialActionModelFrictionFwdDynamics(
      boost::shared_ptr<crocoddyl::StateMultibody> state,
      boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation,
      boost::shared_ptr<crocoddyl::CostModelSum> costs,
      const Eigen::VectorXd& armature = Eigen::VectorXd(),
      const Eigen::VectorXd& damping = Eigen::VectorXd(),
      const Eigen::VectorXd& coulomb_friction = Eigen::VectorXd(),
      // needs to be a double for shared pointer -> converted internally to
      // integer
      // const Eigen::VectorXd& grav_comp_idxs = Eigen::VectorXd(),
      boost::shared_ptr<crocoddyl::ConstraintModelManager> constraints =
          nullptr);
  virtual ~DifferentialActionModelFrictionFwdDynamics();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u = Eigen::VectorXd());

  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);

  void quasiStatic(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxIter = 100, const double tol = 1e-9);

  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> createData();

  const boost::shared_ptr<crocoddyl::CostModelSum>& get_costs() const;
  const boost::shared_ptr<crocoddyl::ActuationModelAbstract>& get_actuation()
      const;

  const boost::shared_ptr<crocoddyl::ConstraintModelManager>& get_constraints()
      const;

  const Eigen::VectorXd& get_armature() const;
  pinocchio::Model& get_pinocchio() const;

  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation_;
  boost::shared_ptr<crocoddyl::ConstraintModelManager> constraints_;
  boost::shared_ptr<crocoddyl::StateMultibody> state_;
  boost::shared_ptr<crocoddyl::CostModelSum> costs_;
  // pinocchio part
  pinocchio::Model& pinocchio_;
  // friction model
  Eigen::VectorXd armature_;
  Eigen::VectorXd damping_;
  Eigen::VectorXd coulomb_friction_;
  // Eigen::VectorXi grav_comp_idxs_;
  FrictionModel friction_model_;
};

struct DifferentialActionDataFrictionFwdDynamics
    : public crocoddyl::DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::DifferentialActionDataAbstract Base;

  DifferentialActionDataFrictionFwdDynamics(
      DifferentialActionModelFrictionFwdDynamics* const model)
      : crocoddyl::DifferentialActionDataAbstract(model),
        pinocchio(pinocchio::Data(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            boost::make_shared<crocoddyl::JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu())),
        costs(model->get_costs()->createData(&multibody)),
        Minv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        u_drift(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        tmp_xstatic(model->get_state()->get_nx()) {
    Logger::log(LogLevel::DEBUG, "DiffData::DiffData: starting.\n");
    multibody.joint->dtau_du.diagonal().setOnes();
    Logger::log(LogLevel::DEBUG, "DiffData::DiffData: diagonal worked.\n");
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    Logger::log(LogLevel::DEBUG, "DiffData::DiffData: share memory worked.\n");
    Minv.setZero();
    u_drift.setZero();
    dtau_dx.setZero();
    tmp_xstatic.setZero();
  }

  pinocchio::Data pinocchio;
  crocoddyl::DataCollectorJointActMultibody multibody;
  boost::shared_ptr<crocoddyl::CostDataSum> costs;
  boost::shared_ptr<crocoddyl::ConstraintDataManager> constraints;
  MatrixXs Minv;
  VectorXs u_drift;
  MatrixXs dtau_dx;
  VectorXs tmp_xstatic;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};
}  // namespace pendulum_acrobatics
#endif  // ACTION_MODEL_HPP