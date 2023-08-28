
#include "ActionModel.hpp"

// debugging
#include <boost/filesystem.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>

#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace pendulum_acrobatics {

DifferentialActionModelFrictionFwdDynamics::
    DifferentialActionModelFrictionFwdDynamics(
        boost::shared_ptr<crocoddyl::StateMultibody> state,
        boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation,
        boost::shared_ptr<crocoddyl::CostModelSum> costs,
        const Eigen::VectorXd& armature, const Eigen::VectorXd& damping,
        const Eigen::VectorXd& coulomb_friction,
        boost::shared_ptr<crocoddyl::ConstraintModelManager> constraints)
    : crocoddyl::DifferentialActionModelAbstract(state, actuation->get_nu(),
                                                 costs->get_nr()),
      actuation_(actuation),
      armature_(Eigen::VectorXd::Zero(state->get_nv())),
      // pinocchio model and data
      pinocchio_(*state->get_pinocchio().get()),
      damping_(state->get_nv()),
      coulomb_friction_(state->get_nv()),
      costs_(costs),
      constraints_(constraints),
      state_(state),
      // custom friction model
      friction_model_(coulomb_friction, damping) {
  if (armature.size() == 0) {
    armature_ = armature;
  } else {
    armature_.setZero();
  }
  if (damping.size() == 0) {
    damping_.setZero();
  } else {
    damping_ = damping;
  }
  if (coulomb_friction.size() == 0) {
    coulomb_friction_.setZero();
  } else {
    coulomb_friction_ = coulomb_friction;
  }
}
DifferentialActionModelFrictionFwdDynamics::
    ~DifferentialActionModelFrictionFwdDynamics() {}

void DifferentialActionModelFrictionFwdDynamics::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u) {
  Logger::log(LogLevel::DEBUG, "DiffModel::calc: start\n");

  DifferentialActionDataFrictionFwdDynamics* d =
      static_cast<DifferentialActionDataFrictionFwdDynamics*>(data.get());
  Logger::log(LogLevel::DEBUG, "DiffModel::calc: data cast worked.\n");

  actuation_->calc(d->multibody.actuation, x, u);

  Logger::log(LogLevel::DEBUG, "x size: " + std::to_string(x.size()) + ".\n");

  Eigen::VectorXd q = x.head(state_->get_nq());
  Eigen::VectorXd v = x.tail(state_->get_nv());

  Logger::log(LogLevel::DEBUG, "DiffModel::calc: q and v are set.\n");

  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  Logger::log(LogLevel::DEBUG, "DiffModel::calc: computeAllTerms worked.\n");

  d->pinocchio.M.diagonal() += armature_;
  d->Minv.setZero();
  pinocchio::cholesky::decompose(pinocchio_, d->pinocchio);
  pinocchio::cholesky::computeMinv(pinocchio_, d->pinocchio, d->Minv);

  Logger::log(LogLevel::DEBUG, "DiffModel::calc: mass matrix worked.\n");

  Eigen::VectorXd g = Eigen::VectorXd::Zero(d->pinocchio.g.size());

  g.block(0, 0, 4, 1) = d->pinocchio.g.block(0, 0, 4, 1);

  Logger::log(LogLevel::DEBUG, "DiffModel::calc: g segmented.\n");

  // data->xout = mass_matrix.ldlt().solve(u + g + friction_model_.calc(v) -
  //                                       d->pinocchio.nle);
  d->xout = d->Minv * (u + g + friction_model_.calc(v) - d->pinocchio.nle);

  Logger::log(LogLevel::DEBUG, "DiffModel::calc: data->xout worked.\n");

  // is already included in pinocchio::computeAllTerms()
  // pinocchio::forwardKinematics(pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);

  d->multibody.joint->a = d->xout;
  d->multibody.joint->tau = u;

  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x);
  }
}

void DifferentialActionModelFrictionFwdDynamics::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u) {
  Logger::log(LogLevel::DEBUG, "DiffModel::calcDiff: start.\n");

  DifferentialActionDataFrictionFwdDynamics* d =
      static_cast<DifferentialActionDataFrictionFwdDynamics*>(data.get());
  Logger::log(LogLevel::DEBUG, "DiffModel::calcDiff: data cast worked.\n");

  Logger::log(LogLevel::DEBUG, "DiffModel::calcDiff: size(x) : " +
                                   std::to_string(x.size()) + "\n");

  // read sizes once
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();

  Eigen::VectorXd q = x.head(nq);
  Eigen::VectorXd v = x.tail(nv);

  // compute derivative for the actuation
  actuation_->calcDiff(d->multibody.actuation, x, u);

  Logger::log(LogLevel::DEBUG, "DiffModel::calcDiff: q and v are set.\n");

  // TODO: also as class attributes to avoid dynamic memory?
  Eigen::MatrixXd grav_comp_derivatives(state_->get_nq(), state_->get_nq());
  pinocchio::computeGeneralizedGravityDerivatives(pinocchio_, d->pinocchio, q,
                                                  grav_comp_derivatives);
  Logger::log(LogLevel::DEBUG,
              "DiffModel::calcDiff: Compute generalized gravity worked.\n");

  grav_comp_derivatives.block(4, 0, 2, 6).setZero();

  Logger::log(LogLevel::DEBUG, "DiffModel::calcDiff: pruned gravity worked.\n");

  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout);

  d->dtau_dx.leftCols(nq) = d->pinocchio.dtau_dq - grav_comp_derivatives;
  d->dtau_dx.rightCols(nv) = d->pinocchio.dtau_dv - friction_model_.calcDiff(v);

  // Minv is already computed
  d->Fx = (-1) * d->Minv * d->dtau_dx;
  d->Fu = d->Minv;
  // std::cout << "Fx = \n" << d->Fx << "\n";
  // std::cout << "Fu = \n" << d->Fu << "\n";

  Logger::log(LogLevel::DEBUG,
              "DiffModel::calcDiff: calling calDiff on costs.\n");

  d->multibody.joint->da_dx = d->Fx;
  d->multibody.joint->da_du = d->Fu;

  // derivative for the costs
  costs_->calcDiff(d->costs, x, u);

  Logger::log(LogLevel::DEBUG, "DiffModel::calcDiff: WORKED calDiff.\n");
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}
void DifferentialActionModelFrictionFwdDynamics::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
    const std::size_t maxIter, const double tol) {
  Logger::log(LogLevel::DEBUG, "ActionModel::quasiStatic: starting.\n");

  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Logger::log(LogLevel::DEBUG, "ActionModel::quasiStatic: static typecast.\n");

  DifferentialActionDataFrictionFwdDynamics* d =
      static_cast<DifferentialActionDataFrictionFwdDynamics*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();

  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  Logger::log(LogLevel::DEBUG,
              "ActionModel::quasiStatic: calling pinocchio rnea.\n");

  pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv),
                  d->tmp_xstatic.tail(nv));
  Logger::log(LogLevel::DEBUG,
              "ActionModel::quasiStatic: WORKED calling pinocchio rnea.\n");

  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
  Logger::log(LogLevel::DEBUG,
              "ActionModel::quasiStatic: WORKED calling calc.\n");

  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);
  Logger::log(LogLevel::DEBUG,
              "ActionModel::quasiStatic: WORKED calling calcDiff.\n");

  // u.noalias() =
  //     pseudoInverse(d->multibody.actuation->dtau_du) * d->pinocchio.tau;
  u.setZero();
  d->pinocchio.tau.setZero();
}

boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>
DifferentialActionModelFrictionFwdDynamics::createData() {
  return boost::allocate_shared<DifferentialActionDataFrictionFwdDynamics>(
      Eigen::aligned_allocator<DifferentialActionDataFrictionFwdDynamics>(),
      this);
}
const boost::shared_ptr<crocoddyl::CostModelSum>&
DifferentialActionModelFrictionFwdDynamics::get_costs() const {
  return costs_;
}

const boost::shared_ptr<crocoddyl::ActuationModelAbstract>&
DifferentialActionModelFrictionFwdDynamics::get_actuation() const {
  return actuation_;
}
pinocchio::Model& DifferentialActionModelFrictionFwdDynamics::get_pinocchio()
    const {
  return pinocchio_;
}
const boost::shared_ptr<crocoddyl::ConstraintModelManager>&
DifferentialActionModelFrictionFwdDynamics::get_constraints() const {
  return constraints_;
}

const Eigen::VectorXd&
DifferentialActionModelFrictionFwdDynamics::get_armature() const {
  return armature_;
}

}  // namespace pendulum_acrobatics

/*
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

  boost::filesystem::path currentPath = boost::filesystem::current_path();
  boost::filesystem::path urdfFilePath = currentPath / relative_path_urdf;

  pinocchio::Model pin_model_;
  pinocchio::urdf::buildModel(urdfFilePath.string(), pin_model_);
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(
          boost::make_shared<pinocchio::Model>(pin_model_));
  Logger::log(LogLevel::DEBUG, "State is built correctly.\n");

  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
      boost::make_shared<crocoddyl::ActuationModelFull>(state);
  Logger::log(LogLevel::DEBUG, "Actuation is built correctly.\n");
  boost::shared_ptr<crocoddyl::CostModelSum> costs =
      boost::make_shared<crocoddyl::CostModelSum>(state);

  // friction model constants
  Eigen::VectorXd armature(6);  // Assuming 6 degrees of freedom
  armature.setZero();
  // armature << 0.111266, 0.053249, 0.0564972, 0.0182617, 0., 0.;
  Eigen::VectorXd damping(6);  // Assuming 6 degrees of freedom
  // damping << 1.31174e+00, 4.92821e-01, 1.48827e+00, 1.50353e-01, 1.00000e-03,
  //     1.00000e-03;
  damping.setZero();
  Eigen::VectorXd coulomb_friction(6);  // Assuming 6 degrees of freedom
  coulomb_friction.setZero();
  // coulomb_friction << 1.70477, 1.43072, 0.685072, 0.854358, 0., 0.;

  pendulum_acrobatics::DifferentialActionModelFrictionFwdDynamics model =
      pendulum_acrobatics::DifferentialActionModelFrictionFwdDynamics(
          state, actuation, costs, armature, damping, coulomb_friction);
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model.createData();

  // test calculation
  Eigen::VectorXd x(12);
  // x << 0, 0.78, 0, 2.37, 0, 0, Eigen::VectorXd::Zero(6);
  x << 0, -1.3, 0, 1.3, 0, 0, Eigen::VectorXd::Ones(6);

  Eigen::VectorXd u = Eigen::VectorXd::Ones(6);
  // u.setZero();

  model.calc(data, x, u);
  model.calcDiff(data, x, u);

  return 0;
}
*/