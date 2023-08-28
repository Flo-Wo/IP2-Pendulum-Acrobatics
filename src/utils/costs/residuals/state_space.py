from utils.costs.residuals import ResidualBase
import numpy as np
import crocoddyl
import warnings


class StateSpaceResidual(ResidualBase):
    """
    In crocoddyl a state is a pair sate = (q, v) of a configuration point q
    and a tangent vector v.

    Residual behaves as follows:
    ----------------------------
    * (q, v) given as matrices of the shape time x state.nq respectively
    time x state.nv --> time step wise based cost
    * (q, None) we set the velocity to zero and either scale up the shape
    of v or use the same q in each time step (e.g. if q is the configuration
    of a resting position) --> in this case a warning is thrown
    as an appropriate weighting of the activation function might be needed
    * (None, v) --> as above with changed roles
    * (None, None) --> we have a resting position with configuration q=0
    and velocity v = 0, also a warning is thrown
    """

    def __init__(self, q: np.ndarray, v: np.ndarray, **kwargs):
        super(StateSpaceResidual, self).__init__(**kwargs)
        self._set_costs(q, v)

    def _set_costs(self, q, v):
        if q is None:
            q = np.zeros(self.state.nq)
            warnings.warn(
                "Config q is set to zeros. Pay attention to an appropriate weighting of the cost function"
            )
        if v is None:
            v = np.zeros(self.state.nv)
            warnings.warn(
                "Vel. v is set to zeros. Pay attention to an appropriate weighting of the cost function"
            )
        if q.ndim == 1 and v.ndim == 1:
            self._both_1d(q, v)
        elif q.ndim > 1 and v.ndim > 1:
            self._both_2d(q, v)
        elif q.ndim > 1:
            # need to scale up v
            v = np.tile(v, (q.shape[0], 1))
            self._both_2d(q, v)
        elif v.ndim > 1:
            # need to scale up q
            q = np.tile(q, (v.shape[0], 1))
            self._both_2d(q, v)

    def _both_2d(self, q, v):
        assert np.shape(q) == np.shape(
            v
        ), "Derivative and trajectory must have the same shape"
        models = []
        for idx in range(np.shape(q)[0]):
            target = np.concatenate((q[idx, :], v[idx, :]))
            res_model = crocoddyl.ResidualModelState(
                self.state, target, self.dim_control
            )
            models.append(
                crocoddyl.CostModelResidual(
                    self.state,
                    self.act_func[idx],
                    res_model,
                )
            )
        assert len(models) == len(
            q[:]
        ), "Different number of residual models than trajectory points"
        self.costs = models
        self.len_costs = len(models)

    def _both_1d(self, q, v):
        target = np.concatenate((q, v))
        models = []
        for idx in range(len(self.act_func)):
            models.append(
                crocoddyl.CostModelResidual(
                    self.state,
                    self.act_func[idx],
                    crocoddyl.ResidualModelState(self.state, target, self.dim_control),
                )
            )
        self.costs = models
        self.len_costs = len(models)
