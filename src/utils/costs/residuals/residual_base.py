import warnings
from typing import List, Union

import crocoddyl

from utils.costs.activation import ActFuncParamType, ActFuncType, Activation


class ResidualBase:
    len_costs: int
    costs: Union[crocoddyl.CostModelAbstract, List[crocoddyl.CostModelAbstract]]

    def __init__(
        self,
        state: crocoddyl.StateMultibody,
        name: str,
        penalty: Union[float, List[float]],
        dim_residual: int,
        dim_control: int = 6,
        act_func: ActFuncType = crocoddyl.ActivationModelQuad,
        act_func_params: ActFuncParamType = None,
    ):
        self.state = state
        self.penalty, self.len_penalty = self._set_penalty(penalty)
        self.name = name
        self.dim_residual = dim_residual
        self.dim_control = dim_control

        self.act_func = self._activation_func(act_func, act_func_params)

    def _activation_func(self, act_func, act_func_params):
        if act_func_params is None:
            act_func_params = self.dim_residual
        return Activation(act_func_params=act_func_params, act_func=act_func)

    def __getitem__(self, key: int):
        return self.name, self._get_costs(key), self._get_penalty(key)

    def _set_penalty(self, penalty):
        if isinstance(penalty, list):
            if len(penalty) > 1:
                len_penalty = len(penalty)
            elif len(penalty) == 1:
                warnings.warn("Penalty length is 1, first element is always used.")
                penalty = penalty[0]
                len_penalty = 1
            else:
                raise ValueError("Penalty list has to be non-empty.")
        else:
            len_penalty = 1
        return penalty, len_penalty

    def _get_penalty(self, key: int):
        if self.len_penalty > 1:
            return self.penalty[key]
        return self.penalty

    # individual implementation for each residual type
    def _get_costs(self, key: int):
        self._raise_index_err(key)
        # the default is a list with length 1 (e.g. control penalties)
        if self.len_costs > 1:
            return self.costs[key]
        return self.costs[0]

    def _raise_index_err(self, key: int):
        if self.len_costs > 1 and key >= self.len_costs:
            raise IndexError("Index out of range")

    def _set_costs(self):
        raise NotImplementedError
