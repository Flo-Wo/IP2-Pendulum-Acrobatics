from typing import List, Union

import crocoddyl

from utils.costs.activation import ActFuncParamType, ActFuncType

PenaltyType = Union[float, List[float]]


class Penalty:
    def __init__(
        self,
        pen: PenaltyType = 0.0,
        act_func: ActFuncType = crocoddyl.ActivationModelQuad,
        act_func_params: ActFuncParamType = None,
    ):
        self.pen = pen
        self.act_func = act_func
        self.act_func_params = act_func_params

    def __str__(self):
        return "pen: {}, a_func: {}, a_f_params: {}".format(
            self.pen, self.act_func, self.act_func_params
        )
