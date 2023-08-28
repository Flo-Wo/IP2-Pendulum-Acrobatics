import crocoddyl
from typing import Union, List

ActFuncType = Union[
    crocoddyl.ActivationModelAbstract, List[crocoddyl.ActivationModelAbstract]
]
ActFuncParamType = Union[int, tuple, List[tuple]]


class Activation:
    def __init__(
        self,
        act_func_params: ActFuncParamType,
        act_func: ActFuncType,
    ):
        # check dimensions first --> then set the attributes
        self._compute_len(act_func, act_func_params)

        self.act_func_params = act_func_params
        self.act_func = act_func

    def _compute_len(self, act_func, act_func_params):
        self.func_list = isinstance(act_func, list)
        self.params_list = isinstance(act_func_params, list)

        self.len_func = len(act_func) if self.func_list else 0
        self.len_params = len(act_func_params) if self.params_list else 0

        if self.len_func > 0 and self.len_params > 0:
            assert (
                self.len_func == self.len_params
            ), "List of parameters and list of activation functions must have the same length."

        # length minimum is 1
        self.len_max = max(max(self.len_func, self.len_params), 1)

    def __getitem__(self, key: int):
        self._raise_index_err(key)  # check if the key is valid

        param = self._get_param(key)
        if isinstance(param, tuple):
            return self._get_func(key)(*param)
        return self._get_func(key)(param)

    def __len__(self):
        return self.len_max

    def _get_func(self, key: int):
        if self.func_list:
            return self.act_func[key]
        return self.act_func

    def _get_param(self, key: int):
        if self.params_list:
            return self.act_func_params[key]
        return self.act_func_params

    def _raise_index_err(self, key: int):
        # if one of them is given as a list, we have to check for out of bounce errors
        if (self.params_list or self.func_list) and key >= self.len_max:
            raise IndexError("Index out of range")
