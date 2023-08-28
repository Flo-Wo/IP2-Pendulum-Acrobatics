import crocoddyl

from utils.costs.integrated_models.int_models_base import IntCostModelsBase


class ExpIntCostModels(IntCostModelsBase):
    def __init__(
        self,
        dict_int_cost_models: dict[int, list[crocoddyl.IntegratedActionModelEuler]],
        mpc_horizon: int,
        schema_int_factors: list[int],
        **kwargs
    ):
        super(IntCostModelsBase, self).__init__()

        self.dict_int_cost_models = dict_int_cost_models
        print(dict_int_cost_models.keys())
        self.mpc_horizon = mpc_horizon

        assert (
            len(schema_int_factors) == mpc_horizon
        ), "Schema has to have the length of the MPC horizon."

    def __getitem__(self, time_idx: int) -> list[crocoddyl.IntegratedActionModelEuler]:
        # if isinstance(time_idx, int):
        #     pass
        # elif isinstance(time_idx, slice):
        #     pass

        return self.dict_int_cost_models[int(time_idx % self.mpc_horizon)][time_idx]
