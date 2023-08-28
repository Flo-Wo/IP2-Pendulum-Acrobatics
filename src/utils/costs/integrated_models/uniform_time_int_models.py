import crocoddyl

from utils.costs.integrated_models.int_models_base import IntCostModelsBase


class UniformIntCostModels(IntCostModelsBase):
    def __init__(
        self,
        int_cost_models: list[crocoddyl.IntegratedActionModelEuler],
        **kwargs,
    ):
        super(IntCostModelsBase, self).__init__()

        self.int_cost_models = int_cost_models

    def __getitem__(self, time_idx: int) -> list[crocoddyl.IntegratedActionModelEuler]:
        return self.int_cost_models[time_idx]
