from utils.costs.integrated_models import ExpIntCostModels, UniformIntCostModels


class IntCostModelsFactory:
    @staticmethod
    def get_int_cost_model(time_schema: str, **kwargs):
        if time_schema == "uniform":
            return UniformIntCostModels(**kwargs)
        elif time_schema == "exp":
            return ExpIntCostModels(**kwargs)
        else:
            raise NotImplementedError
