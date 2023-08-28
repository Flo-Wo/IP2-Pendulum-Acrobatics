import crocoddyl


class IntCostModelsBase:
    def __getitem__(self, time_idx: int) -> list[crocoddyl.IntegratedActionModelEuler]:
        raise NotImplementedError
