from utils.ros_helper.state_observer import StateObserverBase
from utils.ros_helper.state_observer.observer_pin_sim import StateObserverPinSim
from utils.ros_helper.state_observer.state_observer_sync import StateObserverSync
from utils.ros_helper.state_observer.state_observer_thread import StateObserverThread


class StateObserverFactory:
    @staticmethod
    def get_observer(observer_type, *args, **kwargs) -> StateObserverBase:
        if observer_type == "thread":
            return StateObserverThread(*args, **kwargs)
        elif observer_type == "pin_pred":
            return StateObserverPinSim(*args, **kwargs)
        if observer_type == "sync":
            return StateObserverSync(*args, **kwargs)
        else:
            raise NotImplementedError
