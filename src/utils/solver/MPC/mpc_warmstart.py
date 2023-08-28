class MPCWarmStart:
    def __init__(self, states: list = [], torques: list = [], feasible: bool = False):
        assert len(states) == len(
            torques
        ), "Length of states and torques has to be equal."
        self.x = states
        self.u = torques
        self.feasible = feasible

    def __bool__(self):
        return len(self.x) != 0

    def get_warm(self):
        """x, u, feasible"""
        return self.x, self.u, self.feasible

    def set_warm(self, x, u, solved):
        """x, u, solved from the solver call"""
        self.x = x
        self.u = u
        self.feasible = solved
