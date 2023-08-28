from typing import List, Union

import numpy as np


class ControlBounds:
    def __init__(
        self,
        lower_bound: Union[np.ndarray, List[np.ndarray]] = (-1)
        * np.array([1500, 1250, 400, 600, 0, 0], dtype=np.float64),
        upper_bound: Union[np.ndarray, List[np.ndarray]] = np.array(
            [1500, 1250, 400, 600, 0, 0], dtype=np.float64
        ),
        damping: float = None,
        control_dim: int = 6,
    ):
        """Save the control bounds in a multidimensional np.array to specify
        the bounds in each cost model (based on index) or just define one
        bound, which is going to be used for every cost model. The shape for
        each bound should be either
        - num_cost_models x num_controls
        - num_controls

        Parameters
        ----------
        lower_bound : Union[np.ndarray, List[np.ndarray]], optional
            Pointwise lower bounds for the torques, by default
            (-1)*np.array([1500, 1250, 400, 600, 0, 0]).
        upper_bound : Union[np.ndarray, List[np.ndarray]], optional
            Pointwise upper bounds for the torques, by default
            np.array( [1500, 1250, 400, 600, 0, 0] ).
        damping : float, optional
            Factor to uniformly scale both of the bounds, by default None.
            If not None, both will be multiplied pointwise by damping.
        """
        lower_bound = lower_bound[:control_dim]
        upper_bound = upper_bound[:control_dim]

        assert len(lower_bound) == len(
            upper_bound
        ), "Bounds have to be of the same shape."

        if damping is not None:
            lower_bound *= damping
            upper_bound *= damping
        self.lower = lower_bound
        self.upper = upper_bound

    def get_lower(self, idx):
        try:
            return self.lower[idx, :]
        except IndexError:
            return self.lower

    def get_upper(self, idx):
        try:
            return self.upper[idx, :]
        except IndexError:
            return self.upper
