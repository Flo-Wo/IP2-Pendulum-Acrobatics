from typing import List

import crocoddyl
import numpy as np

from utils.decorators import log_time
from utils.solver.DDP.ddp_results import DDPResults


@log_time(get_time=True)
def ddp(
    int_cost_models: List[crocoddyl.IntegratedActionModelEuler],
    terminal_model: crocoddyl.IntegratedActionModelEuler,
    q: np.ndarray,
    qdot: np.ndarray,
    solver_max_iter: int = 5000,
    show_logs: bool = False,
    solver: crocoddyl.SolverAbstract = crocoddyl.SolverBoxFDDP,
    static_warmstart: bool = True,
) -> DDPResults:
    problem = crocoddyl.ShootingProblem(
        np.concatenate((q, qdot)), int_cost_models, terminal_model
    )

    # use the initial state to start planning --> will be updated in the following
    problem.x0 = np.concatenate((q, qdot))
    ddp = solver(problem)

    show_logs = True
    if show_logs:
        log = crocoddyl.CallbackLogger()
        ddp.setCallbacks([log, crocoddyl.CallbackVerbose()])
        ddp.getCallbacks()[0].precision = 3
        ddp.getCallbacks()[0].level = crocoddyl.VerboseLevel._2

    if static_warmstart:
        state = np.concatenate((q, np.zeros(np.shape(q))))
        x_warm = [state] * (ddp.problem.T + 1)
        u_warm = ddp.problem.quasiStatic([state] * ddp.problem.T)
        solved = ddp.solve(x_warm, u_warm, solver_max_iter, True)
    else:
        solved = ddp.solve([], [], solver_max_iter)

    return DDPResults(np.array(ddp.xs), np.array(ddp.us), solved)
