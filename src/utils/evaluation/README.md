## Important information regarding the evaluation of the setpoint task

### Target difference
If the target has size n then the saved real trajectory has size n+1, as the first/initial
state of the robot is used to initialize the caching inside the mpc solver.
Thus, we eliminate the first element of the saved trajectory by evaluating the difference
```python
x_traj[:, 1:] - goal_traj.T
```

### MPC solver params
Furthermore, we assume that the MPC solver always uses the same config for every mpc horizon.
A change would not be a problem, but has to be done.