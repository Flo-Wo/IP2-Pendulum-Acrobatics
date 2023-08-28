# Attention

For evaluation purposes it is really important that we keep the order of the first two sensors
inside any ``.xml``-file we want to evaluate the same:
1) the position of the pendulum's tip in the global frame
2) the rotation of the pendulum's tip in the global frame expresses as a quaternion
--> Thus, we can always read the first 7 values of the mj_sensordata in our results class to obtain the position and the rotation of the 
trajectory

```xml
<framepos name="pend_glob_end" objtype="site" objname="wam/sensor_sites/pend_endeff" reftype="site" refname="wam/ref_sites/global_origin"/>
<framequat name="pend_glob_end_quat" objtype="site" objname="wam/sensor_sites/pend_endeff" reftype="site" refname="wam/ref_sites/global_origin"/>
<framepos name="pend_begin_end" objtype="site" objname="wam/sensor_sites/pend_begin" reftype="site" refname="wam/ref_sites/global_origin"/>
```


# Configurations
The configs are given by
```python
q_config = np.array([0, -0.3, 0, 0.3, 0, 0])
q_config = np.array([0, -1.7, 0, 1.7, 0, 0])
q_config = np.array([0, -0.78, 0, 2.37, 0, 0])
q_config = np.array([0, -1.6, 1.55, 1.6, 0, 1.55])
```

## Optitrack Markers
The Number and positions of the optitrack markers is defined in the ``.xml``-file and the ``config_utils.py`` file

```json
<!-- Observers for the optitrack markers -->
<site name="pendulum/sensor_sites/optitrack_marker_one" pos="0 0 optitrack_marker_one_pos" size="0.005" rgba="0 0 1 1"/>
<site name="pendulum/sensor_sites/optitrack_marker_two" pos="0 0 optitrack_marker_two_pos" size="0.005" rgba="0 0 1 1"/>
<site name="pendulum/sensor_sites/optitrack_marker_three" pos="0 0 optitrack_marker_three_pos" size="0.005" rgba="0 0 1 1"/>
<site name="pendulum/sensor_sites/optitrack_marker_four" pos="0 0 optitrack_marker_four_pos" size="0.005" rgba="0 0 1 1"/>
```

And the corresponding python code:

```python
def _optitrack_marker_pos(
    pendulum_len: float = 0.3,
) -> Dict[str, float]:
    return dict(
        one=pendulum_len / 6,
        two=pendulum_len / 3,
        three=3 * pendulum_len / 4,
        four=pendulum_len - 0.01,
    )
```