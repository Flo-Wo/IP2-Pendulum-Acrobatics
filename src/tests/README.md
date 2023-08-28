
# Rotate the pendulum vector to the local frame again

To compute the Rotation matrix in mujoco, add the following sensors to your xml-file
```json
<framexaxis name="pend_x_axis_global" objtype="site" objname="pendulum/sensor_sites/reference_frame" reftype="site" refname="wam/ref_sites/global_origin"/>
<frameyaxis name="pend_y_axis_global" objtype="site" objname="pendulum/sensor_sites/reference_frame" reftype="site" refname="wam/ref_sites/global_origin"/>
<framezaxis name="pend_z_axis_global" objtype="site" objname="pendulum/sensor_sites/reference_frame" reftype="site" refname="wam/ref_sites/global_origin"/>
```

with a reference frame being the pendulums base:
```json
<body name="pendulum/links/base" pos="-0.045 -0.35 0" euler="1.570796 0 1.570796">
    <site name="pendulum/sensor_sites/reference_frame" pos="0 0 0" size="0.005" rgba="0 1 0 1" />
```