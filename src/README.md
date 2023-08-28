# Tips and Tricks for pinocchio and mujoco

Link to crocoddyl source code: [here](https://github.com/loco-3d/crocoddyl/tree/master/include/crocoddyl/multibody)

## Helpful commands:

Check urdf model for validity:
```bash
check_urdf <model_name>.urdf
```

Manual limits for the urdf model in ``pinocchio``: see 
[here](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/structpinocchio_1_1ModelTpl.html#a472dd6bb4baf9c37a336200c54d4cddc)

## URDF links
For joints see: [joints](http://wiki.ros.org/urdf/XML/joint).


## Pinocchio internal variables 
see [here](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/structpinocchio_1_1DataTpl.html#a21ab78d59471b2a7d7b3a42a8dd4d8d0)
for more information on internal positional variables

Get absolute positions of the **joint** placements (w.r.t the world frame)
```python
for name, oMi in zip(pin_model.names, pin_data.oMi):
    print(
        (
            "{:<24} : {: .4f} {: .4f} {: .4f}".format(
                name, *oMi.translation.T.flat
            )
        )
    )
```

Get absolute positions of all **frames** (w.r.t. the world frame), you need to run
the forward kinematics and updateFramePlacement functions first
```python
for idx in range(len(pin_data.oMf)):
    print(pin_data.oMf[idx].translation)
```

A fully working script:
```python
import pinocchio as pin
pin.forwardKinematics(
        pin_model,
        pin_data,
        start_config,
        np.zeros(pin_model.nv),
    )
pin.updateFramePlacements(pin_model, pin_data)
for idx in range(len(pin_data.oMf)):
    print(
        "idx: {}, pos: {}".format(idx, pin_data.oMf[idx].translation)
    )
```

Compute center of masses:
```python
import pinocchio as pin
com_body = pin.centerOfMass(pin_model, pin_data, q, True)
# results are accessable in pin_data
for i in range(len(pin_data.com)):
    print(pin_data.com[i])
```

Get mass of the model's subtree's:
```python
# mujoco
print(mj_model.body_mass) # mass of each body
# mass of the whole subtree --> summing all following elements + own weight
print(mj_model.body_subtreemass) 

# pinocchio
print([x for x in pin_data.mass]) # mass of subtrees
```

### Important frame numbers
Index of the wam-endeffector frame (without the mounted plates, poles), index
of the base-pole frame, index of the pendulum endeffector frame
(comments give the resting position in the world frame and their orientation)

```python
length_wam_end_to_pole = 0.355 # = 0.005 (base plate) + 0.05 (base pole) + 0.3 (pend pole)
wam_end_id = 14
#  index: 14
# [-1.22124533e-16 -7.50000000e-03  2.44600000e+00]
# [[ 2.22044605e-16  1.00000000e+00  0.00000000e+00]
#  [-1.00000000e+00  2.22044605e-16  4.93038066e-32]
#  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]

pend_tip_id = 24
#  index: 24
# [-1.22124533e-16 -7.50000000e-03  2.80100000e+00]
# [[ 2.22044605e-16  1.00000000e+00  0.00000000e+00]
#  [-1.00000000e+00  2.22044605e-16  4.93038066e-32]
#  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```

Thus, we can follow a trajectory with the wam endeffector and either penalize
the rotation of the pendulum or just add the constant length to the z-coordinates
of the trajectory, to also enforce a upwards oriented pendulum.

# Notes:
Currently we have an offset of ``0.00012168768`` between the endeffector of the ``.xml`` file and
the endeffector of the ``.urdf`` file.