import numpy as np
import pinocchio as pin
import logging


class Derivatives:
    def __init__(
        self,
        v_joint: np.ndarray,
        a_joint: np.ndarray,
        v_task: np.ndarray,
        a_task: np.ndarray,
        normalized: bool = True,
        combined_max: bool = False,
    ):

        self.v_joint = v_joint
        self.a_joint = a_joint
        self.v_task = v_task
        self.a_task = a_task

        self.normalized = normalized
        self.combined_max = combined_max


def derivatives_joint_task(
    model,
    data,
    q_t: np.ndarray,
    dq_t: np.ndarray,
    u_t: np.ndarray,
    frame_id: int,
    ref_frame=pin.ReferenceFrame.WORLD,
    apply_norm: bool = True,
    normalize: bool = True,
    combined_max: bool = False,
) -> Derivatives:
    """Compute derivatives in joint and task space:
        - a_joint
        - v_task
        - a_task
    for a given reference trajectory.

    Parameters
    ----------
    model : pinocchio model
        Internal pinocchio model.
    data : pinocchio data
        Internal pinocchio data.
    q_t : np.ndarray
        Sequence of joint configs with shape (n, n_joints).
    dq_t : np.ndarray
        Sequence of joint velocities with shape (n, n_joints).
    u_t : np.ndarray
        Sequence of torques with shape (n, n_u).
    frame_id : int
        Id of the **FRAME** to compute the derivative in the task space.
        If you want the acceleration of a joint instead, you need to
        rewrite this function.
    ref_frame : pin.ReferenceFrame
        Reference frame w.r.t. which to compute the derivative
        in the task space.
    apply_norm : bool
        Compute the pointwise l2-norm of the derivatives, the
        default is True.
    normalize : bool
        Normalize the value by the maximum value of the array,
        default is True.

    Returns
    -------
    np.ndarray, np.ndarray
        Joint velocities and accelerations.
    """
    assert np.shape(q_t) == np.shape(
        dq_t
    ), "Joint configs and velocities need to have the same shape"
    assert np.shape(q_t) == np.shape(
        u_t
    ), "Torques and joint configs need to have the same shape"

    # preallocate memory
    v_joint: np.ndarray = dq_t
    # only need to compute these three quantities
    a_joint: np.ndarray = np.zeros(q_t.shape)
    v_task: np.ndarray = np.zeros(q_t.shape)
    a_task: np.ndarray = np.zeros(q_t.shape)

    for idx in range(np.shape(q_t)[0]):
        a_joint[idx, :], v_task[idx, :], a_task[idx, :] = _derivatives_pointwise(
            model,
            data,
            q_t[idx, :],
            dq_t[idx, :],
            u_t[idx, :],
            frame_id,
            ref_frame,
        )
    if apply_norm:
        # joint space
        v_joint = np.linalg.norm(v_joint, ord=2, axis=1)
        a_joint = np.linalg.norm(a_joint, ord=2, axis=1)

        # task space
        v_task = np.linalg.norm(v_task, ord=2, axis=1)
        a_task = np.linalg.norm(a_task, ord=2, axis=1)

        if normalize:
            # joint space
            v_joint, a_joint = _normalize(v_joint, a_joint, combined_max=combined_max)

            # task space
            v_task, a_task = _normalize(v_task, a_task, combined_max=combined_max)
    return Derivatives(
        v_joint=v_joint,
        a_joint=a_joint,
        v_task=v_task,
        a_task=a_task,
        normalized=normalize,
        combined_max=combined_max,
    )


def _derivatives_pointwise(
    model,
    data,
    q,
    dq,
    tau,
    frame_id: int,
    ref_frame=pin.ReferenceFrame.WORLD,
):
    data = model.createData()
    pin.computeAllTerms(model, data, q, dq)
    a_joint = pin.aba(model, data, q, dq, tau)
    v_task = np.array(pin.getFrameVelocity(model, data, frame_id, ref_frame))
    a_task = np.array(pin.getFrameAcceleration(model, data, frame_id, ref_frame))
    return a_joint, v_task, a_task


def _normalize(
    *vecs: np.ndarray, max_value: float = None, combined_max: bool = False
) -> np.ndarray:
    if max_value is None and combined_max:
        max_value = max([vec.max() for vec in vecs])
    normed_vecs = []
    for vec in vecs:
        # compute pointwise max, if no max is given and we do not use a combined max
        if max_value is None or not combined_max:
            max_value = vec.max()

        if max_value > 0:
            vec *= 1 / max_value
        else:
            logging.warn(
                "Vector is all zeros, no normalization possible. Returned raw vector."
            )
        normed_vecs.append(vec)
    return tuple(normed_vecs)
