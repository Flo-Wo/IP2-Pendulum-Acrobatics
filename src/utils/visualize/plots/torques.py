import matplotlib.pyplot as plt
import numpy as np


def plot_torques(
    torques: np.ndarray,
    filename: str,
    planned_torques: np.ndarray = None,
    x_axis: np.ndarray = None,
    path: str = "../report/imgs/",
    torque_duration: float = 0.002,
):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size="14")
    f, ax = plt.subplots(4, 1, figsize=(16, 7))

    if x_axis is None:
        duration = planned_torques.shape[0] * torque_duration
        x_axis = np.linspace(0, duration, planned_torques.shape[0])
        planned_x_axis = x_axis
    else:
        duration = planned_torques.shape[0] * torque_duration
        planned_x_axis = np.linspace(0, duration, planned_torques.shape[0])

    for i in range(4):
        ax[i].plot(
            x_axis,
            torques[:, i],
            label="$(u^{{\mathrm{{LQR}}}}_{{t}})_{0}$".format(i + 1),
            color="royalblue",
            linestyle="dashed",
        )
        if planned_torques is not None:
            ax[i].plot(
                planned_x_axis,
                planned_torques[:, i],
                label="$(u^{{\mathrm{{FDDP}}}}_{{t}})_{0}$".format(i + 1),
                color="#ff7f00",
            )
        ax[i].legend(loc="lower right", ncol=2)
    # f.align_labels()
    f.supxlabel("time [s]")
    f.supylabel("torque [Nm]")
    plt.tight_layout()
    plt.savefig(path + filename)
    plt.show()
