import matplotlib.pyplot as plt
import numpy as np


def plot_2D_projected_trajectory(
    actual_pole_tip_positions: np.ndarray,
    ref_pole_tip_pos: np.ndarray,
    planned_pole_tip_positions: np.ndarray = None,
    target_pole_tip_positions: np.ndarray = None,
    filename: str = "LQR_pole_positions_projected",
    save: bool = True,
    time_step: float = 0.002,
    path: str = "../report/imgs/",
):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size="14")

    # orange
    target_color: str = "#ff7f00"  # "orange"
    # brown
    preplanned_color: str = "#a65628"  # "#377eb8"  # "grey"

    target_linestyle: str = "dotted"
    preplanned_linestyle: str = "solid"

    f, axs = plt.subplots(1, 2, figsize=(16, 6))

    num_points = actual_pole_tip_positions.shape[0]

    print("Projection plot, num_points= ", num_points)
    time_shifts = np.linspace(0, num_points * time_step, num_points)

    # Create a scatter plot with varying colors based on time
    scatter_xy = axs[0].scatter(
        actual_pole_tip_positions[:, 0] - ref_pole_tip_pos[0],
        (1) * (actual_pole_tip_positions[:, 1] - ref_pole_tip_pos[1]),
        c=time_shifts,
        cmap="viridis",
        s=10,
    )
    axs[0].set_xlabel("$(x^\mathrm{{tip}}_t -x^\mathrm{{tip}}_0)_{1}$")
    axs[0].set_ylabel("$(x^{\mathrm{tip}}_t - x^{\mathrm{tip}}_0)_{2}$")
    axs[0].set_title("$xy$-Projection")
    axs[0].axis("equal")

    # include preplanning
    if planned_pole_tip_positions is not None:
        axs[0].plot(
            planned_pole_tip_positions[:, 0] - ref_pole_tip_pos[0],
            planned_pole_tip_positions[:, 1] - ref_pole_tip_pos[1],
            color=preplanned_color,
            linestyle=preplanned_linestyle,
            label="FDDP planning",
        )

    if target_pole_tip_positions is not None:
        axs[0].plot(
            target_pole_tip_positions[:, 0],
            target_pole_tip_positions[:, 1],
            color=target_color,
            linestyle=target_linestyle,
            label="target",
        )
    # single target
    else:
        axs[0].plot(0, 0, color=target_color, linestyle=" ", marker="x", label="target")
    axs[0].legend(loc="upper right")

    scatter_yz = axs[1].scatter(
        actual_pole_tip_positions[:, 1] - ref_pole_tip_pos[1],
        actual_pole_tip_positions[:, 2] - ref_pole_tip_pos[2],
        c=time_shifts,
        cmap="viridis",
        s=10,
    )
    axs[1].set_xlabel("$(x^\mathrm{{tip}}_t -x^\mathrm{{tip}}_0)_{2}$")
    axs[1].set_ylabel("$(x^{\mathrm{tip}}_t - x^{\mathrm{tip}}_0)_{3}$")
    axs[1].set_title("$yz$-Projection")
    axs[1].axis("equal")

    # Add a colorbar to show the time shift
    cbar = plt.colorbar(scatter_yz, ax=axs[1])
    cbar.set_label("time [s]")

    # include preplanning
    if planned_pole_tip_positions is not None:
        axs[1].plot(
            planned_pole_tip_positions[:, 1] - ref_pole_tip_pos[1],
            planned_pole_tip_positions[:, 2] - ref_pole_tip_pos[2],
            color=preplanned_color,
            linestyle=preplanned_linestyle,
            label="FDDP planning",
        )

    # include target, trajectory or single point
    if target_pole_tip_positions is not None:
        axs[1].plot(
            target_pole_tip_positions[:, 1],
            target_pole_tip_positions[:, 2],
            color=target_color,
            linestyle=target_linestyle,
            label="target",
        )
    else:
        axs[1].plot(
            0,
            0,
            color=target_color,
            linestyle=" ",
            marker="x",
            label="target",
        )

    axs[1].legend(loc="best")
    if save:
        plt.savefig(path + filename)
    plt.show()
