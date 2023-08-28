import matplotlib.pyplot as plt


def plot_task_space_latex(task_space: dict, labels: dict, filename: str):
    linestyles = {
        "target": "-",
        "planned": "--",
        "ddp": ":",
    }
    colors = {
        "target": "gray",
        "planned": "orange",
        "ddp": "firebrick",
    }

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size="14")

    num_plots = task_space["target"].shape[1]
    fig, ax = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 5))

    x_label = "$(x^\mathrm{{tip}}_t)_{}$"
    for i, ax in enumerate(fig.axes):
        for model in task_space.keys():
            ax.plot(
                task_space[model][:, i],
                color=colors[model],
                linestyle=linestyles[model],
                label=labels[model],
            )
            ax.set_ylabel(x_label.format(i + 1))
        if i < 1:
            ax.legend(loc="lower left", ncol=3)
    fig.align_labels()
    plt.xlabel("time step $t$")
    plt.savefig(f"../report/imgs/{filename}.pdf", transparent=True)
    plt.show()


def task_space_and_angles(
    angles: dict,
    task_space: dict,
    labels: dict,
):
    linestyles = {
        "target": "-",
        "raw": "-.",
        "planned": "--",
        "ddp": ":",
    }
    colors = {
        # "target": "firebrick",
        "target": "gray",
        "raw": "royalblue",
        "planned": "orange",
        # "ddp": "forestgreen",
        "ddp": "firebrick",
    }

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size="14")

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

    x_label = "$(x^\mathrm{{tip}}_t)_{}$"
    for i, ax in enumerate(fig.axes):
        if i == 0:
            for model in angles.keys():
                ax.plot(
                    angles[model],
                    color=colors[model],
                    linestyle=linestyles[model],
                    label=labels[model],
                )
            ax.set_ylabel("angle [$^{\circ}$]")
            ax.legend(loc="lower right", ncol=4)
        else:
            for model in task_space.keys():
                ax.plot(
                    task_space[model][:, i - 1],
                    color=colors[model],
                    linestyle=linestyles[model],
                    label=labels[model],
                )
                ax.set_ylabel(x_label.format(i))
            if i < 1:
                ax.legend(loc="lower right", ncol=4)
    fig.align_labels()
    plt.xlabel("time step $t$")
    plt.savefig("../report/imgs/compare_raw_vs_preplanning.pdf")
    # tikzplotlib.save("../report/imgs/compare_raw_vs_preplanning.tex")
    plt.show()
