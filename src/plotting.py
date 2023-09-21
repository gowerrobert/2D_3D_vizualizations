import matplotlib
import matplotlib.pyplot as plt
import pybenchfunction as bench
import numpy as np

font = {"size": 16}
matplotlib.rc("font", **font)
# markers = {"SP2plus": "x", "SP2": "o", "SGD":"2" , "SP":"P", "Adam":"s", "Newton":"v"}
# colours = {"SP2plus": "g", "SP2": "b", "SGD":"y" , "SP":"m", "Adam":"tab:pink", "Newton":"r"}
# colours = ["g", "b", "y", "m", "tab:pink", "r"]

markers = ["o", "s", "v", "x", "D", "^", "D", "p", "o", "x", "s"]
scatter_marker_size = 100
special_marker_size = 26 
init_marker_size = 20

colors = [
    "#1f77b4",
    "#ff7f0e",
    # "#2ca02c",
    "#d62728",
    "#9467bd",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#8c564b",
    "#17becf",
    "#556B2F",
]

def plot_level_set_results(
    bench_function,
    results,
    show=True,
    logscale=False,
):
    fig = plt.figure(figsize=(6, 4))
    bench.plot_2d(
        bench_function,
        n_space=100,
        fig=fig,
        show=False,
        logscale=logscale,
    )
    X_domain, Y_domain = bench_function.input_domain
    X_min, minimum = bench_function.get_global_minimum(2)
    plt.plot(X_min[0], X_min[1], "*", zorder=3, markersize=special_marker_size, color="yellow")
    for i, key in enumerate(results.keys()):
        times, fvals, x_list, teleport_path = results[key]
        mk = markers[i]
        cl = colors[i]
        if x_list == 0:
            continue
        plt.scatter(
            x_list[0],
            x_list[1],
            s=scatter_marker_size,
            label=key,
            zorder=2,
            color=cl,
            marker=mk,
        )
        if len(teleport_path) > 0:
            teleport_path = np.stack(teleport_path)
            plt.scatter(
                teleport_path[:, 0],
                teleport_path[:, 1],
                s=scatter_marker_size,
                label="Teleportation Path",
                zorder=1,
                color=colors[-3],
                marker="x",
            )

        # plt.plot(x_list[0][0], x_list[1][0], "x", markersize=init_marker_size, color="k")
    handles, labels = plt.gca().get_legend_handles_labels()
    
    plt.xlim(X_domain)
    plt.ylim(Y_domain)
    plt.tight_layout()
    # legend = fig.legend(
    #     loc="lower center",
    #     borderaxespad=0.1,
    #     fancybox=False,
    #     shadow=False,
    #     frameon=False,
    #     ncol=4,
    #     fontsize=20,
    # )

    # fig.subplots_adjust(
    #     bottom=0.22,
    # )

    plt.savefig(
        "figures/" + bench_function.name + "-2d.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    if show:
        plt.show()

    plt.figure()
    bench.plot_3d(bench_function, n_space=100, ax=None, show=False)
    plt.plot(X_min[0], X_min[1], "*", markersize=12, color="yellow")
    for i, key in enumerate(results.keys()):
        times, fvals, x_list, teleport_path = results[key]
        mk = markers[i]
        cl = colors[i]
        if x_list == 0:
            continue
        plt.scatter(
            x_list[0],
            x_list[1],
            s=20,
            label=key,
            zorder=1,
            color=cl,
            marker=mk,
        )
        if len(teleport_path) > 0:
            teleport_path = np.stack(teleport_path)
            plt.scatter(
                teleport_path[:, 0],
                teleport_path[:, 1],
                s=20,
                label=key,
                zorder=2,
                color="k",
                marker=mk,
            )
    plt.xlim(X_domain)
    plt.ylim(Y_domain)
    plt.tight_layout()
    plt.savefig(
        "figures/" + bench_function.name + "-3d.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.legend()

    if show:
        plt.show()


def plot_function_values(
    bench_function,
    results,
    timeplot=True,
    add_caption=None,
    show=True,
):
    fig = plt.figure(figsize=(3, 4))
    linewidth = 4
    for i, key in enumerate(results.keys()):
        times, fvals, x_list, teleport_path = results[key]
        mk = markers[i]
        cl = colors[i]
        x = np.arange(len(fvals))

        try:
            if fvals == 0:
                continue
            plt.plot(
                x,
                fvals,
                color=cl,
                label=key,
                linewidth=linewidth,
                markersize=10,
                marker=mk,
                markevery=int(np.floor(len(fvals) / 5)),
            )
        except:
            plt.plot(
                x,
                fvals,
                color=cl,
                label=key,
                linewidth=linewidth,
                markersize=10,
                marker=mk,
            )

    plt.ylabel("function value")
    plt.xlabel("epochs")
    plt.yscale("log")
    plt.legend()
    title = bench_function.name + "-funcs"

    if add_caption is not None:
        title = title + "-" + add_caption + "-"

    plt.savefig("figures/" + title + ".pdf", bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()

    if timeplot:
        plt.figure()
        for i, key in enumerate(results.keys()):
            times, fvals, x_list, teleport_path = results[key]
            mk = markers[i]
            cl = colors[i]
            if key == "Newton":
                continue
            plt.plot(
                time * np.arange(len(fvals)),
                fvals,
                color=cl,
                label=key,
                linewidth=linewidth,
                markersize=10,
                marker=mk,
                markevery=int(np.floor(len(fvals) / 5)),
            )
        plt.ylabel("function value")
        plt.xlabel("time")
        plt.yscale("log")
        plt.legend()
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 5))
        plt.savefig(
            "figures/" + title + "-time.pdf", bbox_inches="tight", pad_inches=0
        )

        if show:
            plt.show()
