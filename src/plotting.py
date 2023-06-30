import matplotlib
import matplotlib.pyplot as plt
import pybenchfunction as bench
import numpy as np

font = {"size": 12}
matplotlib.rc("font", **font)
# markers = {"SP2plus": "x", "SP2": "o", "SGD":"2" , "SP":"P", "Adam":"s", "Newton":"v"}
# colours = {"SP2plus": "g", "SP2": "b", "SGD":"y" , "SP":"m", "Adam":"tab:pink", "Newton":"r"}
markers = ["x", "o", "2", "P", "s", "v"]
colours = ["g", "b", "y", "m", "tab:pink", "r"]


def plot_level_set_results(
    bench_function,
    results,
    show=True,
    logscale=False,
):
    bench.plot_2d(
        bench_function, n_space=100, ax=None, show=False, logscale=logscale
    )
    X_domain, Y_domain = bench_function.input_domain
    X_min, minimum = bench_function.get_global_minimum(2)
    plt.plot(X_min[0], X_min[1], "*", markersize=10, color="yellow")
    mks = markers.copy()
    cls = colours.copy()
    for key in results.keys():
        times, fvals, x_list = results[key]
        mk = mks.pop()
        cl = cls.pop()
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
        plt.plot(x_list[0][1], x_list[1][1], "X", markersize=10, color=cl)
    plt.xlim(X_domain)
    plt.ylim(Y_domain)
    plt.tight_layout()
    plt.legend()
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

    mks = markers.copy()
    cls = colours.copy()
    for key in results.keys():
        mk = mks.pop()
        cl = cls.pop()
        times, fvals, x_list = results[key]
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
    plt.figure()
    linewidth = 3
    mks = markers.copy()
    cls = colours.copy()
    for key in results.keys():
        mk = mks.pop()
        cl = cls.pop()
        time, fvals, x_list = results[key]
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
        mks = markers.copy()
        cls = colours.copy()
        for key in results.keys():
            mk = mks.pop()
            cl = cls.pop()
            time, fvals, x_list = results[key]
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
