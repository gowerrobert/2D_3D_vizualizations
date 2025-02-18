import requests
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from PIL import Image


cmap = [(0, "#2f9599"), (0.45, "#eeeeee"), (1, "#8800ff")]
cmap = cm.colors.LinearSegmentedColormap.from_list("Custom", cmap, N=256)

title_fs = 24 
axis_fs = 22 


def latex_img(latex):
    base_url = r"https://latex.codecogs.com/png.latex?\dpi{400}\\"
    url = f"{base_url}{latex}"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def plot_2d(
    function,
    n_space=1000,
    cmap=cmap,
    XYZ=None,
    fig=None,
    show=True,
    logscale=False,
):
    X_domain, Y_domain = function.input_domain
    if XYZ is None:
        X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
        X, Y = np.meshgrid(X, Y)
        XY = np.array([X, Y])
        Z = np.apply_along_axis(function, 0, XY)
    else:
        X, Y, Z = XYZ

    # create new ax if None
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = plt.gca()

    # add contours and contours lines
    # CS3 = ax.contour(X, Y, Z, levels=30, linewidths=0.5, colors='#999')
    if logscale:
        CS3 = ax.contourf(
            X,
            Y,
            Z,
            levels=30,
            cmap=cmap,
            alpha=0.7,
            locator=ticker.LogLocator(),
        )
    else:
        CS3 = ax.contourf(
            X,
            Y,
            Z,
            levels=30,
            cmap=cmap,
            alpha=0.7,
        )
    plt.xticks([])
    plt.yticks([])

    cbar = fig.colorbar(CS3)
    cbar.set_ticks([])
    cbar.ax.set_ylabel("$f(w)$", fontsize=axis_fs-2)
    plt.title(function.name, fontsize=title_fs)

    # add labels and set equal aspect ratio
    ax.set_xlabel("$x$", fontsize=axis_fs)
    ax.set_ylabel("$y$", fontsize=axis_fs)
    # ax.set_aspect(aspect="equal")
    if show:
        plt.show()


def plot_3d(function, n_space=1000, cmap=cmap, XYZ=None, ax=None, show=True):
    X_domain, Y_domain = function.input_domain
    if XYZ is None:
        X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
        X, Y = np.meshgrid(X, Y)
        XY = np.array([X, Y])
        Z = np.apply_along_axis(function, 0, XY)
    else:
        X, Y, Z = XYZ

    # create new ax if None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plot the surface.
    ax.plot_surface(
        X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.7
    )
    CS3 = ax.contour(X, Y, Z, zdir="z", levels=30, offset=np.min(Z), cmap=cmap)
    cbar = fig.colorbar(CS3)
    cbar.ax.set_ylabel("function value")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    if show:
        plt.show()
