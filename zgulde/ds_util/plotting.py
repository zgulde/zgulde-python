import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pydataset import data

import zgulde.extend_pandas

# TODO: refactor some of these to iterate over .groupby instead of unique values

# plotting style defaults -- plt.style.use(style) -- mpl.style.use(style)
style = {
    "animation.html": "html5",
    "axes.facecolor": "#FEFEFE",
    "axes.grid": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": "#FEFEFE",
    "figure.figsize": (16, 9),
    "font.size": 13.0,
    "grid.alpha": 0.7,
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "hist.bins": 25,
    "patch.edgecolor": "black",
    "patch.facecolor": "firebrick",
    "patch.force_edgecolor": True,
}


def dual_axis(df: pd.DataFrame, x: str) -> Callable:
    """
    >>> mpg = data("mpg")
    >>> dual_axis(mpg, "displ")("hwy")("cty")
    <Figure size ... with 2 Axes>
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    def plot_y2(y, *args, **kwargs):
        ax2.plot(df[x], df[y], *args, **kwargs)
        return fig

    def plot_y1(y, *args, **kwargs):
        ax1.plot(df[x], df[y], *args, **kwargs)
        return plot_y2

    return plot_y1


def bar_dual_y(
    df: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    aggfunc="mean",
    space=0.1,
    ax=None,
    colors=["lightblue", "orange"],
):
    g = df[[x, y1, y2]].groupby(x).agg(aggfunc)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1_xticks = [x - (0.5 - space) for x in range(len(g.index))]

    ax1.bar(
        ax1_xticks, g[y1], width=(0.5 - space), align="edge", color=colors[0], label=y1
    )
    ax2.bar(
        g.index, g[y2], width=(0.5 - space), align="edge", label=y2, color=colors[1]
    )

    ax1.set_ylabel(y1)
    ax2.set_ylabel(y2, rotation=270, labelpad=14)
    ax1.set_xlabel(x)
    ax1.set(title=f"{aggfunc} of {y1} and {y2} by {x}")

    fig.legend()

    return fig, (ax1, ax2)


def plot_3d(df, x, y, z, g):
    fig = plt.figure()
    ax = Axes3D(fig)
    for v in sorted(df[g].unique()):
        subset = df[df[g] == v]
        ax.scatter(subset[x], subset[y], subset[z], label=v)
    ax.set(xlabel=x, ylabel=y, zlabel=z)
    ax.legend(title=g)


def hist_by_group(x: pd.Series, g: pd.Series, *args, **kwargs):
    """
    >>> mpg = data('mpg')
    >>> hist_by_group(mpg.hwy, mpg.cyl)
    (<Figure size ... with 4 Axes>, array([[<matplotlib.axes._subplots.AxesSubplot object at ...>,
            <matplotlib.axes._subplots.AxesSubplot object at ...>],
           [<matplotlib.axes._subplots.AxesSubplot object at ...>,
            <matplotlib.axes._subplots.AxesSubplot object at ...>]],
          dtype=object))
    """
    fig, axs = plt.subplots(2, 2) if g.nunique() > 2 else plt.subplots(1, 2)
    fig.suptitle(f"Distribution of {x.name} by {g.name}")
    g = g.top_n(3)
    g.index = x.index

    for ax, v in zip(axs.ravel(), g.unique()):
        x_g = x[g == v]

        ax.hist(x_g, color="pink", *args, **kwargs)
        ax.set_title(v)

        # mean + ci
        xbar = x_g.mean()
        z = 1.96  # 95% ci
        ci = z * (x_g.std() / math.sqrt(x_g.shape[0]))
        ub, lb = xbar + ci, xbar - ci

        ymin, ymax = ax.get_ylim()
        ax.vlines(xbar, ymin, ymax, ls="--", color="gray")
        ax.vlines([lb, ub], ymin, ymax, ls=":", color="gray")

    return fig, axs


def bar_by_group(x: pd.Series, g: pd.Series, aggfunc="mean", *args, **kwargs):
    """
    >>> mpg = data('mpg')
    >>> bar_by_group(mpg.hwy, mpg['class'])
    (<Figure size ... with 1 Axes>, <matplotlib.axes._subplots.AxesSubplot object at ...>)
    """
    g = g.top_n(3)
    fig, ax = plt.subplots()
    x.groupby(g).agg(aggfunc).plot.bar(ax=ax, color="pink", width=1)
    ax.set(title=f"{aggfunc} of {x.name} by {g.name}")
    if aggfunc == "mean":
        xbar = x.agg(aggfunc)
        z = 2.58  # 99% ci
        ci = z * (x.std() / math.sqrt(x.shape[0]))
        ub, lb = xbar + ci, xbar - ci
        ax.hlines(xbar, -0.5, 3.5, ls="--", color="gray")
        ax.hlines([lb, ub], -0.5, 3.5, ls=":", color="gray")
    return fig, ax


def scatter_by_group(df: pd.DataFrame, x: str, y: str, g: str):
    for name, subset in df.groupby(g):
        plt.scatter(subset[x], subset[y], label=name)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=g)


def group_proportions(df: pd.DataFrame, x1: str, x2: str, proportions=False, ax=None):
    """
    Visualize the proportion of each group in x2 for each unique group in x1.

    x1 and x2 should both be categorical variables.
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    (
        df.groupby(x1)[x2]
        .apply(pd.Series.value_counts, normalize=proportions)
        .unstack()
        .plot.bar(stacked=True, width=0.9, ax=ax)
    )
    ax.set(ylabel="proportion" if proportions else "count")
    ax.legend(title=x2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    return ax


def crosstab_heatmap(
    x: pd.Series,
    y: pd.Series,
    ax=None,
    values=None,
    aggfunc=None,
    normalize=False,
    cmap="Purples",
    fmt=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    if fmt is None:
        fmt = ".2f" if normalize or aggfunc is not None else "d"

    ctab = pd.crosstab(x, y, values=values, aggfunc=aggfunc)
    sns.heatmap(ctab, annot=True, cmap=cmap, fmt=fmt)
    return ax


def crosstab_scatter(
    x: pd.Series, y: pd.Series, ax=None, scale=1000, values=None, aggfunc=None
):
    """
    Visualize the crosstabulation of x and y with a scatter plot where the size
    of the points corresponds to the value of that cell in the crosstabulation.

    For an aggregation other than counting, values and aggfunc can be passed.
    Internally, x, y, values, and aggfunc are all passed to pd.crosstab.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ctab = pd.crosstab(x, y, values=values, aggfunc=aggfunc)
    ctab = ctab.reset_index().melt(id_vars=x.name)
    # min-max scale -- 0 to `scale`
    ctab.value = ((ctab.value - ctab.value.min()) * scale) / (
        ctab.value.max() - ctab.value.min()
    )

    ctab["x_rank"] = ctab[x.name].rank(method="dense")
    ctab["y_rank"] = ctab[y.name].rank(method="dense")

    ax.scatter(ctab.x_rank, ctab.y_rank, s=ctab.value)
    ax.vlines(ctab.x_rank, *ax.get_ylim(), ls="--", color="grey", lw=0.8)
    ax.hlines(ctab.y_rank, *ax.get_xlim(), ls="--", color="grey", lw=0.8)

    ax.set(xticks=ctab.x_rank.unique(), xticklabels=ctab[x.name].unique())
    ax.set(yticks=ctab.y_rank.unique(), yticklabels=ctab[y.name].unique())
    ax.set(xlabel=x.name, ylabel=y.name)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


# WIP below
arrowstyles = [
    "-",
    "->",
    "-[",
    "|-|",
    "-|>",
    "<-",
    "<->",
    "<|-",
    "<|-|>",
    "fancy",
    "simple",
    "wedge",
]


def _annotate(s, xy, xytext, style="->"):
    return plt.annotate(s, xy=xy, xytext=xytext, arrowprops={"arrowstyle": style})


def _arrow(fromxy, toxy, style="->"):
    plt.annotate(
        "",
        xy=(11, 0.07),
        xytext=(30, 0.07),
        xycoords="data",
        textcoords="data",
        arrowprops={"arrowstyle": "|-|"},
    )
