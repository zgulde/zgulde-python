import math
from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: refactor some of these to iterate over .groupby instead of unique values


def plot_dual_axis(df: pd.DataFrame, x: str) -> Callable:
    """
    plot_dual_axis(df, 'x')('y1')('y2')
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


def plot_3d(df, x, y, z, g):
    fig = plt.figure()
    ax = Axes3D(fig)
    for v in sorted(df[g].unique()):
        subset = df[df[g] == v]
        ax.scatter(subset[x], subset[y], subset[z], label=v)
    ax.set(xlabel=x, ylabel=y, zlabel=z)
    ax.legend(title=g)


def plot_dual_axis(df: pd.DataFrame, x: str):
    "plot_dual_axis(df, 'x')('y1', marker='x')('y2', color='firebrick')"
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    def plot_y2(y, *args, **kwargs):
        ax2.plot(df[x], df[y], *args, **kwargs)
        return fig

    def plot_y1(y, *args, **kwargs):
        ax1.plot(df[x], df[y], *args, **kwargs)
        return plot_y2

    return plot_y1


def top_n(s: pd.Series, n=3, other_val="Other"):
    top_n = s.value_counts().index[:n]
    return pd.Series(np.where(s.isin(top_n), s, other_val), name=s.name)


def plot_hist_by_group(x: pd.Series, g: pd.Series, *args, **kwargs):
    """
    >>> mpg = data('mpg')
    >>> hist_by_group(mpg.hwy, mpg.cyl)
    """
    fig, axs = plt.subplots(2, 2) if g.nunique() > 2 else plt.subplots(1, 2)
    fig.suptitle(f"Distribution of {x.name} by {g.name}")
    g = top_n(g, 3)
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


def plot_bar_by_group(x: pd.Series, g: pd.Series, aggfunc="mean", *args, **kwargs):
    """
    >>> mpg = data('mpg')
    >>> plot_bar_by_group(mpg.hwy, mpg['class'])
    """
    g = top_n(g, 3)
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
    else:
        ax.hlines(x.agg(aggfunc), -0.5, 3.5, ls="--", color="gray")
    return fig, ax


def plot_scatter_by_group(df: pd.DataFrame, x: str, y: str, g: str):
    for name, subset in df.groupby(g):
        plt.scatter(subset[x], subset[y], label=name)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=g)


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


def annotate(s, xy, xytext, style="->"):
    return plt.annotate(s, xy=xy, xytext=xytext, arrowprops={"arrowstyle": style})


def arrow(fromxy, toxy, style="->"):
    plt.annotate(
        "",
        xy=(11, 0.07),
        xytext=(30, 0.07),
        xycoords="data",
        textcoords="data",
        arrowprops={"arrowstyle": "|-|"},
    )
