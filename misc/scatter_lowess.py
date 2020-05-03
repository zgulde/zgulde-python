import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.ion()


def scatter_lowess(
    df: pd.DataFrame, x: str, y: str, g=None, ax=None, scatter_kwargs={}, plot_kwargs={}
):
    if ax is None:
        ax = plt.gca()
    if g is not None:
        for group, subset in df.groupby(g):
            ax.scatter(subset[x], subset[y], label=group, **scatter_kwargs)
            ax.plot(
                *by_col(lowess(subset[y], subset[x], return_sorted=True)), **plot_kwargs
            )
        ax.legend(title=g)
    else:
        ax.scatter(df[x], df[y], **scatter_kwargs)
        ax.plot(
            *by_col(lowess(df[y], df[x], return_sorted=True)),
            **{"c": "black", **plot_kwargs}
        )
    return ax


def by_col(m):
    for i in range(m.shape[1]):
        yield m[:, i]

import pydataset

mpg = pydataset.data("mpg")
tips = pydataset.data('tips')

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
for ax, (sex, subset) in zip(axs, tips.groupby("sex")):
    scatter_lowess(
        subset,
        "total_bill",
        "tip",
        "smoker",
        scatter_kwargs={"ec": "black", "alpha": 0.6},
        ax=ax,
    )
    ax.set(title=sex)

tips.groupby(['size', 'sex']).size().rename('count')
