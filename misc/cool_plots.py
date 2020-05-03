import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset

from zgulde.ds_util.plotting import style

plt.ion()
plt.style.use(style)


def _get_ellipses(x: pd.Series, y: pd.Series, how, **kwargs):
    if not "alpha" in kwargs:
        kwargs["alpha"] = 0.1
    if how == "mean":
        center = (x.mean(), y.mean())
        s_x = x.std()
        s_y = y.std()
        widths = [2 * s_x, 4 * s_x, 6 * s_x]
        heights = [2 * s_y, 4 * s_y, 6 * s_y]
    elif how == "median":
        center = (x.median(), y.median())
        iqr_x = x.quantile(0.75) - x.quantile(0.25)
        iqr_y = y.quantile(0.75) - y.quantile(0.25)
        widths = [iqr_x, 1.5 * iqr_x, 3 * iqr_x]
        heights = [iqr_y, 1.5 * iqr_y, 3 * iqr_y]
    else:
        raise ValueError("`how` must be one of {mean,median}, got %s" % how)
    return [
        mpl.patches.Ellipse(center, widths[0], heights[0], **kwargs),
        mpl.patches.Ellipse(center, widths[1], heights[1], **kwargs),
        mpl.patches.Ellipse(center, widths[2], heights[2], **kwargs),
    ]


def scatter_by_group(df: pd.DataFrame, x: str, y: str, g: str, ax=None, how="mean"):
    """
    Shows a scatter plot of an x and y value colored by group, along with a
    visual representation of the midpoint and spread of each in the form of
    several ellipses.

    By default, widths of the ellipses are 2, 4, and 6 times the standard
    deviation.  This way a point on the edge of the circle is 1, 2, or 3
    standard deviations away from the mean. Similarly, the heights is 2, 4, and
    6 times the standard deviatino of the y variable.

    Instead of the mean and standard deviation, the ``how`` keyword argument can
    be supplied with a value of `median`, and then the median and IQR will be
    used.  Here the widths and heights will be the IQR, 1.5 * IQR and 3 * IQR
    for the x and y variables, respectively.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for name, subset in df.groupby(g):
        points = ax.scatter(subset[x], subset[y], label=name)
        ellipses = _get_ellipses(subset[x], subset[y], how, color=points.get_fc()[0])
        for patch in ellipses:
            ax.add_patch(patch)
    ax.set(xlabel=x, ylabel=y)
    ax.legend(title=g)


# TODO: better control over colors and labels
# TODO: we could probably use this for kmeans cluster prediction/visualization
# too
def decision_boundaries(clf, x, y, target, n_points=250, ax=None):
    """
    Visualize decision boundaries for the given classifier.

    Only works for classification models trained on two features.

    x and y could be the features the model was trained on, or unseed datapoints
    in the same space.

    Parameters
    ----------

    - ``clf``: a fit classification model
    - ``x`` and ``y``
    - ``target``: should be pre label-encoded
    """
    if ax is None:
        ax = plt.gca()
    xmin = x.min() - 0.1 * (x.max() - x.min())
    ymin = y.min() - 0.1 * (y.max() - y.min())
    xmax = x.max() + 0.1 * (x.max() - x.min())
    ymax = y.max() + 0.1 * (y.max() - y.min())
    #
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #
    ax.contourf(xx, yy, Z, alpha=0.2)
    scatter = ax.scatter(x, y, c=target, ec="black")
    return ax
