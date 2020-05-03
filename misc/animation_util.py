"""
Module full of various helpers for creating matplotlib animations.
"""

import numpy as np
import pandas as pd
import pytweening
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# In order to get the final points, there's 2 high-level steps:
#
# 1. Generate points between x1 and x2 that are spaced evenly
# 2. Use pytweening to transform the generated points such that they match the
#    supplied timing function.
#
# To make step 2 happen, we have to go about it in complicated way (there might
# be a better, more clever way to do this).
#
# The ``pytweening`` module is nice, but only works on values between 0 and 1.
# We'll transform the generated evenly spaced points such that they have a min
# of 0 and max of 1, then apply the pytweening function, then transform the new
# points back to their original scale.
# We'll use sklearn's MinMaxScaler for this process.
def tween(x1, x2, fn, n_points=50):
    """
    Generate intermediate points between x1 and x2 based on a specified tweening
    function.

    x1 and x2 may be either single scaler values or 1-d numpy arrays of the same
    shape.

    Parameters
    ----------

    - x1: a scaler value or a 1-d array (numpy array or pandas series)
    - x2: a scaler value or a 1-d array (numpy array or pandas series)
    - fn: a timing function from pytweening
    - n_points: the number of points to generate, including the starting and
      stopping points.

    Returns
    -------

    If x1 and x2 are scalers, a 1-d array with n_points elements where the first
    element is x1 and the last is x2.

    If x1 and x2 are 1-d arrays, a matrix of shape (n_points, x1.size). Each row
    in the matrix is the data points at one step. The first row is x1 and the
    last row is x2.

    Examples
    --------

    >>> import pytweening
    >>> tween(1, 10, pytweening.linear, 10)
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    >>> tween(1, 10, pytweening.easeInQuad, 10)
    array([ 1.        ,  1.11111111,  1.44444444,  2.        ,  2.77777778,
            3.77777778,  5.        ,  6.44444444,  8.11111111, 10.        ])
    >>> tween(1, 10, pytweening.easeOutQuad, 10)
    array([ 1.        ,  2.88888889,  4.55555556,  6.        ,  7.22222222,
            8.22222222,  9.        ,  9.55555556,  9.88888889, 10.        ])
    >>> x1 = np.array([1, 1, 2])
    >>> x2 = np.array([2, 5, 4])
    >>> tween(x1, x2, pytweening.linear, 5)
    array([[1.  , 1.  , 2.  ],
           [1.25, 2.  , 2.5 ],
           [1.5 , 3.  , 3.  ],
           [1.75, 4.  , 3.5 ],
           [2.  , 5.  , 4.  ]])
    """
    # handle the case where we have scaler values first
    if np.isscalar(x1) and np.isscalar(x2):
        if x1 == x2:
            return np.repeat(x1, n_points)
        xx = np.linspace(x1, x2, n_points).reshape(-1, 1)
        scaler = MinMaxScaler().fit(xx)
        linspace = (
            np.linspace(0, 1, n_points) if x1 < x2 else np.linspace(1, 0, n_points)
        )
        return scaler.inverse_transform(
            np.array([fn(x) for x in linspace]).reshape(-1, 1)
        ).ravel()

    # sanity check arguments
    if len(x1.shape) != 1 or len(x2.shape) != 1 or x1.shape[0] != x2.shape[0]:
        raise ValueError(
            "x1 and x2 must be either scaler values or 1-d numpy arrays of the same shape"
        )

    xx_linear = np.linspace(x1, x2, n_points)
    scaler = MinMaxScaler().fit(xx_linear)
    xx_minmax = scaler.transform(xx_linear)
    # because rounding, sometimes we end up w/ numbers like 1.0000000002
    xx_minmax = np.where(xx_minmax > 1, 1, xx_minmax)
    xx_minmax_t = pd.DataFrame(xx_minmax).apply(lambda col: [fn(x) for x in col]).values
    return scaler.inverse_transform(xx_minmax_t)


# @anim
# def scale_xaxis(i):
#     if i == 0:
#         ax.set(title="Center x by subtracting the mean\n$x' = x - \\mu_x$")
#     xx = mytween(ax.get_xlim()[0], x2.min() - .5)
#     ax.set(xlim=(xx[i], ax.get_xlim()[1]))

# anim.transform(x, x2, y, y)
# anim.pause()

# @anim
# def scale_yaxis(i):
#     if i == 0:
#         ax.set(title="Center y by subtracting the mean\n$y' = y - \\mu_y$")
#     yy = mytween(ax.get_ylim()[0], y2.min() - .5)
#     ax.set(ylim=(yy[i], ax.get_ylim()[1]))


class Animation:
    """
    Helper class for creating matplotlib animations.

    After creating an instance, it can be used as a decorator to add a function
    to the animation.

    The main benefit of using this class is that each function in the animation
    can be reasoned about in isolation. Each function will be passed a single
    integer that ranges from 0 to fps, so that the function just has to worry
    about drawing each frame, based on the passed argument.

    Also contains a handful of helper methods for common animation needs.

    >>> fig, ax = plt.subplots()
    >>> lines, = ax.plot([], [], marker='o', ls='')
    >>> anim = Animation(fig, fps=12, lines=lines)
    >>> x1, x2 = np.random.rand(100), np.random.rand(100)
    >>> y1, y2 = np.random.rand(100), np.random.rand(100)
    >>> anim.show_points(x1, x2)
    >>> anim.pause()
    >>> anim.transform(x1, x2, y1, y2)
    >>> @anim
    ... def custom_animation_fn(i):
    ...     title = 'My Animation'
    ...     ax.set(title=title[:i])
    >>> anim.animate()
    <matplotlib.animation.FuncAnimation object at ...>
    """

    def __init__(self, fig, fps=24, lines=None, ax=None, timing_fn=None):
        self._fps = fps
        self._fig = fig
        self._fns = []
        self._lines = lines
        self._ax = ax
        self._timing_fn = timing_fn or pytweening.easeInQuad

    def __call__(self, fn):
        self._fns.append(fn)
        return fn

    def pause(self):
        "Add a 1-second pause to the animation."
        self._fns.append(None)

    def show_points(self, x, y, lines=None, title=None):
        "Animate the appearance of x and y points."
        if lines is None:
            lines = self._lines

        def fn(i):
            if i == 0 and title is not None and self._ax is not None:
                self._ax.set(title=title)
            n = x.shape[0]
            i = round(i * (n / self._fps))
            lines.set_data(x[:i], y[:i])

        self._fns.append(fn)

    def transform(self, x1, x2, y1, y2, lines=None, title=None):
        "Animate the transformation from x=x1, y=y1 to x=x2, y=y2"
        if lines is None:
            lines = self._lines
        xx = tween(x1, x2, self._timing_fn, self._fps)
        yy = tween(y1, y2, self._timing_fn, self._fps)

        def fn(i):
            lines.set_data(xx[i], yy[i])
            if title is not None and self._ax is not None:
                ax.set(title=title)

        self._fns.append(fn)

    def scale_axis(self, xmin, xmax, ymin, ymax, ax=None, title=None):
        "Animate changing the axis limits"
        if ax is None:
            ax = self._ax

        # If we set these outside the function, the limits of ax will be
        # captured when scale_axis is called, *not* when the animation function
        # is called. For example, if a previous animation changed the axis
        # scale, we'd still be looking at the original axis limits, not the
        # transformed ones when we started the animation.
        #
        # To workaround this, we'll initialize the variables here, and set them
        # the first time the animation function is called.
        xmin_i = None
        xmax_i = None
        ymin_i = None
        ymax_i = None

        def fn(i):
            nonlocal xmin_i, xmax_i, ymin_i, ymax_i
            if i == 0:
                if title is not None:
                    ax.set(title=title)
                xmin_i = tween(ax.get_xlim()[0], xmin, self._timing_fn, self._fps)
                xmax_i = tween(ax.get_xlim()[1], xmax, self._timing_fn, self._fps)
                ymin_i = tween(ax.get_ylim()[0], ymin, self._timing_fn, self._fps)
                ymax_i = tween(ax.get_ylim()[1], ymax, self._timing_fn, self._fps)
            ax.set(xlim=(xmin_i[i], xmax_i[i]), ylim=(ymin_i[i], ymax_i[i]))

        self._fns.append(fn)

    def animate(self):
        """
        Put everything together and return the resulting
        matplolib.animation.FuncAnimation object.
        """
        if len(self._fns) == 0:
            raise Exception("No functions to animate!")

        def _animate(i):
            fn = self._fns[i // self._fps]
            if fn is not None:
                fn(i % self._fps)

        return FuncAnimation(
            self._fig,
            _animate,
            interval=1000 / self._fps,
            frames=len(self._fns) * self._fps,
            repeat=True,
        )
