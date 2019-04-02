'''
This module adds functionaliry to pandas Series and DataFrame objects. The
objects in pandas will be modified by simply importing this module.

The following methods are added to all Series:

- zscore
- outliers
- log
- log2
- ln
- get_scaler

and the following are added to all DataFrames

- correlation_heatmap
- nnull
- nnull
- drop_outliers

See the documentation for the individual methods for more details
(e.g. ``help(pd.Series.outliers)``)

>>> import pandas as pd
>>> import numpy as np
'''

# More general summary function

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
from matplotlib.pyplot import cm
from typing import List, Callable
import operator as op
from functools import reduce

def get_scalers(df: DataFrame, columns, **kwargs) -> Callable:
    '''
    Obtain a function that will scale multiple columns on a data frame.

    The returned function accepts a data frame and returns the data frame with
    the specified column(s) scaled.

    This can be useful to make sure you apply the same transformation to both
    training and test data sets.

    See the docstring for Series.get_scaler for more details.

    Parameters
    ----------

    - columns : Either a single string, or a list of strings where each string
                is a column name to scale
    - kwargs : any additional arguments are passed to Series.get_scaler for each
      column specified

    Example
    -------

    >>> df = pd.DataFrame(dict(x=[1, 2, 3, 10], y=[-10, 1, 1, 2]))
    >>> df
        x   y
    0   1 -10
    1   2   1
    2   3   1
    3  10   2
    >>> scale_x = df.get_scalers('x', how='minmax')
    >>> scale_x(df)
              x   y
    0  0.000000 -10
    1  0.111111   1
    2  0.222222   1
    3  1.000000   2
    >>> scale_x_and_y = df.get_scalers(['x', 'y'])
    >>> scale_x_and_y(df)
              x         y
    0 -0.734847 -1.494836
    1 -0.489898  0.439658
    2 -0.244949  0.439658
    3  1.469694  0.615521
    '''
    # allow either a single string or a list of strings
    if type(columns) is str:
        columns = [columns]
    scalers = [df[col].get_scaler(**kwargs) for col in columns]
    return lambda df: reduce(lambda df, f: df.pipe(f), scalers, df)

def get_scaler(s: Series, how='zscore'):
    '''
    Obtain a function that will scale the series on a data frame.

    The returned function accepts a data frame and returns the data frame with
    the specified column scaled.

    This can be useful to make sure you apply the same transformation to both
    training and test data sets.

    - zscore = (x - mu) / sigma
    - minmax = (x - min) / (max - min)

    Parameters
    ----------

    - how : One of {'zscore', 'minmax'} to either apply z-score or min-max
      normalization

    Example
    -------

    >>> df = pd.DataFrame(dict(x=[1, 2, 3, 4, 5, 1000], y=[1000, 2, 3, 4, 5, 6]))
    >>> scale_x = df.x.get_scaler()
    >>> scale_x(df)
              x     y
    0 -0.413160  1000
    1 -0.410703     2
    2 -0.408246     3
    3 -0.405789     4
    4 -0.403332     5
    5  2.041229     6
    >>> scale_y = df.y.get_scaler('minmax')
    >>> scale_y(df)
          x         y
    0     1  1.000000
    1     2  0.000000
    2     3  0.001002
    3     4  0.002004
    4     5  0.003006
    5  1000  0.004008
    >>> df.pipe(scale_x).pipe(scale_y)
              x         y
    0 -0.413160  1.000000
    1 -0.410703  0.000000
    2 -0.408246  0.001002
    3 -0.405789  0.002004
    4 -0.403332  0.003006
    5  2.041229  0.004008
    '''
    name = s.name
    if how == 'zscore':
        mu = s.mean()
        sigma = s.std()
        def scale(df: DataFrame) -> DataFrame:
            scaled_series = (df[name] - mu) / sigma
            kwargs = {name: scaled_series}
            return df.assign(**kwargs)
        return scale
    elif how == 'minmax':
        min = s.min()
        max = s.max()
        def scale(df: DataFrame) -> DataFrame:
            scaled_series = (df[name] - min) / (max - min)
            kwargs = {name: scaled_series}
            return df.assign(**kwargs)
        return scale
    raise ValueError('how must be one of {zscore, minmax}')

def zscore(s: Series) -> Series:
    '''
    Returns the z-score for every value in the series.

    Z = (x - mu) / sigma

    Example
    -------

    >>> x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> x
    0    1
    1    2
    2    3
    3    4
    4    5
    5    6
    6    7
    7    8
    8    9
    dtype: int64
    >>> x.zscore()
    0   -1.460593
    1   -1.095445
    2   -0.730297
    3   -0.365148
    4    0.000000
    5    0.365148
    6    0.730297
    7    1.095445
    8    1.460593
    dtype: float64
    '''
    return (s - s.mean()) / s.std()

def drop_outliers(df: DataFrame, *cols: List[str], **kwargs) -> Series:
    '''
    Drop rows with outliers in the given columns from the dataframe

    See the docs for .outliers for more details on parameters, and to customize
    how the outliers are detected.

    Examples
    --------
    >>> df = pd.DataFrame(dict(x=[1, 2, 3, 4, 5, 1000], y=[1000, 2, 3, 4, 5, 6]))
    >>> df
          x     y
    0     1  1000
    1     2     2
    2     3     3
    3     4     4
    4     5     5
    5  1000     6
    >>> df.drop_outliers('x')
       x     y
    0  1  1000
    1  2     2
    2  3     3
    3  4     4
    4  5     5
    >>> df.drop_outliers('y')
          x  y
    1     2  2
    2     3  3
    3     4  4
    4     5  5
    5  1000  6
    >>> df.drop_outliers('x', 'y')
       x  y
    1  2  2
    2  3  3
    3  4  4
    4  5  5
    '''
    to_keep = [~ df[col].outliers(**kwargs) for col in cols]
    return df[list(reduce(op.and_, to_keep))]

def outliers(s: Series, how='iqr', k=1.5, std_cutoff=2) -> Series:
    '''
    Detect outliers in the series.

    Parameters
    ----------

    how : {'iqr', 'std'}, default 'iqr'
        - 'iqr' : identify outliers based on whether they are > q3 + k * iqr
          or < q1 - k * iqr
        - 'std' : identify outliers based on whether they are further than 2
          standard deviations from the mean
    k : value to multiply the iqr by for outlier detection. Ignored when
        how='std'. Default 1.5
    std_cutoff : cutoff for identifying an outlier based on standard deviation.
                 Ignored when how='iqr'. Default 2

    Examples
    --------

    >>> x = pd.Series([1, 2, 3, 4, 5, 6, 100])
    >>> x.outliers()
    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool
    '''
    if how == 'iqr':
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - k * iqr) | (s > q3 + k * iqr)
    elif how == 'std':
        return zscore(s).abs() > std_cutoff
    raise ValueError('how must be one of {iqr,std}')

def nnull(df: DataFrame, percent=True) -> Series:
    '''
    Provide a summary of null values in each column

    alias of nna

    Parameters
    ----------

    - percent: whether to show the number of null values as a percent of the
               total. Default True

    Examples
    --------

    >>> df = pd.DataFrame(dict(x=[1, 2, np.nan], y=[4, np.nan, np.nan]))
    >>> df
         x    y
    0  1.0  4.0
    1  2.0  NaN
    2  NaN  NaN
    >>> df.nnull()
    x    0.333333
    y    0.666667
    dtype: float64
    >>> df.nnull(percent=False)
    x    1
    y    2
    dtype: int64
    '''
    nulls = df.isnull().sum()
    if percent:
        return nulls / df.shape[0]
    else:
        return nulls

def correlation_heatmap(df: DataFrame, **kwargs):
    '''
    Plot a heatmap of the correlation matrix for the data frame.

    Any additional kwargs are passed to sns.heatmap
    '''
    return sns.heatmap(df.corr(), cmap=cm.PiYG, center=0, annot=True, **kwargs)

def log(s: Series):
    '''
    Returns the log base 10 of the values in the series using np.log10

    Example
    -------

    >>> pd.Series([1, 10, 100, 1000]).log()
    0    0.0
    1    1.0
    2    2.0
    3    3.0
    dtype: float64
    '''
    return np.log10(s)

def ln(s: Series):
    '''
    Returns the natural log of the values in the series using np.log

    >>> pd.Series([1, np.e, np.e ** 2, np.e ** 3]).ln()
    0    0.0
    1    1.0
    2    2.0
    3    3.0
    dtype: float64
    '''
    return np.log(s)

def log2(s: Series):
    '''
    Returns the log base 2 of the values in the series using np.log2

    Example
    -------

    >>> pd.Series([1, 2,4, 8, 16]).log2()
    0    0.0
    1    1.0
    2    2.0
    3    3.0
    4    4.0
    dtype: float64
    '''
    return np.log2(s)

def pipe(df: DataFrame, fn: Callable):
    return df.pipe(fn)

extensions = [correlation_heatmap, nnull, nnull, drop_outliers, zscore, outliers, log, log2, ln, get_scaler]

pd.Series.zscore = zscore
pd.Series.outliers = outliers
pd.Series.get_scaler = get_scaler
pd.Series.log = log
pd.Series.log2 = log2
pd.Series.ln = ln

pd.DataFrame.correlation_heatmap = correlation_heatmap
pd.DataFrame.nnull = nnull
pd.DataFrame.nna = nnull
pd.DataFrame.drop_outliers = drop_outliers
pd.DataFrame.get_scalers = get_scalers
pd.DataFrame.__lshift__ = pipe
pd.DataFrame.__rshift__ = pipe
