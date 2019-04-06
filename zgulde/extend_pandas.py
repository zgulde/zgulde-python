'''
This module adds functionality to pandas Series and DataFrame objects. The
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
- nnull (nna)
- drop_outliers
- unnest
- get_scalers
- crosstab (xtab)

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
from matplotlib import pyplot as plt
from typing import List, Callable
import operator as op
from functools import reduce
from scipy.stats import ttest_ind

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

def unnest_df(df: DataFrame, col: str, split=True, sep=',', reset_index=True) -> DataFrame:
    '''
    Turns a column with multiple values in each row in it into separate rows,
    each with a single value.

    Parameters
    ----------

    - col: name of the column to unnest
    - split: default True. whether or not to split the data in the column. Set
             this to False if the column already contains lists in each row
    - sep: separator to split on. Ignored if split=False
    - reset_index: default True. whether to reset the index in the resulting
                   data frame. If False, the resulting data frame will have an
                   index that could contain duplicates.

    Examples
    --------

    >>> df = pd.DataFrame(dict(x=list('abc'), y=['a,b,c', 'd,e', 'f']))    
    >>> df
       x      y
    0  a  a,b,c
    1  b    d,e
    2  c      f
    >>> df.unnest('y')
       x  y
    0  a  a
    1  a  b
    2  a  c
    3  b  d
    4  b  e
    5  c  f
    '''
    s = df[col].str.split(sep) if split else df[col]
    s = s.apply(pd.Series)\
        .stack()\
        .reset_index(level=1, drop=True)

    s.name = col

    return df.drop(columns=[col]).join(s)\
        .pipe(lambda df: df.reset_index(drop=True) if reset_index else df)

def correlation_heatmap(df: DataFrame, fancy=False, **kwargs):
    '''
    Plot a heatmap of the correlation matrix for the data frame.

    Any additional kwargs are passed to sns.heatmap
    '''
    if not fancy:
        return sns.heatmap(df.corr(), cmap=cm.PiYG, center=0, annot=True, **kwargs)

    cmat = df.corr()

    sns.set(style="white")
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap

    mask = np.zeros_like(cmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(12, 12))
    sns.heatmap(cmat, mask=mask, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, **kwargs)
    plt.yticks(rotation=0)
    plt.show()

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

def crosstab(df: DataFrame, rows, cols, values=None, **kwargs) -> DataFrame:
    '''
    Shortcut to call to pd.crosstab.

    Parameters
    ----------

    - rows: the name of the columns that will make up the rows in resulting
            contingency table
    - cols: the name of the columns that will make up the columns in resulting
            contingency table
    - values: (optional) name of the column to use for the cell values in the
              resulting contingency table. If supplied, aggfunc must be provided
              as well. See pd.crosstab for more details.
    
    Examples
    --------

    >>> df = pd.DataFrame(dict(x=list('aaabbb'), y=list('cdcdcd'), z=range(6)))
    >>> df
       x  y  z
    0  a  c  0
    1  a  d  1
    2  a  c  2
    3  b  d  3
    4  b  c  4
    5  b  d  5
    >>> df.crosstab('x', 'y')
    y  c  d
    x      
    a  2  1
    b  1  2
    >>> (df.crosstab('x', 'y') == pd.crosstab(df.x, df.y)).all(axis=None)
    True
    >>> df.crosstab('x', 'y', margins=True)
    y    c  d  All
    x             
    a    2  1    3
    b    1  2    3
    All  3  3    6
    >>> df.xtab(rows='x', cols='y', values='z', aggfunc='mean')
    y  c  d
    x      
    a  1  1
    b  4  4
    '''
    if values is not None:
        kwargs['values'] = df[values]
    return pd.crosstab(df[rows], df[cols], **kwargs)

def ttest(df: DataFrame, target: str) -> DataFrame:
    '''
    Runs ttests for target for every unique value from every column in the data
    frame.

    The resulting t-statistic and pvalue are based on subdividing the data for
    each unique value for each column, with each individual value indicating
    that the test was performed based on belonging to that unique value vs not
    belonging to that group.

    Examples
    --------

    >>> from seaborn import load_dataset
    >>> tips = load_dataset('tips')
    >>> tips = tips[['total_bill', 'day', 'time']]
    >>> tips.ttest('total_bill')
                     statistic    pvalue    n
    variable value                           
    day      Sun      1.927317  0.055111   76
             Sat      0.855634  0.393046   87
             Thur    -2.170294  0.030958   62
             Fri     -1.345462  0.179735   19
    time     Dinner   2.897638  0.004105  176
             Lunch   -2.897638  0.004105   68
    '''
    results = []
    for col in df.drop(columns=target):
        unique_vals = df[col].unique()
        ttests = DataFrame([ttest_ind(df[df[col] == v][target],
                                      df[df[col] != v][target]) 
                            for v in unique_vals])
        ns = [df[df[col] == v].shape[0] for v in unique_vals]
        ttests = ttests.assign(n=ns, value=unique_vals, variable=col)
        results.append(ttests)
    return pd.concat(results, axis=0).set_index(['variable', 'value'])

def pipe(df: DataFrame, fn: Callable):
    return df.pipe(fn)

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
pd.DataFrame.unnest = unnest_df
pd.DataFrame.crosstab = crosstab
pd.DataFrame.xtab = crosstab
pd.DataFrame.ttest = ttest
