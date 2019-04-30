'''
This module adds functionality to pandas Series and DataFrame objects. The
objects in pandas will be modified by simply importing this module.

>>> import zgulde.extend_pandas

The following methods are added to all Series:

- `cut`_ (bin): put data into bins; shortcut to pd.cut
- `get_scaler`_: obtain a function that scales a series
- `ln`_: natural log
- `log2`_: log base 2
- `log`_: log base 10
- `outliers`_: detect outliers
- `zscore`_: obtain the z-score for every value

and the following are added to all DataFrames:

- `chi2`_: run chi square tests on all column combinations
- `correlation_heatmap`_: plot a heatmap of the correlations
- `crosstab`_ (xtab): shortcut to pd.crosstab
- `drop_outliers`_: remove outliers
- `get_scalers`_: obtain a function that scales multiple columns
- `hdtl`_: look at the head and tail of the data frame
- `nnull`_ (nna): summarize the number of missing values
- `n_outliers`_: summarize the number of outliers in each numeric column
- `ttest`_: run multiple 1 sample t-tests for multiple categories
- `ttest_2samp`_: run multiple 2 sample t-tests for multiple categories
- `unnest`_: handle multiple values in a single cell

It also defines the left and right shift operators to be similar to
``pandas.DataFrame.pipe``. For example:

>>> import pandas as pd
>>> import numpy as np
>>> df = pd.DataFrame(dict(x=np.arange(4)))
>>> df
   x
0  0
1  1
2  2
3  3
>>> create_y = lambda df: df.assign(y=df.x + 1)
>>> df >> create_y
   x  y
0  0  1
1  1  2
2  2  3
3  3  4
>>> # This gives the same results as .pipe
>>> ((df >> create_y) == df.pipe(create_y)).all(axis=None)
True
'''

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from typing import List, Callable, Tuple, Union
import operator as op
from functools import reduce, partial
import itertools as it
from scipy.stats import ttest_ind, chi2_contingency, ttest_1samp
import re

column_name_re = re.compile(r'[^a-zA-Z_0-9]')

def clean_col_name(col: str) -> str:
    col = col.strip().lower().replace(' ', '_').replace('.', '_').replace('-', '_')
    return re.sub(column_name_re, '', col)

def cleanup_column_names(df: DataFrame, inplace=False) -> DataFrame:
    '''
    Returns a data frame with the column names cleaned up. Special characters
    are removed and spaces, dots, and dashes are replaced with underscores.

    Parameters
    ----------

    - inplace : Whether or not to modify the data frame in-place and return
                None.

    Example
    -------

    >>> df = pd.DataFrame({'*Feature& A': [1, 2], ' feature.b  ': [2, 3], 'FEATURE-C': [3, 4]})
    >>> df
       *Feature& A   feature.b    FEATURE-C
    0            1             2          3
    1            2             3          4
    >>> df.cleanup_column_names()
       feature_a  feature_b  feature_c
    0          1          2          3
    1          2          3          4
    >>> df
       *Feature& A   feature.b    FEATURE-C
    0            1             2          3
    1            2             3          4
    >>> df.cleanup_column_names(inplace=True)
    >>> df
       feature_a  feature_b  feature_c
    0          1          2          3
    1          2          3          4
    '''
    return df.rename(mapper=clean_col_name, axis='columns', inplace=inplace)

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
    >>> df.pipe(scale_x_and_y)
              x         y
    0 -0.734847 -1.494836
    1 -0.489898  0.439658
    2 -0.244949  0.439658
    3  1.469694  0.615521
    '''
    if type(columns) is str: # allow either a single string or a list of strings
        columns = [columns]
    scalers = [df[col].get_scaler(**kwargs) for col in columns]
    return partial(reduce, lambda df, f: df.pipe(f), scalers)

def cut(s: Series, *args, **kwargs):
    '''
    Bin series values into discrete intervals.

    Shortcut to pd.cut

    Parameters
    ----------

    - args : positional arguments passed to ``pandas.cut``
    - keyword : keyword arguments passed to ``pandas.cut``

    Example
    -------

    >>> x = pd.Series(range(1, 7))
    >>> x
    0    1
    1    2
    2    3
    3    4
    4    5
    5    6
    dtype: int64
    >>> x.cut(2)
    0    (0.995, 3.5]
    1    (0.995, 3.5]
    2    (0.995, 3.5]
    3      (3.5, 6.0]
    4      (3.5, 6.0]
    5      (3.5, 6.0]
    dtype: category
    Categories (2, interval[float64]): [(0.995, 3.5] < (3.5, 6.0]]
    >>> x.cut(bins=[0, 3, 6])
    0    (0, 3]
    1    (0, 3]
    2    (0, 3]
    3    (3, 6]
    4    (3, 6]
    5    (3, 6]
    dtype: category
    Categories (2, interval[int64]): [(0, 3] < (3, 6]]
    '''
    return pd.cut(s, *args, **kwargs)

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
            scaling

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

def drop_outliers(df: DataFrame, cols: Union[str, List[str]], **kwargs) -> Series:
    '''
    Drop rows with outliers in the given columns from the dataframe

    See the docs for .outliers for more details on parameters, and to customize
    how the outliers are detected.

    Parameters
    ----------

    cols : either a string or a list of strings of which column(s) to drop the
           outliers in
    kwargs : additional key-word arguments passed on to
             ``pandas.Series.outliers``

    Example
    -------

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
    >>> df.drop_outliers(['x', 'y'])
       x  y
    1  2  2
    2  3  3
    3  4  4
    4  5  5
    '''
    if type(cols) is str:
        cols = [cols]
    to_keep = [~ df[col].outliers(**kwargs) for col in cols]
    return df[list(reduce(op.and_, to_keep))]


def n_outliers(df: DataFrame, **kwargs) -> Series:
    '''
    Provide a summary of the number of outliers in each numeric column.

    Parameters
    ----------

    - kwargs : any additional arguments to pass along to
               ``pandas.Series.outliers``

    Returns
    -------

    A ``pandas.DataFrame`` indexed by the column names of the the data frame,
    with columns that indicate the number of outliers and the percentage of
    outliers in each column.

    Example
    -------

    >>> x = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    >>> y = [1, 2, 3, 4, 5, 100, 2, 3, 4, 5]
    >>> z = [1, 2, 3, 4, 5, -100, 2, 3, 100, 5]
    >>> df = pd.DataFrame(dict(x=x, y=y, z=z))
    >>> df
       x    y    z
    0  1    1    1
    1  2    2    2
    2  3    3    3
    3  4    4    4
    4  5    5    5
    5  1  100 -100
    6  2    2    2
    7  3    3    3
    8  4    4  100
    9  5    5    5
    >>> df.n_outliers()
       n_outliers  p_outliers
    x           0         0.0
    y           1         0.1
    z           2         0.2
    '''
    n_outliers = df.select_dtypes('number').apply(lambda s: s.outliers(**kwargs).sum())
    p_outliers = n_outliers / df.shape[0]
    return pd.DataFrame(dict(n_outliers=n_outliers, p_outliers=p_outliers))


def outliers(s: Series, how='iqr', k=1.5, std_cutoff=2) -> Series:
    '''
    Detect outliers in the series.

    Returns
    -------

    A pandas Series of boolean values indicating whether each point is an
    outlier or not.

    Parameters
    ----------

    how : {'iqr', 'std'}, default 'iqr'
        - 'iqr' : identify outliers based on whether they are > q3 + k * iqr
          or < q1 - k * iqr
        - 'std' : identify outliers based on whether they are further than a
                  specified number of standard deviations from the mean
    k : value to multiply the iqr by for outlier detection. Ignored when
        how='std'. Default 1.5
    std_cutoff : cutoff for identifying an outlier based on standard deviation.
                 Ignored when how='iqr'. Default 2

    Example
    -------

    >>> df = pd.DataFrame(dict(x=[1, 2, 3, 4, 5, 6, 100],
    ...                        y=[-100, 5, 3, 4, 1, 2, 0]))
    >>> df
         x    y
    0    1 -100
    1    2    5
    2    3    3
    3    4    4
    4    5    1
    5    6    2
    6  100    0
    >>> df.x.outliers()
    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    Name: x, dtype: bool
    >>> df[df.x.outliers()]
         x  y
    6  100  0
    '''
    if how == 'iqr':
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - k * iqr) | (s > q3 + k * iqr)
    elif how == 'std':
        return zscore(s).abs() > std_cutoff
    raise ValueError('how must be one of {iqr,std}')

def nnull(df: DataFrame, axis=0) -> DataFrame:
    '''
    Provide a summary of null values in each column.

    alias of nna

    Example
    -------

    >>> df = pd.DataFrame(dict(x=[1, 2, np.nan], y=[4, np.nan, np.nan]))
    >>> df
         x    y
    0  1.0  4.0
    1  2.0  NaN
    2  NaN  NaN
    >>> nulls_by_column = df.nnull()
    >>> nulls_by_column
       n_missing  p_missing
    x          1   0.333333
    y          2   0.666667
    >>> nulls_by_row = df.nnull(axis=1)
    >>> nulls_by_row
       n_missing  p_missing
    0          0        0.0
    1          1        0.5
    2          2        1.0
    '''
    n_missing = df.isnull().sum(axis=axis)
    p_missing = n_missing / df.shape[axis]
    return pd.DataFrame(dict(n_missing=n_missing, p_missing=p_missing))

def unnest(df: DataFrame, col: str, split=True, sep=',', reset_index=True) -> DataFrame:
    '''
    Turns a column with multiple values in each row in it into separate rows,
    each with a single value.

    Parameters
    ----------

    - col : name of the column to unnest
    - split : default True. whether or not to split the data in the column. Set
              this to False if the column already contains lists in each row
    - sep : separator to split on. Ignored if split=False
    - reset_index : default True. whether to reset the index in the resulting
                    data frame. If False, the resulting data frame will have an
                    index that could contain duplicates.

    Example
    -------

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

    Any additional kwargs are passed to ``seaborn.heatmap`` and the resulting
    axes object is returned.

    >>> x = np.arange(0, 10)
    >>> y = x / 2
    >>> df = pd.DataFrame(dict(x=x, y=y))
    >>> df.correlation_heatmap()
    <matplotlib.axes._subplots.AxesSubplot object at ...>
    '''
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cm.coolwarm_r
        # kwargs['cmap'] = cm.PiYG

    if not fancy:
        return sns.heatmap(df.corr(), center=0, annot=True, **kwargs)

    cmat = df.corr()

    sns.set(style="white")
    mask = np.zeros_like(cmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(12, 12))
    sns.heatmap(cmat, mask=mask, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, **kwargs)
    plt.yticks(rotation=0)
    return plt

def hdtl(df: DataFrame, n=3) -> DataFrame:
    '''
    Return the head and the tail of the data frame.

    Parameters
    ----------

    - n : number of rows to get from both the head and tail

    Example
    -------

    >>> df = pd.DataFrame(dict(x=np.arange(10), y=np.arange(10)))
    >>> df
       x  y
    0  0  0
    1  1  1
    2  2  2
    3  3  3
    4  4  4
    5  5  5
    6  6  6
    7  7  7
    8  8  8
    9  9  9
    >>> df.hdtl(1)
       x  y
    0  0  0
    9  9  9
    >>> df.hdtl()
       x  y
    0  0  0
    1  1  1
    2  2  2
    7  7  7
    8  8  8
    9  9  9
    '''
    return pd.concat([df.head(n), df.tail(n)])

def log(s: Series):
    '''
    Returns the log base 10 of the values in the series using np.log10

    Example
    -------

    >>> x = pd.Series([1, 10, 100, 1000])
    >>> x
    0       1
    1      10
    2     100
    3    1000
    dtype: int64
    >>> x.log()
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

    Example
    -------

    >>> x = pd.Series([1, np.e, np.e ** 2, np.e ** 3])
    >>> x
    0     1.000000
    1     2.718282
    2     7.389056
    3    20.085537
    dtype: float64
    >>> x.ln()
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

    >>> x = pd.Series([1, 2,4, 8, 16])
    >>> x
    0     1
    1     2
    2     4
    3     8
    4    16
    dtype: int64
    >>> x.log2()
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

    - rows : the name of the columns that will make up the rows in resulting
             contingency table
    - cols : the name of the columns that will make up the columns in resulting
             contingency table
    - values : (optional) name of the column to use for the cell values in the
               resulting contingency table. If supplied, aggfunc must be
               provided as well. See ``pd.crosstab`` for more details.
    - kwargs : any additional key word arguments to pass along to
               ``pd.crosstab``

    Example
    -------

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
    Runs a 1 sample t-test comparing the specified target variable to the
    overall mean among all of the possible subgroups.

    Parameters
    ----------

    - target : name of the column that holds the target (continuous) variable

    Example
    -------

    >>> from seaborn import load_dataset
    >>> tips = load_dataset('tips')
    >>> tips = tips[['total_bill', 'day', 'time']]
    >>> tips.ttest('total_bill')
                     statistic    pvalue    n
    variable value
    day      Sun      1.603035  0.113130   76
             Sat      0.644856  0.520737   87
             Thur    -2.099957  0.039876   62
             Fri     -1.383042  0.183569   19
    time     Dinner   1.467432  0.144054  176
             Lunch   -2.797882  0.006710   68
    '''
    results = []
    for col in df.drop(columns=target):
        unique_vals = df[col].unique()
        ttests = DataFrame([ttest_1samp(df[df[col] == v][target],
                                        df[target].mean())
                            for v in unique_vals])
        ns = [df[df[col] == v].shape[0] for v in unique_vals]
        ttests = ttests.assign(n=ns, value=unique_vals, variable=col)
        results.append(ttests)
    return pd.concat(results, axis=0).set_index(['variable', 'value'])

def ttest_2samp(df: DataFrame, target: str) -> DataFrame:
    '''
    Runs a 2 sample t-test comparing the specified target variable for every
    unique value from every other column in the data frame.

    The resulting t-statistic and pvalue are based on subdividing the data for
    each unique value for each column, with each individual value indicating
    that the test was performed based on belonging to that unique value vs not
    belonging to that group.

    Parameters
    ----------

    - target : name of the column that holds the target (continuous) variable

    Example
    -------

    >>> from seaborn import load_dataset
    >>> tips = load_dataset('tips')
    >>> tips = tips[['total_bill', 'day', 'time']]
    >>> tips.ttest_2samp('total_bill')
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

def chi2(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    '''
    Performs a chi squared contingency table test between all the combinations
    of two columns in the data frame.

    Returns
    -------

    (pvals, chi2s)

    A tuple with two data frames, each which have all of the columns from the
    original data frame as both the indexes and the columns. The values in the
    first are the p-values, the values in the second are the chi square test
    statistics.

    Example
    -------

    >>> from seaborn import load_dataset
    >>> tips = load_dataset('tips')
    >>> p_vals, chi2s = tips[['smoker', 'time', 'day']].chi2()
    >>> p_vals
                 smoker        time          day
    smoker          NaN    0.477149  1.05676e-05
    time       0.477149         NaN   8.4499e-47
    day     1.05676e-05  8.4499e-47          NaN
    >>> chi2s
              smoker      time      day
    smoker       NaN  0.505373  25.7872
    time    0.505373       NaN  217.113
    day      25.7872   217.113      NaN
    '''
    p_vals = pd.DataFrame(index=df.columns, columns=df.columns)
    chi2s = p_vals.copy()
    for x1, x2 in it.combinations(df.columns, 2):
        stat, p, *_ = chi2_contingency(df.xtab(x1, x2))
        p_vals.loc[x1, x2] = p
        p_vals.loc[x2, x1] = p
        chi2s.loc[x1, x2] = stat
        chi2s.loc[x2, x1] = stat
    return p_vals, chi2s

def pipe(df: DataFrame, fn: Callable):
    return df.pipe(fn)

pd.Series.bin = cut
pd.Series.cut = cut
pd.Series.get_scaler = get_scaler
pd.Series.ln = ln
pd.Series.log2 = log2
pd.Series.log = log
pd.Series.outliers = outliers
pd.Series.zscore = zscore

pd.DataFrame.cleanup_column_names = cleanup_column_names
pd.DataFrame.chi2 = chi2
pd.DataFrame.correlation_heatmap = correlation_heatmap
pd.DataFrame.crosstab = crosstab
pd.DataFrame.drop_outliers = drop_outliers
pd.DataFrame.get_scalers = get_scalers
pd.DataFrame.hdtl = hdtl
pd.DataFrame.__lshift__ = pipe
pd.DataFrame.n_outliers = n_outliers
pd.DataFrame.nna = nnull
pd.DataFrame.nnull = nnull
pd.DataFrame.__rshift__ = pipe
pd.DataFrame.ttest_2samp = ttest_2samp
pd.DataFrame.ttest = ttest
pd.DataFrame.unnest = unnest
pd.DataFrame.xtab = crosstab

series_extensions = [
    cut,
    get_scaler,
    ln,
    log,
    log2,
    outliers,
    zscore,
]

data_frame_extensions = [
    chi2,
    cleanup_column_names,
    correlation_heatmap,
    crosstab,
    drop_outliers,
    get_scalers,
    hdtl,
    nnull,
    n_outliers,
    ttest,
    ttest_2samp,
    unnest,
]
