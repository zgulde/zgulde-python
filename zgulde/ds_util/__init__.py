import os
from functools import wraps

import graphviz
import pandas as pd
from pydataset import data
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import zgulde.ds_util.plotting as zplot
from zgulde.ds_util.modeling import *

# mode aggregation
# tips.groupby('size').agg(stats.mode).apply(lambda s: s.apply(lambda v: v[0][0]))


def with_caching(filename="data.csv", index=False, with_prints=True):
    """
    Decorator for wrapping any function that returns a dataframe so that it
    saves the dataframe to a local file and on subsequent calls reads from the
    local file. Useful for data acquisition that can take a while.

    Parameters
    ----------
    - filename: filename for the cache file, e.g. data.csv
    - index: whether or not to include the dataframe index
    - with_prints: booleean indicating whether to print information about
      whether the data is coming from the original function or the cache file

    Examples
    --------

    >>> @with_caching('demo.csv')
    ... def get_some_data():
    ...     return pd.DataFrame(dict(x=[1, 2, 3], y=list('abc')))
    >>> os.unlink('demo.csv') if os.path.exists('demo.csv') else None # remove the cache file if it exists
    >>> get_some_data()
    Reading data from get_some_data
       x  y
    0  1  a
    1  2  b
    2  3  c
    >>> get_some_data()
    Reading from cached csv file
       x  y
    0  1  a
    1  2  b
    2  3  c
    """

    def wrapper(fn):
        @wraps(fn)
        def f(*args, **kwargs):
            if os.path.exists(filename):
                if with_prints:
                    print("Reading from cached csv file")
                return pd.read_csv(filename)
            if with_prints:
                print(f"Reading data from {fn.__name__}")
            df = fn(*args, **kwargs)
            df.to_csv(filename, index=index)
            return df

        return f

    return wrapper


# better interaction w/ sklearn.metrics.confusion_matrix, e.g. auto labelling
def better_confusion_matrix(actual, predicted, labels=None):
    cm = pd.crosstab(actual, predicted)
    if labels is not None:
        cm.index = pd.Index(labels, name="Actual")
        cm.columns = pd.Index(labels, name="Predictions")
    return cm


def labelled_confusion_matrix():
    matrix = [
        ["TP", "FN (Type II Error)", "Recall = TP / (TP + FN)"],
        ["FP (Type I Error)", "TN", ""],
        ["Precision = TP / (TP + FP)", "", ""],
    ]
    return pd.DataFrame(
        matrix,
        columns=["Predicted +", "Predicted -", ""],
        index=["Actual +", "Actual -", ""],
    )


def rdatasets():
    datasets = data()
    shapes = (
        datasets.apply(lambda row: data(row.dataset_id).shape, axis=1)
        .apply(pd.Series)
        .rename({0: "nrows", 1: "ncols"}, axis=1)
    )
    return pd.concat([datasets, shapes], axis=1)


# https://stackoverflow.com/questions/50559078/generating-random-dates-within-a-given-range-in-pandas
def rand_dates(start, end, n):
    start_u = pd.to_datetime(start).value // 10**9
    end_u = pd.to_datetime(end).value // 10**9
    return pd.DatetimeIndex(
        (10**9 * np.random.randint(start_u, end_u, n)).view("M8[ns]")
    )


def viz_dtree(X, y, **kwargs):
    tree = DecisionTreeClassifier(**kwargs).fit(X, y)
    feature_names = X.columns
    class_names = sorted(y.unique())
    dot = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,  # target value names
        special_characters=True,
        filled=True,  # fill nodes w/ informative colors
        impurity=False,  # show impurity at each node
        leaves_parallel=True,  # all leaves at the bottom
        proportion=True,  # show percentages instead of numbers at each leaf
        rotate=True,  # left to right instead of top-bottom
        rounded=True,  # rounded boxes and sans-serif font
    )
    graph = graphviz.Source(dot, filename="dtree", format="png")
    graph.view(cleanup=True)


def to_gsheet(
    df: pd.DataFrame, sheet_name: str, worksheet_name: str, include_index=True
):
    import gspread
    from gspread_dataframe import get_as_dataframe, set_with_dataframe

    gc = gspread.oauth()
    sh = gc.open(sheet_name)
    try:
        sh.add_worksheet(worksheet_name, 100, 100)
    except:
        pass  # sheet already exists
    worksheet = sh.worksheet(worksheet_name)
    set_with_dataframe(worksheet, df, include_index=include_index)


def from_gsheet(sheet_name: str, worksheet_name: str):
    """
    read a worksheet (worksheet_name) from a google sheets spreadsheet (sheet_name)
    """
    import gspread
    from gspread_dataframe import get_as_dataframe, set_with_dataframe

    gc = gspread.oauth()
    sh = gc.open(sheet_name)
    worksheet = sh.worksheet(worksheet_name)

    return get_as_dataframe(worksheet)
