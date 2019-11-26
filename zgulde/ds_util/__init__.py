import pandas as pd
from pydataset import data

from .plotting import *
from .modeling import *

# mode aggregation
# tips.groupby('size').agg(stats.mode).apply(lambda s: s.apply(lambda v: v[0][0]))

# better interaction w/ sklearn.metrics.confusion_matrix, e.g. auto labelling
def better_confusion_matrix(actual, predicted, labels=None):
    cm = pd.crosstab(actual, predicted)
    if labels is not None:
        cm.index = pd.Index(labels, name='Actual')
        cm.columns = pd.Index(labels, name='Predictions')
    return cm


def labelled_confusion_matrix():
    matrix = [['TP',                         'FN (Type II Error)', 'Recall = TP / (TP + FN)'],
              ['FP (Type I Error)',          'TN',                 ''],
              ['Precision = TP / (TP + FP)', '',                   '']]
    return pd.DataFrame(matrix,
                        columns=['Predicted +', 'Predicted -', ''],
                        index=['Actual +', 'Actual -', ''])

def rdatasets():
    datasets = data()
    shapes = datasets.apply(lambda row: data(row.dataset_id).shape, axis=1)\
                     .apply(pd.Series)\
                     .rename({0: 'nrows', 1: 'ncols'}, axis=1)
    return pd.concat([datasets, shapes], axis=1)

class MyRange:
    """
    A simple class that defines a convenient (if not conventional) range syntax

    >>> r = MyRange()
    >>> r[1:10]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> r[1:10:2]
    [1, 3, 5, 7, 9]
    """
    def __getitem__(_, aslice: slice):
        start = aslice.start if aslice.start is not None else 0
        step = aslice.step if aslice.step is not None else 1
        stop = aslice.stop + 1 if aslice.stop is not None else 1
        return list(range(start, stop, step))

# https://stackoverflow.com/questions/50559078/generating-random-dates-within-a-given-range-in-pandas
def rand_dates(start, end, n):
    start_u = pd.to_datetime(start).value // 10**9
    end_u = pd.to_datetime(end).value // 10**9
    return pd.DatetimeIndex((10**9*np.random.randint(start_u, end_u, n)).view('M8[ns]'))

import graphviz
from sklearn.tree import export_graphviz, DecisionTreeClassifier

def viz_dtree(X, y, **kwargs):
    tree = DecisionTreeClassifier(**kwargs).fit(X, y)
    feature_names = X.columns
    class_names = sorted(y.unique())
    dot = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names, # target value names
        special_characters=True,
        filled=True,             # fill nodes w/ informative colors
        impurity=False,          # show impurity at each node
        leaves_parallel=True,    # all leaves at the bottom
        proportion=True,         # show percentages instead of numbers at each leaf
        rotate=True,             # left to right instead of top-bottom
        rounded=True,            # rounded boxes and sans-serif font
    )
    graph = graphviz.Source(dot, filename='dtree', format='png')
    graph.view(cleanup=True)

