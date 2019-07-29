import pandas as pd
from pydataset import data

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

def datasets():
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
        stop = aslice.stop + 1
        return list(range(start, stop, step))

