import pandas as pd
from pydataset import data

def plot_dual_axis(df, x):
    '''
    plot_dual_axis(df, 'x')('y1')('y2')
    '''
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    def plot_y2(y, *args, **kwargs):
        ax2.plot(df[x], df[y], *args, **kwargs)
        return fig

    def plot_y1(y, *args, **kwargs):
        ax1.plot(df[x], df[y], *args, **kwargs)
        return plot_y2

    return plot_y1

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

import math
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV

def tree(df, rformula, **kwargs):
    X, y = df.rformula(rformula)
    if np.issubdtype(y.dtype, np.number):
        return dtree_regressor(X, y, **kwargs)
    else:
        y = y.astype('category').cat.codes
        return dtree_classifier(X, y, **kwargs)

def lr(df, rformula, **kwargs):
    X, y = df.rformula(rformula)
    if np.issubdtype(y.dtype, np.number):
        return linear_regression(X, y, **kwargs)
    else:
        y = y.astype('category').cat.codes
        return logistic_regression(X, y, **kwargs)

def logistic_regression(X, y, **kwargs):
    lr = LogisticRegression(**kwargs).fit(X, y)
    yhat = lr.predict(X)
    coefs = pd.Series(dict(zip(X.columns, lr.coef_[0])))
    return {
        'model': lr,
        'accuracy': accuracy_score(y, yhat),
        'precision': precision_score(y, yhat, average=None),
        'recall': recall_score(y, yhat, average=None),
        'coef': coefs.sort_values()
    }

def dtree_classifier(X, y, **kwargs):
    tree = DecisionTreeClassifier(**kwargs).fit(X, y)
    yhat = tree.predict(X)
    features = pd.Series(dict(zip(X.columns, tree.feature_importances_)))
    return {
        'model': tree,
        'accuracy': accuracy_score(y, yhat),
        'precision': precision_score(y, yhat, average=None),
        'recall': recall_score(y, yhat, average=None),
        'feature_importances': features.sort_values()
    }

def dtree_regressor(X, y, **kwargs):
    tree = DecisionTreeRegressor(**kwargs).fit(X, y)
    yhat = tree.predict(X)
    mse = mean_squared_error(y, yhat)
    features = pd.Series(dict(zip(X.columns, tree.feature_importances_)))
    return {
        'model': tree,
        'r2': r2_score(y, yhat),
        'mse': mse,
        'rmse': math.sqrt(mse),
        'mae': mean_absolute_error(y, yhat),
        'feature_importances': features.sort_values(),
    }

def linear_regression(X, y, **kwargs):
    lr = LinearRegression(**kwargs).fit(X, y)
    yhat = lr.predict(X)
    mse = mean_squared_error(y, yhat)
    coefs = pd.Series(dict(zip(X.columns, lr.coef_)))
    return {
        'model': lr,
        'r2': r2_score(y, yhat),
        'mse': mse,
        'rmse': math.sqrt(mse),
        'mae': mean_absolute_error(y, yhat),
        'coef': coefs.sort_values(),
    }
