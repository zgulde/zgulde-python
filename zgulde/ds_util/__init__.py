import graphviz
import pandas as pd
from pydataset import data
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import zgulde.ds_util.plotting as zplot
from zgulde.ds_util.modeling import *

# mode aggregation
# tips.groupby('size').agg(stats.mode).apply(lambda s: s.apply(lambda v: v[0][0]))

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
    start_u = pd.to_datetime(start).value // 10 ** 9
    end_u = pd.to_datetime(end).value // 10 ** 9
    return pd.DatetimeIndex(
        (10 ** 9 * np.random.randint(start_u, end_u, n)).view("M8[ns]")
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
