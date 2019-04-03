import pandas as pd

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
