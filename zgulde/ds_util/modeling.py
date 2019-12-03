import math
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Union

import sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

def residuals(model, X, y):
    return y - model.predict(X)

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


def cv_results_to_df(grid: GridSearchCV) -> pd.DataFrame:
    results = grid.cv_results_
    params_and_scores = zip(results["params"], results["mean_test_score"])
    df = pd.DataFrame([dict(**p, score=s) for p, s in params_and_scores])
    df["model"] = grid.best_estimator_.__class__.__name__
    return df


GridCandidates = List[
    Tuple[sklearn.base.BaseEstimator, Dict[str, List[Union[str, int]]]]
]


def multi_grid_search(models: GridCandidates, X, y, cv=4) -> pd.DataFrame:
    return pd.concat([
        cv_results_to_df(GridSearchCV(model, params, cv=cv).fit(X, y))
        for model, params in models
    ])

def inspect_coefs(lm, X):
    coef = lm.coef_
    if coef.shape[0] > 1:
        return pd.DataFrame(coef, index=lm.classes_, columns=X.columns)
    else:
        return pd.Series(dict(zip(X.columns, lm.coef_.ravel()))).sort_values()

def inspect_feature_importances(tree, X):
    return pd.Series(dict(zip(X.columns, tree.feature_importances_))).sort_values()


classification_models = [
    (DecisionTreeClassifier(), {"max_depth": range(1, 11)}),
    (KNeighborsClassifier(), {"n_neighbors": range(1, 11)}),
    (LogisticRegression(), {"C": [0.01, 0.1, 1, 10, 100, 1000], "solver": ["lbfgs"]}),
    (SVC(), {"kernel": ["rbf", "linear"]}),
    (SVC(), {"kernel": ["poly"], "degree": [2]}),
]

regression_models = [
    (DecisionTreeRegressor(), {"max_depth": range(1, 11)}),
    (KNeighborsRegressor(), {"n_neighbors": range(1, 11)}),
    (LinearRegression(), {}),
    (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100, 1000]}),
    (SVR(), {"kernel": ["rbf", "linear"]}),
    (SVR(), {"kernel": ["poly"], "degree": [2]}),
]

