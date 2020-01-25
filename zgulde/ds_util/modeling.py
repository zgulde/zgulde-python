import math
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def model(X, y, m, **kwargs):
    return m(**kwargs).fit(X, y)


def residuals(model, X, y):
    return y - model.predict(X)


def tree(df, rformula, **kwargs):
    X, y = df.rformula(rformula)
    if np.issubdtype(y.dtype, np.number):
        return dtree_regressor(X, y, **kwargs)
    else:
        y = y.astype("category").cat.codes
        return dtree_classifier(X, y, **kwargs)


def lr(df, rformula, **kwargs):
    X, y = df.rformula(rformula)
    if np.issubdtype(y.dtype, np.number):
        return linear_regression(X, y, **kwargs)
    else:
        y = y.astype("category").cat.codes
        return logistic_regression(X, y, **kwargs)


def logistic_regression(X, y, **kwargs):
    lr = LogisticRegression(**kwargs).fit(X, y)
    yhat = lr.predict(X)
    coefs = pd.Series(dict(zip(X.columns, lr.coef_[0])))
    return {
        "model": lr,
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, average=None),
        "recall": recall_score(y, yhat, average=None),
        "coef": coefs.sort_values(),
    }


def dtree_classifier(X, y, **kwargs):
    tree = DecisionTreeClassifier(**kwargs).fit(X, y)
    yhat = tree.predict(X)
    features = pd.Series(dict(zip(X.columns, tree.feature_importances_)))
    return {
        "model": tree,
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, average=None),
        "recall": recall_score(y, yhat, average=None),
        "feature_importances": features.sort_values(),
    }


def dtree_regressor(X, y, **kwargs):
    tree = DecisionTreeRegressor(**kwargs).fit(X, y)
    yhat = tree.predict(X)
    mse = mean_squared_error(y, yhat)
    features = pd.Series(dict(zip(X.columns, tree.feature_importances_)))
    return {
        "model": tree,
        "r2": r2_score(y, yhat),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y, yhat),
        "feature_importances": features.sort_values(),
    }


def linear_regression(X, y, **kwargs):
    lr = LinearRegression(**kwargs).fit(X, y)
    yhat = lr.predict(X)
    mse = mean_squared_error(y, yhat)
    coefs = pd.Series(dict(zip(X.columns, lr.coef_)))
    return {
        "model": lr,
        "r2": r2_score(y, yhat),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y, yhat),
        "coef": coefs.sort_values(),
    }


def cv_results_to_df(grid: GridSearchCV) -> pd.DataFrame:
    """
    Presents GridSearchCV results as a data frame

    >>> from pydataset import data
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> iris = data('iris')
    >>> X, y = iris[['Petal.Length', 'Sepal.Length']], iris.Species
    >>> params = {'max_depth': [3, 4, 6], 'n_estimators': [6, 12]}
    >>> algo = RandomForestClassifier(random_state=123)
    >>> grid = GridSearchCV(algo, params, cv=8, iid=False).fit(X, y)
    >>> cv_results_to_df(grid)
       max_depth  n_estimators     score                   model
    0          3             6  0.946429  RandomForestClassifier
    1          3            12  0.932540  RandomForestClassifier
    2          4             6  0.946429  RandomForestClassifier
    3          4            12  0.939484  RandomForestClassifier
    4          6             6  0.952381  RandomForestClassifier
    5          6            12  0.945437  RandomForestClassifier
    """
    results = grid.cv_results_
    params_and_scores = zip(results["params"], results["mean_test_score"])
    df = pd.DataFrame([dict(**p, score=s) for p, s in params_and_scores])
    df["model"] = grid.best_estimator_.__class__.__name__
    return df


GridCandidates = List[
    Tuple[sklearn.base.BaseEstimator, Dict[str, List[Union[str, int]]]]
]


def multi_grid_search(models: GridCandidates, X, y, cv=4, **kwargs) -> pd.DataFrame:
    """
    combine grid searches for multiple models

    kwargs are passed along to GridSearchCV

    >>> from sklearn.linear_model import Ridge, LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from pydataset import data
    >>> tips = data('tips')
    >>> X, y = tips[['total_bill', 'size']], tips.tip
    >>> models = []
    >>> models += [(Ridge(), {'alpha': [.1, 1, 10]})]
    >>> models += [(RandomForestRegressor(random_state=123), {'max_depth': range(3, 5), 'n_estimators': [3, 7]})]
    >>> models += [(LinearRegression(), {})]
    >>> multi_grid_search(models, X, y)
       alpha     score                  model  max_depth  n_estimators
    0   10.0  0.458305                  Ridge        NaN           NaN
    1    1.0  0.457351                  Ridge        NaN           NaN
    2    0.1  0.457235                  Ridge        NaN           NaN
    3    NaN  0.457221       LinearRegression        NaN           NaN
    4    NaN  0.325329  RandomForestRegressor        3.0           7.0
    5    NaN  0.294933  RandomForestRegressor        4.0           7.0
    6    NaN  0.272333  RandomForestRegressor        3.0           3.0
    7    NaN  0.254083  RandomForestRegressor        4.0           3.0
    """
    return (
        pd.concat(
            [
                cv_results_to_df(
                    GridSearchCV(model, params, cv=cv, n_jobs=-1, **kwargs).fit(X, y)
                )
                for model, params in models
            ],
            sort=False,
        )
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )


def inspect_coefs(lm, X):
    """
    View a summary of a linear model's coefficients.

    Works for both linear and logistic regression models.

    >>> import pydataset
    >>> iris = pydataset.data('iris')
    >>> X, y = iris[['Sepal.Length', 'Sepal.Width']], iris.Species
    >>> lm = LogisticRegression(multi_class='auto', solver='lbfgs', random_state=123)
    >>> lm = lm.fit(X, y)
    >>> inspect_coefs(lm, X)
                Sepal.Length  Sepal.Width
    setosa         -2.708902     2.324024
    versicolor      0.612733    -1.570588
    virginica       2.096170    -0.753436

    >>> tips = pydataset.data('tips')
    >>> X, y = tips[['size', 'total_bill']], tips['tip']
    >>> lm = LinearRegression().fit(X, y)
    >>> inspect_coefs(lm, X)
    total_bill    0.092713
    size          0.192598
    dtype: float64
    """
    coef = lm.coef_
    if len(coef.shape) > 1:
        return pd.DataFrame(coef, index=lm.classes_, columns=X.columns)
    else:
        return pd.Series(dict(zip(X.columns, lm.coef_.ravel()))).sort_values()


def inspect_feature_importances(tree, X):
    return pd.Series(dict(zip(X.columns, tree.feature_importances_))).sort_values()


classification_models = [
    (DecisionTreeClassifier(), {"max_depth": range(1, 11)}),
    (KNeighborsClassifier(), {"n_neighbors": range(1, 11)}),
    (
        LogisticRegression(solver="lbfgs", multi_class="auto"),
        {"C": [0.01, 0.1, 1, 10, 100, 1000]},
    ),
]

regression_models = [
    (DecisionTreeRegressor(), {"max_depth": range(1, 11)}),
    (KNeighborsRegressor(), {"n_neighbors": range(1, 11)}),
    (LinearRegression(), {}),
    (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100, 1000]}),
]
