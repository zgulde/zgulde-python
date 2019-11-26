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
